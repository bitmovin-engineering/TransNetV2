"""One VOD per process: raw RGB stdin, bounded JSON progress, atomic result metadata.

The caller owns the work directory. On cancellation it must close stdin, terminate
and observe worker exit before removing files. No FFmpeg or network I/O is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import resource
import time
from .engine import GuardedVOD
from .frames import FrameSpool
from .models import POLICY_ID


def event(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--input-rgb", type=Path, help="Use an existing raw RGB file instead of stdin")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-spool-bytes", type=int, default=8*1024**3)
    args = parser.parse_args()
    try:
        started = time.monotonic()
        engine = GuardedVOD(args.manifest, threads=args.threads, progress=event)
        model_load_seconds = time.monotonic()-started
        args.work_dir.mkdir(parents=True, exist_ok=False)
        event(dict(type="ready", protocolVersion=1, policyId=POLICY_ID))
        spool = FrameSpool(args.input_rgb or args.work_dir/"frames.rgb",
                           stream=None if args.input_rgb else sys.stdin.buffer,
                           max_bytes=args.max_spool_bytes)
        result = engine.predict(spool, args.work_dir/"output")
        spool.close()
        result.update(inputSha256=spool.digest.hexdigest(), modelLoadSeconds=model_load_seconds,
                      elapsedSeconds=time.monotonic()-started, peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        temporary = args.work_dir/"result.json.part"
        with temporary.open("x") as stream:
            json.dump(result, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.work_dir/"result.json")
        event(dict(type="result", **result))
    except Exception as error:
        # No partial result marker is published. A daemon input reader cannot delay
        # process exit after a main-thread inference or model failure.
        event(dict(type="error", errorType=type(error).__name__, message=str(error)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
