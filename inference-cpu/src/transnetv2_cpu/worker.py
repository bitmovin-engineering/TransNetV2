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
from .models import POLICY_ID, FP32_POLICY_ID
from .single_pass import SinglePassVOD


def event(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--input-rgb", type=Path, help="Use an existing raw RGB file instead of stdin")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-spool-bytes", type=int, default=8*1024**3)
    parser.add_argument("--events", type=Path, help="Optional new JSONL diagnostics file, owned by caller")
    parser.add_argument("--capture-windows", action="store_true", help="Retain full-window logits for quality evidence")
    args = parser.parse_args()
    diagnostics = None
    try:
        diagnostics = args.events.open("x") if args.events else None
        def report(value):
            value = dict(value, monotonicSeconds=time.monotonic())
            event(value)
            if diagnostics is not None:
                diagnostics.write(json.dumps(value, separators=(",", ":"))+"\n")
                diagnostics.flush()
        started = time.monotonic()
        policy_id = json.loads(args.manifest.read_text())["policy"]["id"]
        if policy_id == FP32_POLICY_ID:
            engine = SinglePassVOD(args.manifest, threads=args.threads, progress=report,
                                   capture_windows=args.capture_windows)
        elif policy_id == POLICY_ID:
            engine = GuardedVOD(args.manifest, threads=args.threads, progress=report)
        else:
            raise ValueError("Unsupported CPU precision policy")
        model_load_seconds = time.monotonic()-started
        args.work_dir.mkdir(parents=True, exist_ok=False)
        report(dict(type="ready", protocolVersion=1, policyId=policy_id, modelLoadSeconds=model_load_seconds))
        spool = FrameSpool(args.input_rgb or args.work_dir/"frames.rgb",
                           stream=None if args.input_rgb else sys.stdin.buffer,
                           max_bytes=args.max_spool_bytes)
        result = engine.predict(spool, args.work_dir/"output")
        spool.close()
        result.update(inputSha256=spool.digest.hexdigest(), modelLoadSeconds=model_load_seconds,
                      elapsedSeconds=time.monotonic()-started, peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      processCpuSeconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
        temporary = args.work_dir/"result.json.part"
        with temporary.open("x") as stream:
            json.dump(result, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.work_dir/"result.json")
        report(dict(type="result", **result))
    except Exception as error:
        # No partial result marker is published. A daemon input reader cannot delay
        # process exit after a main-thread inference or model failure.
        event(dict(type="error", errorType=type(error).__name__, message=str(error)))
        return 1
    finally:
        if diagnostics is not None:
            diagnostics.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
