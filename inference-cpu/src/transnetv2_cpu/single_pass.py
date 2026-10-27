"""One FP32 inference per original window, without an INT8 pilot or refinement."""
from pathlib import Path
import os
import shutil
import threading
import time
import numpy as np
from .engine import GuardedVOD
from .models import FP32_POLICY, load_manifest, require_supported_platform, sha256


class SinglePassVOD(GuardedVOD):
    def __init__(self, manifest, threads=2, progress=None, capture_windows=False):
        require_supported_platform()
        if threads < 1:
            raise ValueError("threads must be positive")
        import openvino as ov
        self.manifest = load_manifest(manifest, policy=FP32_POLICY)
        core = ov.Core()
        config = dict(INFERENCE_PRECISION_HINT="f32", INFERENCE_NUM_THREADS=threads,
                      NUM_STREAMS=1, ENABLE_CPU_PINNING=False)
        self.reference = core.compile_model(str(Path(manifest).resolve().parent/self.manifest["model"]), "CPU", config).create_infer_request()
        self.ov = ov
        self.progress = progress or (lambda event: None)
        self.capture_windows = capture_windows
        self.lease = threading.Lock()

    def _predict(self, spool, root):
        root.mkdir(parents=True, exist_ok=False)
        count, emitted, total_wait, total_infer = 0, 0, 0.0, 0.0
        capture = (root/"windows.f32").open("xb") if self.capture_windows else None
        try:
            with (root/"head0.part").open("xb") as head0, (root/"head1.part").open("xb") as head1:
                while True:
                    wait_started = time.monotonic()
                    spool.await_frames(count*950+975)
                    waited = time.monotonic()-wait_started
                    total_wait += waited
                    if spool.eof and count*950 >= spool.frames:
                        break
                    window = spool.window(count)
                    started = time.monotonic()
                    values = self.infer(self.reference, window)
                    seconds = time.monotonic()-started
                    total_infer += seconds
                    if capture is not None:
                        capture.write(values.astype("<f4").tobytes())
                    length = min(950, spool.frames-count*950)
                    head0.write(values[0,25:25+length].astype("<f4").tobytes())
                    head1.write(values[1,25:25+length].astype("<f4").tobytes())
                    emitted += length
                    count += 1
                    self.progress(dict(type="progress", phase="single-pass", window=count,
                                       frames=spool.frames, processedFrames=emitted, emittedFrames=0,
                                       windowWaitSeconds=waited, inferenceStartedMonotonic=started,
                                       inferenceSeconds=seconds, windowFinishedMonotonic=time.monotonic()))
        finally:
            if capture is not None:
                capture.close()
        n = spool.frames
        if not spool.eof or emitted != n or count != (n+949)//950:
            raise ValueError("Incomplete single-pass input")
        temp = root/"logits.f32.part"
        with temp.open("xb") as output:
            for name in ["head0.part", "head1.part"]:
                with (root/name).open("rb") as source:
                    shutil.copyfileobj(source, output, 1024*1024)
            output.flush()
            os.fsync(output.fileno())
        temp.replace(root/"logits.f32")
        for name in ["head0.part", "head1.part"]:
            (root/name).unlink()
        done = time.monotonic()
        result = dict(protocolVersion=1, policyId=FP32_POLICY["id"], frames=n, heads=2,
                      dtype="float32-le", layout="heads-frames", logitsFile="logits.f32",
                      logitsSha256=sha256(root/"logits.f32"), routing="single-pass-fp32",
                      fastExecutedFrames=0, pilotExecutedFrames=0, refinementExecutedFrames=0,
                      referenceExecutedFrames=count*1000, windows=count,
                      inputWaitSeconds=total_wait, inferenceSeconds=total_infer,
                      inputEofMonotonic=spool.eof_monotonic, logitsReadyMonotonic=done,
                      eofToLogitsSeconds=max(0.0, done-spool.eof_monotonic),
                      phases=[dict(phase="single-pass", crops=count, executedFrames=count*1000)])
        self.progress(dict(type="progress", phase="complete", frames=n, emittedFrames=n))
        return result
