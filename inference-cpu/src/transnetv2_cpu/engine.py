"""Disk-backed implementation of aisa-vod-v1. One engine lease owns one video.

The INT8 guard is empirical, not a certified bound. Policy changes require a new
ID, independent consumer-quality qualification, and a new timing distribution.
"""
from pathlib import Path
import threading
import numpy as np
from .models import POLICY, load_manifest, require_supported_platform, sha256


def runs(mask):
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def merged_stream(intervals):
    pending = None
    for start, end in intervals:
        if pending is not None and start <= pending[1]:
            pending = (pending[0], max(end, pending[1]))
        else:
            if pending is not None:
                yield pending
            pending = (start, end)
    if pending is not None:
        yield pending


def hard_regions(scores, n, chunk=65536):
    opened = None
    for start in range(0, n, chunk):
        mask = np.any(np.abs(scores[:, start:start+chunk]) <= 1.5, axis=0)
        if opened is not None and not mask[0]:
            yield max(0, opened-2), min(n, start+2)
            opened = None
        for left, right in runs(mask):
            a, b = start+left, start+right
            if opened is None:
                opened = a
            if right < len(mask):
                yield max(0, opened-2), min(n, b+2)
                opened = None
        if opened is not None and not mask[-1]:
            raise AssertionError("Unclosed hard region")
    if opened is not None:
        yield max(0, opened-2), n


def soft_regions(head, n, chunk=65536):
    opened, peak = None, 0.0
    for start in range(0, n, chunk):
        end = min(n, start+chunk)
        left, right = max(0, start-31), min(n, end+30)
        with np.errstate(over="ignore"):
            probability = 1/(1+np.exp(-head[left:right].astype(np.float64)))
        above = probability >= .5
        onsets = np.flatnonzero(above & ~np.pad(above[:-1], (1, 0))) + left
        exclusion = np.zeros(end-start, bool)
        for onset in onsets:
            if onset != 0:
                a, b = max(start, onset-30), min(end, onset+31)
                if a < b:
                    exclusion[a-start:b-start] = True
        rolling = np.convolve(probability, np.ones(31), "full")[15:15+len(probability)]
        offset = start-left
        values = probability[offset:offset+end-start]
        candidate = (rolling[offset:offset+end-start] > .03) & (values < .5) & ~exclusion
        if opened is not None and not candidate[0]:
            if peak >= .015:
                yield max(0, opened-64), min(n, start+64)
            opened, peak = None, 0.0
        for a, b in runs(candidate):
            if opened is None:
                opened, peak = start+a, 0.0
            peak = max(peak, float(np.max(values[a:b])))
            if b < len(candidate):
                if peak >= .015:
                    yield max(0, opened-64), min(n, start+b+64)
                opened, peak = None, 0.0
    if opened is not None and peak >= .015:
        yield max(0, opened-64), n


class GuardedVOD:
    def __init__(self, manifest, threads=2, progress=None):
        require_supported_platform()
        if threads < 1:
            raise ValueError("threads must be positive")
        import openvino as ov
        self.manifest = load_manifest(manifest)
        root = Path(manifest).resolve().parent
        core = ov.Core()
        config = dict(INFERENCE_PRECISION_HINT="f32", INFERENCE_NUM_THREADS=threads,
                      NUM_STREAMS=1, ENABLE_CPU_PINNING=False)
        self.fast = core.compile_model(str(root/self.manifest["fastModel"]), "CPU", config).create_infer_request()
        self.reference = core.compile_model(str(root/self.manifest["referenceModel"]), "CPU", config).create_infer_request()
        self.ov = ov
        self.progress = progress or (lambda event: None)
        self.lease = threading.Lock()

    def infer(self, request, frames):
        values = np.ascontiguousarray(frames, dtype=np.float32)
        request.set_input_tensor(self.ov.Tensor(values, shared_memory=True))
        request.infer()
        result = np.stack([request.get_tensor(name).data.copy().reshape(-1)
                           for name in ["output_1", "output_2"]])
        if result.shape != (2, frames.shape[1]) or not np.isfinite(result).all():
            raise ValueError("Invalid model outputs")
        return result

    def predict(self, spool, directory):
        if not self.lease.acquire(blocking=False):
            raise RuntimeError("Engine is already leased to a video")
        try:
            return self._predict(spool, Path(directory))
        finally:
            self.lease.release()

    def _predict(self, spool, root):
        root.mkdir(parents=True, exist_ok=False)
        spool.await_frames(975)
        pilot = self.infer(self.fast, spool.window(0))
        length = min(950, spool.frames)
        uncertain = np.any(np.abs(pilot[:, 25:25+length]) <= 1.5, axis=0)
        planned = merge((max(0, 25+a-100), min(1000, 25+b+100)) for a, b in runs(uncertain))
        full = sum(b-a for a, b in planned) >= 800
        count = 0
        with (root/"windows.f32").open("xb") as output:
            while True:
                spool.await_frames(count*950+975)
                if spool.eof and count*950 >= spool.frames:
                    break
                window = spool.window(count)
                values = self.infer(self.reference, window) if full else (pilot if count == 0 else self.infer(self.fast, window))
                output.write(values.astype("<f4").tobytes())
                count += 1
                self.progress(dict(type="progress", phase="initial", window=count, frames=spool.frames, emittedFrames=0))
        n = spool.frames
        if not spool.eof or count != (n+949)//950:
            raise ValueError("Incomplete input stream")
        values = np.memmap(root/"windows.f32", dtype="<f4", mode="r+", shape=(count, 2, 1000))
        scores = np.memmap(root/"logits.f32", dtype="<f4", mode="w+", shape=(2, n))
        known = np.memmap(root/"refined.u8", dtype=np.uint8, mode="w+", shape=(count, 1000))
        phases = []
        try:
            self._emit(values, scores, n)
            if full:
                phases.append(dict(phase="full-fp32", crops=count, executedFrames=1000*count,
                                   replacedFrames=1000*count, fullWindowFallbacks=count))
            else:
                phases.append(self._refine(spool, values, known, hard_regions(scores, n), n, "hard"))
                self._emit(values, scores, n)
                phases.append(self._refine(spool, values, known, merged_stream(soft_regions(scores[0], n)), n, "soft"))
                self._emit(values, scores, n)
            values.flush()
            scores.flush()
            result = dict(protocolVersion=1, policyId=POLICY["id"], frames=n, heads=2,
                          dtype="float32-le", layout="heads-frames", logitsFile="logits.f32",
                          logitsSha256=sha256(root/"logits.f32"), routing="pilot-full-fp32" if full else "guarded",
                          fastExecutedFrames=1000 if full else count*1000, phases=phases)
            self.progress(dict(type="progress", phase="complete", frames=n, emittedFrames=n))
            return result
        finally:
            values._mmap.close()
            scores._mmap.close()
            known._mmap.close()

    @staticmethod
    def _emit(windows, scores, n):
        for index in range(len(windows)):
            start = index*950
            length = min(950, n-start)
            scores[:, start:start+length] = windows[index, :, 25:25+length]

    def _refine(self, spool, logits, known, queries, n, phase):
        queries = iter(queries)
        pending = next(queries, None)
        stats = dict(phase=phase, crops=0, executedFrames=0, replacedFrames=0, fullWindowFallbacks=0)
        for index in range(len(logits)):
            start = index*950
            end = min(n, start+950)
            local = []
            while pending is not None and pending[0] < end:
                a, b = max(start, pending[0]), min(end, pending[1])
                if a < b:
                    local.append((25+a-start, 25+b-start))
                if pending[1] > end:
                    break
                pending = next(queries, None)
            missing = np.zeros(1000, bool)
            for a, b in merge(local):
                missing[a:b] = True
            targets = list(runs(missing & ~known[index].astype(bool)))
            if not targets:
                continue
            intervals = []
            for a, b in targets:
                left, right = max(0, a-98), min(1000, b+98)
                desired = min(1000, ((right-left+63)//64)*64)
                left = max(0, min(left-(desired-(right-left))//2, 1000-desired))
                intervals.append((left, left+desired))
            intervals = merge(intervals)
            while True:
                rounded = []
                for left, right in intervals:
                    desired = min(1000, ((right-left+63)//64)*64)
                    left = max(0, min(left-(desired-(right-left))//2, 1000-desired))
                    rounded.append((left, left+desired))
                rounded = merge(rounded)
                if rounded == intervals:
                    break
                intervals = rounded
            if sum(b-a for a, b in intervals) >= 800:
                intervals = [(0, 1000)]
                stats["fullWindowFallbacks"] += 1
            window = spool.window(index)
            for left, right in intervals:
                result = self.infer(self.reference, window[:, left:right])
                a, b = left+(98 if left > 0 else 0), right-(98 if right < 1000 else 0)
                logits[index, :, a:b] = result[:, a-left:b-left]
                known[index, a:b] = 1
                stats["crops"] += 1
                stats["executedFrames"] += right-left
                stats["replacedFrames"] += b-a
            self.progress(dict(type="progress", phase=phase, window=index+1, frames=n, emittedFrames=0))
        return stats
