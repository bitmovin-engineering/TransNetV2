"""A disk spool allows one shared decoder to feed inference without retaining RGB in RAM."""
import os
import hashlib
from pathlib import Path
import threading
import numpy as np

FRAME_BYTES = 27 * 48 * 3

class FrameSpool:
    def __init__(self, path, stream=None, max_bytes=8 * 1024**3):
        self.path = Path(path)
        self.condition = threading.Condition()
        self.frames = 0
        self.eof = stream is None
        self.error = None
        self.thread = None
        self.max_bytes = max_bytes
        self.digest = hashlib.sha256()
        if max_bytes < FRAME_BYTES:
            raise ValueError("Spool limit is smaller than one frame")
        if stream is None:
            size = self.path.stat().st_size
            if not size or size % FRAME_BYTES or size > max_bytes:
                raise ValueError("Empty, truncated or oversized RGB input")
            self.frames = size // FRAME_BYTES
            with self.path.open("rb") as source:
                self.digest = hashlib.file_digest(source, "sha256")
        else:
            self.thread = threading.Thread(target=self._receive, args=(stream,), name="transnet-rgb-spool", daemon=True)
            self.thread.start()
        self.fd = None

    def _receive(self, stream):
        try:
            with self.path.open("xb", buffering=0) as out:
                pending = b""
                while True:
                    block = stream.read(64 * FRAME_BYTES)
                    if not block:
                        break
                    pending += block
                    count = len(pending) // FRAME_BYTES
                    complete = count * FRAME_BYTES
                    if complete:
                        if (self.frames + count) * FRAME_BYTES > self.max_bytes:
                            raise ValueError("RGB spool byte limit exceeded")
                        remaining = memoryview(pending)[:complete]
                        while remaining:
                            written = out.write(remaining)
                            if not written:
                                raise OSError("RGB spool write made no progress")
                            remaining = remaining[written:]
                        self.digest.update(pending[:complete])
                        pending = pending[complete:]
                        with self.condition:
                            self.frames += count
                            self.condition.notify_all()
                if pending or not self.frames:
                    raise ValueError("Empty or truncated RGB stream")
        except BaseException as error:
            self.error = error
        finally:
            with self.condition:
                self.eof = True
                self.condition.notify_all()

    def await_frames(self, count):
        with self.condition:
            self.condition.wait_for(lambda: self.frames >= count or self.eof)
            if self.error:
                raise self.error
            return self.frames >= count

    def window(self, index):
        start = index * 950 - 25
        end = start + 1000
        self.await_frames(end)
        with self.condition:
            count = self.frames
        left, right = max(0, start), min(count, end)
        if left >= right:
            raise IndexError(index)
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        raw = os.pread(self.fd, (right-left)*FRAME_BYTES, left*FRAME_BYTES)
        if len(raw) != (right-left)*FRAME_BYTES:
            raise ValueError("RGB spool changed or was truncated")
        values = np.frombuffer(raw, np.uint8).reshape(-1, 27, 48, 3)
        if left != start or right != end:
            values = np.pad(values, ((left-start, end-right), (0,0), (0,0), (0,0)), mode="edge")
        return values[None]

    def close(self):
        if self.thread:
            self.thread.join()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
