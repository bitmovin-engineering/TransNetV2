# Experimental offline CPU inference

This optional package implements guarded OpenVINO CPU inference. The original
TensorFlow, PyTorch and Live APIs remain independent. Supported runtime: Linux
x86_64, Python 3.12, NumPy 2.2.6 and OpenVINO 2025.3.0. TensorFlow and NNCF are
build-time dependencies and are not required by the worker.

The `aisa-vod-v1` profile evaluates the CNN with INT8, keeps auxiliary features and
heads in FP32, and recomputes sensitive regions with an FP32 reference. It is an
empirical policy: confidently wrong INT8 predictions can escape it. Qualify new
content against the original model and the actual consumer before adoption.
Raw INT8 alone is not a supported replacement for the original model.

## Installation and models

```sh
python -m pip install ./inference-cpu
```

`models/aisa-vod-v1` contains the evaluated XML/BIN pairs and their SHA256 manifest.
The BIN files use Git LFS. Obtain them from this fork at an immutable revision;
there is no runtime download. Model bytes and the exact policy are verified before
compiling either model. Incorrect, missing or incompatible models are fatal.

Build a fresh pair with explicit local inputs in a separate build environment:

```sh
python -m pip install './inference-cpu[build-models]'
python tools/build_cpu_models.py --weights /path/to/original-savedmodel \
  --calibration-rgb news.rgb drama.rgb animation.rgb --output /path/to/new-models
```

Each calibration file is raw RGB48x27 uint8. The recipe takes four evenly spaced
100-frame windows from each source, excludes operations outside the CNN from
quantization and audits placement. It records input hashes and offsets. Rebuilding
with different calibration bytes creates a new candidate requiring qualification;
the supplied model manifest identifies the evaluated model bytes. Original
SavedModel and variable-file hashes are checked before conversion.

## Worker contract

```sh
transnetv2-cpu --manifest /path/to/models/manifest.json \
  --work-dir /path/to/new-job --threads 2 < frames.rgb
```

The caller supplies **complete RGB48x27 uint8 frames in decoded order**. The worker
does not decode, resample, assign timestamps, or publish AISA shot lists. It starts
inference as input arrives and spools RGB to disk. Supply `--input-rgb frames.rgb`
to read an existing file instead of stdin. A directory must be new for every job.

Stdout contains bounded JSON lines: `ready`, `progress`, then one `result` or an
`error` followed by nonzero exit. Protocol version is 1; policy ID is
`aisa-vod-v1`. Progress reports real inference/refinement work; `emittedFrames`
remains zero until refinement is complete. Diagnostics use stderr.

Only `result.json`, atomically published after all inference, indicates a complete
result. `output/logits.f32` contains little-endian float32 `[2,frames]`, single
raw logits followed by many-hot raw logits. Metadata records frame count, layout,
dtype, checksum, routing and refinement execution counts. Apply sigmoid once.
Preserve the caller's original frame timestamps independently.

The caller owns process and directory lifetime. On cancellation: stop producing,
terminate the worker, observe its exit and all reader/writer completion, then
remove its directory. Partial files are never valid predictions. Empty/truncated
input, spool-limit exhaustion and model/inference errors fail the job. Disk-full
errors propagate. `--max-spool-bytes` bounds RGB storage; account separately for
small per-window logits and output curves (approximately 18 bytes per frame).

## Policy and resource ownership

The profile preserves 1000/950 windows, 25-frame edge padding, both-head hard
queries, global fade-mound queries, 98-frame crop context, crop rounding and the
pilot/full-window FP32 routes. Soft/hard decisions remain global across chunks.
AISA's fade consumer owns the downstream semantics this profile protects. Any
change to these rules requires a new policy ID and consumer-parity qualification;
ordinary runtime configuration cannot silently alter thresholds.

RGB reads and query scans are bounded; per-window logits, refinement masks and
emitted curves are disk-backed. Execution statistics are aggregate counters, not
an unbounded list of crops. Temporary storage grows with video length; inference
workspace memory is bounded by model state, a window and fixed query chunks.
The caller's own result/metadata retention must be budgeted separately.

One engine instance is exclusively leased to one video. Concurrent calls fail
instead of sharing mutable requests. The file-input API can reuse a `GuardedVOD`
instance sequentially; streamed input belongs to the process-owned worker.
The spool's input thread is daemonized so failure cannot strand process exit
waiting on an open producer. No decoder or network provisioning runs inside it.

Run the model-free contract suite with `pytest inference-cpu/tests`. Real-model
parity, long-media resource checks, actual-consumer quality and end-to-end timing
are separate qualification gates. Warm steady-state, cold start, model-only and
whole-application results must be reported separately. Unqualified platforms
should keep their existing TensorFlow path; an explicit unsupported CPU request
fails clearly. macOS has not been qualified.

## Container execution

Build the runtime from the repository root:

```sh
docker build -f inference-cpu/Dockerfile -t transnet-cpu .
```

Mount a verified model directory read-only and an empty caller-owned parent for
job files. Run under a UID that can write that parent. The worker needs no network
at runtime. Environments with an HTTPS inspection proxy can supply a combined CA
bundle as BuildKit secret `proxy_ca`; it is not copied into the image. Pin the
resulting image digest for deployment. The AISA integration uses an isolated
Python environment inside its own application image, rather than a second decoder.

The supplied model build provenance is in `models/aisa-vod-v1/BUILD.json`.
Calibration media is not redistributed; the model build recipe accepts explicit
licensed inputs and the provenance identifies the evaluated calibration bytes.
