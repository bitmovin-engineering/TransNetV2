"""Content-addressed model and consumer-policy contract; no network provisioning."""
import hashlib
import json
import platform
from pathlib import Path

POLICY_ID = "aisa-vod-v1"
POLICY = dict(id=POLICY_ID, window=1000, stride=950, edgePadding=25,
              hardMargin=1.5, hardExpansion=2, context=98, cropRounding=64,
              fullWindowFraction=0.8, softPeakMin=0.015, softExpansion=64,
              fadeRollingFrames=31, fadeRollingSum=0.03, hardExclusion=30)
FP32_POLICY_ID = "aisa-vod-fp32-v1"
FP32_POLICY = dict(id=FP32_POLICY_ID, window=1000, stride=950, edgePadding=25,
                   precision="fp32", passes=1)
SOURCE = {"saved_model.pb": "8ac2a52c5719690d512805b6eaf5ce12097c1d8860b3d9de245dcbbc3100f554",
          "variables/variables.data-00000-of-00001": "b8c9dc3eb807583e6215cabee9ca61737b3eb1bceff68418b43bf71459669367",
          "variables/variables.index": "8b99e28b4ad11372d9a1ad9703298c2e370df14859da4245fdbe818e92dd403f"}

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write_manifest(directory, fast="folded8", reference="first-composed-f32"):
    root = Path(directory)
    files = {name: sha256(root / name) for stem in [fast, reference]
             for name in [stem + ".xml", stem + ".bin"]}
    manifest = dict(schemaVersion=1, policy=POLICY, sourceWeights=SOURCE,
                    fastModel=fast + ".xml", referenceModel=reference + ".xml", files=files)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def write_fp32_manifest(directory, model="first-composed-f32"):
    root = Path(directory)
    files = {model+ext: sha256(root/(model+ext)) for ext in [".xml", ".bin"]}
    manifest = dict(schemaVersion=1, policy=FP32_POLICY, sourceWeights=SOURCE,
                    model=model+".xml", inputs={"rgb": "input_1"}, files=files)
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest

def load_manifest(path, policy=POLICY):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if policy not in [POLICY, FP32_POLICY] or manifest.get("schemaVersion") != 1 or manifest.get("policy") != policy:
        raise ValueError("Unsupported model manifest or precision policy")
    if manifest.get("sourceWeights") != SOURCE:
        raise ValueError("Unexpected source weights")
    expected = set()
    if policy == FP32_POLICY and manifest.get("inputs") != {"rgb": "input_1"}:
        raise ValueError("Unsupported FP32 model input contract")
    for key in (["model"] if policy == FP32_POLICY else ["fastModel", "referenceModel"]):
        model = Path(manifest[key])
        if model.suffix != ".xml":
            raise ValueError("Model must be an XML/BIN pair")
        expected.update([str(model), str(model.with_suffix(".bin"))])
    if set(manifest["files"]) != expected:
        raise ValueError("Incomplete model file manifest")
    for name, digest in manifest["files"].items():
        source = (path.parent / name).resolve()
        if not source.is_relative_to(path.parent) or sha256(source) != digest:
            raise ValueError("Model checksum/path mismatch: " + name)
    return manifest

def require_supported_platform():
    if platform.system() != "Linux" or platform.machine().lower() not in ["x86_64", "amd64"]:
        raise RuntimeError("This experimental backend supports Linux x86_64 only")
