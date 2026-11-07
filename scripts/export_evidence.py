# Export only this test's public-safe evidence. No accounts, secrets or weights.
import importlib.metadata, hashlib, json, shutil, pathlib
from google.colab import files
versions={}
for package in ["torch","transformers","accelerate","huggingface_hub","safetensors","diffusers","numpy","Pillow"]:
    try: versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: pass
versions["diffusers_source_commit"]="84532a058702c74ae5d1dd6ceb05683fc5d5e768"
(REPORT_DIR/"runtime_versions.json").write_text(json.dumps(versions,indent=2))
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REPORT_DIR.iterdir() if p.is_file() and p.name != "sha256_manifest.json"}
(REPORT_DIR/"sha256_manifest.json").write_text(json.dumps(manifest,indent=2))
archive=shutil.make_archive("/content/a100_validation_evidence","zip",str(REPORT_DIR))
print("EVIDENCE_EXPORT_FILES",sorted(manifest))
print("EXPORT_SHA256",hashlib.sha256(pathlib.Path(archive).read_bytes()).hexdigest())
print("Next: disconnect and delete this paid runtime; then verify zero active sessions.")
files.download(archive)

