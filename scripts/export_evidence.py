# Export only the selected run's explicit file allowlist.
from google.colab import files
versions = {package: observed_distribution(package) for package in ["torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "diffusers", "numpy", "Pillow"]}
(REPORT_DIR / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
archive, manifest = export_allowlisted(REPORT_DIR, MODEL_PROFILE)
print("EVIDENCE_EXPORT_FILES", sorted(manifest))
print("EXPORT_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
print("ZIP contains exactly manifest-listed payload files plus sha256_manifest.json.")
print("Next: disconnect and delete this paid runtime; verify zero active sessions.")
files.download(str(archive))
