# Requires runtime_helpers and MODEL_PROFILE from the notebook bootstrap.
import os, sys, json, time, pathlib, shutil, subprocess, platform, urllib.request
from datetime import datetime, timezone
import torch
import psutil
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
PROFILE = MODEL_PROFILES[MODEL_PROFILE]
REPORT_DIR = new_report_dir("/content/a100_validation", MODEL_PROFILE)
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0], "torch": torch.__version__, "gpu": [], "ram_gib": round(psutil.virtual_memory().total / 2**30, 2), "available_ram_gib": psutil.virtual_memory().available / 2**30, "disk_free_gib": shutil.disk_usage("/content").free / 2**30, "profile": MODEL_PROFILE, "resource_thresholds": {"available_ram_gib": PROFILE["ram_gib"], "free_disk_gib": PROFILE["disk_gib"]}}
assert torch.cuda.is_available(), "STOP: no GPU. No model download permitted."
p = torch.cuda.get_device_properties(0)
report["gpu"].append({"name": p.name, "vram_gib": p.total_memory / 2**30, "capability": list(torch.cuda.get_device_capability(0)), "bf16_supported": torch.cuda.is_bf16_supported()})
assert "A100" in p.name and p.total_memory / 2**30 > 70 and torch.cuda.is_bf16_supported(), "STOP: A100 80GB BF16 required; 40GB is not validated."
require_resources(MODEL_PROFILE, report["available_ram_gib"], report["disk_free_gib"])
url = "https://huggingface.co/api/models/" + PROFILE["repo"] + "/revision/" + PROFILE["revision"]
with urllib.request.urlopen(url, timeout=30) as response:
    data = json.load(response)
assert data.get("sha") == PROFILE["revision"], "STOP: resolved model revision mismatch."
report["models"] = {PROFILE["repo"]: {"revision": data["sha"], "gated": data.get("gated"), "license": data.get("cardData", {}).get("license"), "safetensors": data.get("safetensors")}}
print(json.dumps(report, ensure_ascii=False, indent=2))
(REPORT_DIR / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("PREFLIGHT_ONLY: no model has been loaded or tested. Evidence directory:", REPORT_DIR)
