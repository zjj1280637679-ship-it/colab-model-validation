# A100 BF16 preflight. No model weights, Drive mount, secrets or public tunnels.
import os, sys, json, time, pathlib, shutil, subprocess, platform, urllib.request
from datetime import datetime, timezone
import torch
import psutil
REPORT_DIR = pathlib.Path("/content/a100_validation")
REPORT_DIR.mkdir(exist_ok=True)
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0], "torch": torch.__version__, "gpu": [], "ram_gib": round(psutil.virtual_memory().total / 2**30, 2), "available_ram_gib": round(psutil.virtual_memory().available / 2**30,2), "disk_free_gib": round(shutil.disk_usage("/content").free / 2**30,2)}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        report["gpu"].append({"name":p.name,"vram_gib":round(p.total_memory/2**30,2),"capability":list(torch.cuda.get_device_capability(i)),"bf16_supported":torch.cuda.is_bf16_supported()})
report["models"] = {}
for repo in ["huihui-ai/Huihui-Qwen3.8-27B-abliterated", "Qwen/Qwen-Image-2.1"]:
    with urllib.request.urlopen("https://huggingface.co/api/models/"+repo, timeout=30) as r:
        data=json.load(r)
    report["models"][repo]={"revision":data.get("sha"),"gated":data.get("gated"),"license":data.get("cardData",{}).get("license"),"safetensors":data.get("safetensors")}
print(json.dumps(report,ensure_ascii=False,indent=2))
(REPORT_DIR/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
assert report["gpu"], "STOP: no GPU. No model download permitted."
assert all("A100" in g["name"] and g["bf16_supported"] for g in report["gpu"]), "STOP: A100 BF16 required."
print("PREFLIGHT_ONLY: no model has been loaded or tested.")


assert all(g["vram_gib"] > 70 for g in report["gpu"]), "STOP: A100 80GB required; 40GB is not validated."
