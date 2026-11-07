# Thin safety/logging wrapper around shinshin86/comfy-agent's two Colab cells.
import os, sys, json, time, threading, subprocess, pathlib, uuid, re, importlib.metadata


def release_runtime(reason):
    print("RELEASE_RUNTIME", reason, flush=True)
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as error:
        print("Release unavailable:", type(error).__name__, "Use Runtime > Disconnect and delete runtime.", flush=True)

if MODE not in ("CPU_NO_WEIGHTS", "GPU_FULL_MODEL") or not ALLOW_THIS_RUN:
    release_runtime("default gate closed")
    raise RuntimeError("Explicit single-run authorization is required before setup")
if MODE == "GPU_FULL_MODEL" and not (
    I_READ_H3_LICENSE and KNOWN_APPLICABLE_BLOCK_STATUS == "none_identified"
    and HOST_REGION_STATUS in ("unknown", "reviewed")
    and I_UNDERSTAND_UNRESOLVED_LICENSE_RISK
):
    release_runtime("license review gate closed")
    raise RuntimeError("Read the license and review applicable blocks before GPU/model download")
if type(GPU_BUDGET_MINUTES) is not int or not 1 <= GPU_BUDGET_MINUTES <= 50:
    release_runtime("invalid total budget")
    raise ValueError("GPU_BUDGET_MINUTES must be an integer from 1 through 50")
if globals().get("_H3_AUTHOR_STARTED"):
    raise RuntimeError("This single-run cell already started. Do not reset or extend its deadline.")
_H3_AUTHOR_STARTED = True
RUN_ROOT = pathlib.Path("/content") / ("h3_author_" + uuid.uuid4().hex[:12])
RUN_ROOT.mkdir()
TOTAL_BUDGET_MINUTES = 12 if MODE == "CPU_NO_WEIGHTS" else GPU_BUDGET_MINUTES
RUN_STARTED_MONOTONIC = time.monotonic()
DEADLINE = RUN_STARTED_MONOTONIC + TOTAL_BUDGET_MINUTES * 60
_RELEASE_TIMER = threading.Timer(max(0, DEADLINE - time.monotonic()), release_runtime, args=("single lifecycle deadline",))
_RELEASE_TIMER.daemon = True
_RELEASE_TIMER.start()
RESULT = {"mode": MODE, "program_source_sha256": PROGRAM_SOURCE_SHA256,
          "status": "SETUP_STARTED", "model_loaded": False, "inference_executed": False,
          "total_budget_minutes": TOTAL_BUDGET_MINUTES,
          "gpu_peak_memory": "NOT_MEASURED", "compute_autocast_dtype": "NOT_MEASURED",
          "source": "shinshin86/comfy-agent@a386b5cfaca565cc24cc592a0b480cb6c06971c9",
          "known_applicable_block_status": KNOWN_APPLICABLE_BLOCK_STATUS if MODE == "GPU_FULL_MODEL" else "NOT_APPLICABLE_NO_WEIGHTS",
          "host_region_status": HOST_REGION_STATUS if MODE == "GPU_FULL_MODEL" else "NOT_RECORDED"}

def safe_text(value):
    value = str(value)
    value = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)(token|authorization|api[_-]?key|password)([=:\s]+)[^&\s\"']+", r"\1\2[REDACTED]", value)
    value = re.sub(r"\bhf_[A-Za-z0-9]+\b", "[REDACTED]", value)
    return value

def save_result():
    RESULT["elapsed_total_seconds"] = round(time.monotonic() - RUN_STARTED_MONOTONIC, 3)
    RESULT["remaining_budget_seconds"] = round(max(0, DEADLINE - time.monotonic()), 3)
    RESULT["runtime_versions"] = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "Pillow", "transformers", "safetensors", "av", "comfy-kitchen", "comfy-aimdo", "comfyui-frontend-package", "jedi"):
        try: RESULT["runtime_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: RESULT["runtime_versions"][name] = None
    payload = json.dumps(RESULT, ensure_ascii=False, indent=2)
    (RUN_ROOT / "result.json").write_text(payload)
    print("H3_AUTHOR_RESULT_JSON\n" + payload, flush=True)

# Only Colab provider hardware is queried; no location or credentials are read.
probe = subprocess.run([sys.executable, "-c", "import torch,json; print(json.dumps({'cuda':torch.cuda.is_available(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'vram':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0}))"], capture_output=True, text=True)
try:
    if probe.returncode: raise RuntimeError("Hardware probe failed: " + safe_text(probe.stderr))
    RESULT["hardware"] = json.loads(probe.stdout.strip().splitlines()[-1])
    if MODE == "CPU_NO_WEIGHTS" and RESULT["hardware"]["cuda"]:
        raise RuntimeError("CPU_NO_WEIGHTS requires a CPU runtime, not a billed GPU")
    if MODE == "GPU_FULL_MODEL":
        h = RESULT["hardware"]
        if not h["cuda"] or "A100" not in h["name"] or h["vram"] < 75 * 2**30:
            raise RuntimeError("This BF16 trial requires A100 80GB; no quantization fallback")
        import shutil
        available = int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        RESULT["available_ram_bytes"] = available
        RESULT["free_disk_bytes"] = shutil.disk_usage(RUN_ROOT).free
        if available < 140 * 2**30 or RESULT["free_disk_bytes"] < 123599845928 + 20 * 2**30:
            raise RuntimeError("Conservative RAM/disk budget not met; these gates do not guarantee no OOM")
except Exception as error:
    RESULT.update(status="FAILED_HARDWARE", error_type=type(error).__name__, error=safe_text(error))
    save_result(); release_runtime("preflight failure"); raise
print("HARDWARE", RESULT["hardware"], "PROGRAM", PROGRAM_SOURCE_SHA256, flush=True)
