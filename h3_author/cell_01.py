PROGRAM_SOURCE_SHA256 = '9f6bb9f7189e124f3b908997faa7d9362686c75f2a3e9d117232490934cb7dc8'
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

try:
    # Colab cell: set up native MiniMax H3 support in ComfyUI.
    # Paste this file into one Colab cell and run once per session.
    
    USE_GOOGLE_DRIVE = False
    UPDATE_COMFYUI = False
    DOWNLOAD_FL2VA = True
    DOWNLOAD_REF2VA = False
    DOWNLOAD_REF2V_TURBO_LORA = False
    
    COMFYUI_REVISION = "e01fb4c56b7a88149d469b99cbbfe3223d715054"
    MODEL_REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"
    
    import hashlib
    import os
    import subprocess
    import sys
    
    
    def run(*args, check=True):
        stage = str(args[0]) + (' ' + str(args[1]) if len(args) > 1 else '')
        print('BEGIN', stage, flush=True)
        proc = subprocess.run(list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = safe_text(proc.stdout)
        print(output, end='', flush=True)
        with (RUN_ROOT / 'setup.log').open('a') as f: f.write('BEGIN ' + stage + '\n' + output + '\nEXIT ' + str(proc.returncode) + '\n')
        print('END', stage, 'exit', proc.returncode, flush=True)
        if check and proc.returncode: raise RuntimeError('Command failed: ' + stage + ' exit=' + str(proc.returncode))
        return proc
    
    
    def sha256_file(file_path):
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    
    
    def ensure_download(url, destination, expected_size, expected_sha256):
        asset_name = os.path.basename(destination)
        stage = {'asset': asset_name, 'url': url, 'expected_bytes': expected_size,
                 'expected_sha256': expected_sha256, 'download_seconds': 0.0, 'hash_seconds': 0.0}
        RESULT.setdefault('asset_stages', []).append(stage)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.isfile(destination):
            current_size = os.path.getsize(destination)
            if current_size == expected_size:
                print('HASH_START', asset_name, 'existing_file', flush=True)
                hash_started = time.monotonic()
                actual = sha256_file(destination)
                stage['hash_seconds'] += round(time.monotonic() - hash_started, 3)
                print('HASH_END', asset_name, 'seconds', stage['hash_seconds'], 'sha256', actual, flush=True)
                if actual == expected_sha256:
                    stage['status'] = 'EXISTING_FILE_VERIFIED'
                    print(f"Using verified file: {destination}")
                    return
                print(f"Removing checksum-mismatched file: {destination}")
                os.remove(destination)
            elif current_size > expected_size:
                print(f"Removing oversized file: {destination}")
                os.remove(destination)
            else:
                print(f"Resuming partial download: {destination}")
    
        print('DOWNLOAD_START', asset_name, 'expected_bytes', expected_size, 'url', url, flush=True)
        download_started = time.monotonic()
        try:
            run("wget", "-nv", "-c", "-O", destination, url)
        finally:
            stage['download_seconds'] = round(time.monotonic() - download_started, 3)
            print('DOWNLOAD_END', asset_name, 'seconds', stage['download_seconds'], flush=True)
        actual_size = os.path.getsize(destination)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {destination}: expected {expected_size}, got {actual_size}"
            )
        print('HASH_START', asset_name, 'bytes', actual_size, flush=True)
        hash_started = time.monotonic()
        actual_sha256 = sha256_file(destination)
        stage['hash_seconds'] += round(time.monotonic() - hash_started, 3)
        print('HASH_END', asset_name, 'seconds', stage['hash_seconds'], 'sha256', actual_sha256, flush=True)
        stage['actual_bytes'] = actual_size
        stage['actual_sha256'] = actual_sha256
        stage['status'] = 'VERIFIED' if actual_sha256 == expected_sha256 else 'SHA256_MISMATCH' 
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {destination}: expected {expected_sha256}, got {actual_sha256}"
            )
    
    
    WORKSPACE = str(RUN_ROOT / "ComfyUI")
    
    if not os.path.isdir(WORKSPACE):
        run("git", "clone", "--filter=blob:none", "https://github.com/Comfy-Org/ComfyUI.git", WORKSPACE)
    
    os.chdir(WORKSPACE)
    if UPDATE_COMFYUI:
        run("git", "fetch", "--depth", "1", "origin", "master")
        run("git", "checkout", "--detach", "FETCH_HEAD")
    else:
        run("git", "fetch", "--depth", "1", "origin", COMFYUI_REVISION)
        run("git", "checkout", "--detach", COMFYUI_REVISION)
    
    # Preserve the Colab torch/CUDA stack and already-loaded numpy/Pillow; never hot-reload.
    protected = ['torch', 'torchvision', 'torchaudio', 'numpy', 'Pillow']
    constraints = RUN_ROOT / 'colab_constraints.txt'
    constraint_rows = []
    for package in protected:
        try: constraint_rows.append(package + '==' + importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError: pass
    constraints.write_text('\n'.join(constraint_rows) + '\n')
    run(sys.executable, '-m', 'pip', 'check', check=False)  # baseline, not an exemption
    run(sys.executable, '-m', 'pip', 'install', '-c', str(constraints), '-r', f'{WORKSPACE}/requirements.txt', 'jedi==0.19.2')
    run(sys.executable, '-m', 'pip', 'check')
    RESULT['installed_freeze'] = safe_text(run(sys.executable, '-m', 'pip', 'freeze').stdout).splitlines()
    RESULT['comfyui_revision'] = run('git', 'rev-parse', 'HEAD').stdout.strip()
    if RESULT['comfyui_revision'] != COMFYUI_REVISION: raise RuntimeError('ComfyUI commit mismatch')
    
    if not DOWNLOAD_FL2VA and not DOWNLOAD_REF2VA:
        raise ValueError("Enable DOWNLOAD_FL2VA, DOWNLOAD_REF2VA, or both")
    
    MODEL_FILES = [
        (
            "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
            51506295256,
            "600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d",
        ),
        (
            "vae/minimax_h3_video_vae_fp16.safetensors",
            5207808496,
            "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        ),
        (
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            605254808,
            "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        ),
    ]
    
    if DOWNLOAD_REF2VA:
        MODEL_FILES.insert(
            0,
            (
                "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                20970379616,
                "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
            ),
        )
    
    if DOWNLOAD_FL2VA:
        MODEL_FILES.insert(
            0,
            (
                "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
                66280487368,
                "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",
            ),
        )
    
    if DOWNLOAD_REF2V_TURBO_LORA:
        MODEL_FILES.append(
            (
                "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                1956193000,
                "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
            )
        )
    
    for relative_path, expected_size, expected_sha256 in (MODEL_FILES if MODE == "GPU_FULL_MODEL" else []):
        ensure_download(
            f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{MODEL_REVISION}/{relative_path}",
            f"{WORKSPACE}/models/{relative_path}",
            expected_size,
            expected_sha256,
        )
    
    print(f"Setup complete. WORKSPACE = {WORKSPACE}")
    print(f"ComfyUI revision = {COMFYUI_REVISION if not UPDATE_COMFYUI else 'latest master'}")
    print(f"MiniMax H3 model revision = {MODEL_REVISION}")
    print(f"DOWNLOAD_FL2VA = {DOWNLOAD_FL2VA}")
    print(f"DOWNLOAD_REF2VA = {DOWNLOAD_REF2VA}")
    print(f"DOWNLOAD_REF2V_TURBO_LORA = {DOWNLOAD_REF2V_TURBO_LORA}")
    download_size_gb = sum(item[1] for item in MODEL_FILES) / 1_000_000_000
    print(f"Model download size = {download_size_gb:.2f} GB (decimal)")
    
    
    RESULT['planned_model_assets'] = [{'path': p, 'bytes': size, 'sha256': digest} for p,size,digest in MODEL_FILES]
    RESULT['model_revision'] = MODEL_REVISION
    RESULT['model_downloads'] = 'VERIFIED_SIZE_AND_SHA256' if MODE == 'GPU_FULL_MODEL' else 'SKIPPED_NO_WEIGHTS'
    
    hook_source = "# Read-only summaries after native ComfyUI loaders. No tensor movement or dtype change.\ndef _h3_precision(component, module, options):\n    import os, json, collections\n    counts, devices, parameter_classes = collections.Counter(), collections.Counter(), collections.Counter()\n    quantized = []\n    for name, parameter in module.named_parameters():\n        counts[str(parameter.dtype)] += parameter.numel()\n        devices[str(parameter.device)] += parameter.numel()\n        parameter_classes[type(parameter).__module__ + '.' + type(parameter).__name__] += 1\n        if getattr(parameter, 'is_quantized', False) or any(x in type(parameter).__name__.lower() for x in ('quantized', 'int8', 'float8', 'fp8', 'nvfp')):\n            quantized.append(name)\n    quant_modules = [type(m).__name__ for m in module.modules() if any(x in type(m).__name__.lower() for x in ('quantized', 'int8', 'float8', 'fp8', 'nvfp'))]\n    row = {'component': component, 'parameter_numel_by_dtype': dict(counts),\n           'parameter_numel_by_device': dict(devices), 'parameter_classes': dict(parameter_classes),\n           'quantized_parameter_count': len(quantized), 'quantized_module_classes': sorted(set(quant_modules)),\n           'loader_options': {str(k): str(v) for k,v in options.items()},\n           'scope': 'parameter storage at native-loader return, not autocast/compute dtype or a complete checkpoint audit'}\n    print('H3_NATIVE_PRECISION ' + json.dumps(row, sort_keys=True), flush=True)\n    with open(os.environ['H3_PRECISION_LOG'], 'a') as f: f.write(json.dumps(row) + '\\n')\n    allowed = {'torch.float16', 'torch.float32'} if component == 'video_vae' else ({'torch.float32'} if component == 'audio_vae' else {'torch.bfloat16', 'torch.float32'})\n    if not counts or set(counts) - allowed or quantized or quant_modules:\n        raise RuntimeError('Unexpected parameter precision or quantization: ' + component)\n    if component in ('transformer','text_encoder') and not counts.get('torch.bfloat16'):\n        raise RuntimeError('No BF16 parameters in ' + component)\n"
    node_path = pathlib.Path(WORKSPACE) / 'nodes.py'
    node_source = node_path.read_text()
    patches = {
        '        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)':
        '        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)\n        _h3_precision("transformer", model.model, model_options)',
        '        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)':
        '        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)\n        _h3_precision("text_encoder", clip.cond_stage_model, model_options)',
        '        vae = comfy.sd.VAE(sd=sd, metadata=metadata)':
        '        vae = comfy.sd.VAE(sd=sd, metadata=metadata)\n        _h3_precision("audio_vae" if "audio" in vae_name else "video_vae", vae.first_stage_model, {"vae_dtype": str(vae.vae_dtype)})',
    }
    for old,new in patches.items():
        if node_source.count(old) != 1: raise RuntimeError('Pinned native loader hook target changed')
        node_source = node_source.replace(old,new)
    node_source += '\n' + hook_source
    compile(node_source, 'nodes.py', 'exec')
    node_path.write_text(node_source)
    os.environ['H3_PRECISION_LOG'] = str(RUN_ROOT / 'precision.jsonl')
    RESULT['nodes_py_sha256'] = hashlib.sha256(node_source.encode()).hexdigest()
    RESULT['status'] = 'SETUP_READY'
    save_result()
except Exception as error:
    RESULT.update(status='SETUP_FAILED', error_type=type(error).__name__, error=safe_text(error))
    save_result()
    release_runtime('setup failure')
    raise
