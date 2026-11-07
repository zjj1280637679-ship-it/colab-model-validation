"""CPU-only helpers shared by generated notebooks. No network or GPU work."""
import hashlib
import importlib.metadata
import json
import pathlib
import threading
import time
import uuid
import zipfile

MODEL_PROFILES = {
    "qwen3_8": {"repo": "huihui-ai/Huihui-Qwen3.8-27B-abliterated", "revision": "739e3c5b89849f6c238ce1e5b70008612ae42cdd", "ram_gib": 64, "disk_gib": 80, "files": ("preflight.json", "qwen3_8_bf16.json", "qwen_checkpoint_audit.json", "runtime_versions.json")},
    "qwen_image_2_1": {"repo": "Qwen/Qwen-Image-2.1", "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3", "ram_gib": 48, "disk_gib": 55, "files": ("preflight.json", "qwen_image_2_1_bf16.json", "qwen_image_2_1_bf16_2048.png", "runtime_versions.json")},
}

def arm_shutdown(state, release, seconds=1500, timer_factory=threading.Timer, clock=time.monotonic):
    """One timer/deadline per kernel lifecycle; rerunning cannot extend it."""
    now = clock()
    deadline = state.setdefault("deadline", now + seconds)
    old = state.get("timer")
    if old is not None:
        old.cancel()
    timer = timer_factory(max(0, deadline - now), release)
    timer.daemon = True
    state["timer"] = timer
    timer.start()
    return deadline

def shutdown_now(state, release):
    """Keep fallback timer armed if immediate runtime release raises."""
    release()
    if state.get("timer") is not None:
        state["timer"].cancel()


def new_report_dir(base, profile):
    path = pathlib.Path(base) / (profile + "-" + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_resources(profile, available_ram_gib, disk_gib):
    config = MODEL_PROFILES[profile]
    if available_ram_gib < config["ram_gib"] or disk_gib < config["disk_gib"]:
        raise RuntimeError(f"STOP: {profile} conservatively requires available RAM >= {config['ram_gib']} GiB and free disk >= {config['disk_gib']} GiB. No model download started.")


def observed_distribution(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    source = None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "null")
        # Publish only an allowlisted source, never arbitrary URLs/usernames/tokens.
        if isinstance(direct, dict) and direct.get("url") == "https://github.com/huggingface/diffusers.git":
            commit = direct.get("vcs_info", {}).get("commit_id")
            if isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                source = {"url": direct["url"], "commit": commit}
    except (ValueError, TypeError):
        pass
    return {"version": dist.version, "source": source}


def export_allowlisted(report_dir, profile):
    """Archive a snapshot of approved files only; reject symlinks/missing files."""
    root = pathlib.Path(report_dir)
    payload = {}
    for name in MODEL_PROFILES[profile]["files"]:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe expected evidence file: {name}")
        payload[name] = path.read_bytes()
    # Manifest covers payload entries, excluding itself by definition.
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    archive = root.parent / (root.name + "_evidence.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in payload.items():
            zipped.writestr(name, data)
        zipped.writestr("sha256_manifest.json", manifest_bytes)
    (root / "sha256_manifest.json").write_bytes(manifest_bytes)
    return archive, manifest
