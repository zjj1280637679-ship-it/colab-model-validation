import pathlib,json,hashlib,difflib
r=pathlib.Path(__file__).parent
orig=(r/'upstream/01_setup.py').read_text()
s=orig.replace('CLOUDFLARED_VERSION = "2026.7.2"\n','').replace('CLOUDFLARED_SHA256 = "88195157a136199a86977c122a22084dae6907480bbe3640222b7b55834afc3a"\n','')
s=s.replace('    return subprocess.run(list(args), check=check)', '''    stage = str(args[0]) + (' ' + str(args[1]) if len(args) > 1 else '')
    print('BEGIN', stage, flush=True)
    proc = subprocess.run(list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = safe_text(proc.stdout)
    print(output, end='', flush=True)
    with (RUN_ROOT / 'setup.log').open('a') as f: f.write('BEGIN ' + stage + '\\n' + output + '\\nEXIT ' + str(proc.returncode) + '\\n')
    print('END', stage, 'exit', proc.returncode, flush=True)
    if check and proc.returncode: raise RuntimeError('Command failed: ' + stage + ' exit=' + str(proc.returncode))
    return proc''')
a=s.index('if USE_GOOGLE_DRIVE:');b=s.index('if not os.path.isdir(WORKSPACE):',a)
s=s[:a]+'WORKSPACE = str(RUN_ROOT / "ComfyUI")\n\n'+s[b:]
s=s.replace('run(sys.executable, "-m", "pip", "install", "-q", "-r", f"{WORKSPACE}/requirements.txt")', '''# Preserve the Colab torch/CUDA stack and already-loaded numpy/Pillow; never hot-reload.
protected = ['torch', 'torchvision', 'torchaudio', 'numpy', 'Pillow']
constraints = RUN_ROOT / 'colab_constraints.txt'
constraint_rows = []
for package in protected:
    try: constraint_rows.append(package + '==' + importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError: pass
constraints.write_text('\\n'.join(constraint_rows) + '\\n')
run(sys.executable, '-m', 'pip', 'check', check=False)  # baseline, not an exemption
run(sys.executable, '-m', 'pip', 'install', '-c', str(constraints), '-r', f'{WORKSPACE}/requirements.txt', 'jedi==0.19.2')
run(sys.executable, '-m', 'pip', 'check')
RESULT['installed_freeze'] = safe_text(run(sys.executable, '-m', 'pip', 'freeze').stdout).splitlines()
RESULT['comfyui_revision'] = run('git', 'rev-parse', 'HEAD').stdout.strip()
if RESULT['comfyui_revision'] != COMFYUI_REVISION: raise RuntimeError('ComfyUI commit mismatch')''')
s=s.replace('text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors').replace('15687142551','51506295256').replace('35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6','600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d')
s=s.replace('diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors','diffusion_models/minimax_h3_fl2va_bf16.safetensors').replace('''            20970379616,
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",''','''            66280487368,
            "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",''')
s=s.replace('    os.makedirs(os.path.dirname(destination), exist_ok=True)', "    asset_name = os.path.basename(destination)\n    stage = {'asset': asset_name, 'url': url, 'expected_bytes': expected_size,\n             'expected_sha256': expected_sha256, 'download_seconds': 0.0, 'hash_seconds': 0.0}\n    RESULT.setdefault('asset_stages', []).append(stage)\n    os.makedirs(os.path.dirname(destination), exist_ok=True)")
s=s.replace('            actual = sha256_file(destination)', "            print('HASH_START', asset_name, 'existing_file', flush=True)\n            hash_started = time.monotonic()\n            actual = sha256_file(destination)\n            stage['hash_seconds'] += round(time.monotonic() - hash_started, 3)\n            print('HASH_END', asset_name, 'seconds', stage['hash_seconds'], 'sha256', actual, flush=True)")
s=s.replace('                print(f"Using verified file: {destination}")', '                stage[\'status\'] = \'EXISTING_FILE_VERIFIED\'\n                print(f"Using verified file: {destination}")')
s=s.replace('    run("wget", "-c", "-O", destination, url)', '    print(\'DOWNLOAD_START\', asset_name, \'expected_bytes\', expected_size, \'url\', url, flush=True)\n    download_started = time.monotonic()\n    try:\n        run("wget", "-nv", "-c", "-O", destination, url)\n    finally:\n        stage[\'download_seconds\'] = round(time.monotonic() - download_started, 3)\n        print(\'DOWNLOAD_END\', asset_name, \'seconds\', stage[\'download_seconds\'], flush=True)')
s=s.replace('    actual_sha256 = sha256_file(destination)', "    print('HASH_START', asset_name, 'bytes', actual_size, flush=True)\n    hash_started = time.monotonic()\n    actual_sha256 = sha256_file(destination)\n    stage['hash_seconds'] += round(time.monotonic() - hash_started, 3)\n    print('HASH_END', asset_name, 'seconds', stage['hash_seconds'], 'sha256', actual_sha256, flush=True)\n    stage['actual_bytes'] = actual_size\n    stage['actual_sha256'] = actual_sha256\n    stage['status'] = 'VERIFIED' if actual_sha256 == expected_sha256 else 'SHA256_MISMATCH' ")
s=s.replace('for relative_path, expected_size, expected_sha256 in MODEL_FILES:', 'for relative_path, expected_size, expected_sha256 in (MODEL_FILES if MODE == "GPU_FULL_MODEL" else []):')
a=s.index('cloudflared_deb = ');b=s.index('print(f"Setup complete.',a);s=s[:a]+s[b:]
s=s.replace('print(f"cloudflared version = {CLOUDFLARED_VERSION}")','')
s += '''\nRESULT['planned_model_assets'] = [{'path': p, 'bytes': size, 'sha256': digest} for p,size,digest in MODEL_FILES]
RESULT['model_revision'] = MODEL_REVISION
RESULT['model_downloads'] = 'VERIFIED_SIZE_AND_SHA256' if MODE == 'GPU_FULL_MODEL' else 'SKIPPED_NO_WEIGHTS'
'''
# Tiny read-only hooks inserted at the original loader's return points. Fail on source drift.
hook=(r/'precision_hook.py').read_text()
s += '\n'+f"hook_source = {hook!r}\n"
s += '''node_path = pathlib.Path(WORKSPACE) / 'nodes.py'
node_source = node_path.read_text()
patches = {
    '        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)':
    '        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)\\n        _h3_precision("transformer", model.model, model_options)',
    '        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)':
    '        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)\\n        _h3_precision("text_encoder", clip.cond_stage_model, model_options)',
    '        vae = comfy.sd.VAE(sd=sd, metadata=metadata)':
    '        vae = comfy.sd.VAE(sd=sd, metadata=metadata)\\n        _h3_precision("audio_vae" if "audio" in vae_name else "video_vae", vae.first_stage_model, {"vae_dtype": str(vae.vae_dtype)})',
}
for old,new in patches.items():
    if node_source.count(old) != 1: raise RuntimeError('Pinned native loader hook target changed')
    node_source = node_source.replace(old,new)
node_source += '\\n' + hook_source
compile(node_source, 'nodes.py', 'exec')
node_path.write_text(node_source)
os.environ['H3_PRECISION_LOG'] = str(RUN_ROOT / 'precision.jsonl')
RESULT['nodes_py_sha256'] = hashlib.sha256(node_source.encode()).hexdigest()
RESULT['status'] = 'SETUP_READY'
save_result()
'''
(r/'01_setup.py').write_text(s)
(r/'setup.diff').write_text(''.join(difflib.unified_diff(orig.splitlines(True),s.splitlines(True),fromfile='author/01_setup.py',tofile='minimal/01_setup.py')))
flow=json.loads((r/'upstream/minimax_h3_t2v.json').read_text())
flow['6']['inputs']['unet_name']='minimax_h3_fl2va_bf16.safetensors'
flow['13']['inputs']['clip_name']='qwen3vl_32b_minimax_h3_bf16.safetensors'
(r/'minimax_h3_t2v.json').write_text(json.dumps(flow,indent=2)+'\n')
# All nodes, 20 steps, prompt, seed, 864x480, 124 frames and links unchanged.
start=(r/'02_start_comfyui.py').read_text()
preamble=(r/'preamble.py').read_text()
# Source hash covers exactly the two executable cell bodies + literal author graph,
# excludes settings and the hash declaration itself. CPU and GPU use the same body.
flow_literal='AUTHOR_WORKFLOW = '+repr(flow)+'\n'
setup_body=preamble+'\ntry:\n'+''.join('    '+line+'\n' for line in s.splitlines())+'''except Exception as error:
    RESULT.update(status='SETUP_FAILED', error_type=type(error).__name__, error=safe_text(error))
    save_result()
    release_runtime('setup failure')
    raise
'''
source_hash=hashlib.sha256((setup_body+'\n'+flow_literal+start).encode()).hexdigest()
setup_cell='PROGRAM_SOURCE_SHA256 = '+repr(source_hash)+'\n'+setup_body
start_cell=flow_literal+start
(r/'cell_01.py').write_text(setup_cell)
(r/'cell_02.py').write_text(start_cell)
def cell(kind,text):return {'cell_type':kind,'metadata':{},'source':text.splitlines(True),**({'execution_count':None,'outputs':[]} if kind=='code' else {})}
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
 cell('markdown', '# Mature author H3: validated native inference baseline\nBased on shinshin86/comfy-agent at a386b5cfaca565cc24cc592a0b480cb6c06971c9. MIT ©2026 Yuki Shindo; full license and originals are in h3_author/upstream. Read h3_author/README.md and separate model license before enabling a run.\n\nVersion 2 completed native 20-step GPU inference and media-structure checks: 864×480, 124 frames, 24 fps, 32 kHz stereo. This is mixed high-precision parameter storage, NOT all-BF16: DiT and H3 conditioner BF16 plus FP32 elements, video VAE FP16, audio VAE FP32. Compute dtype and true peak memory were not measured. Original MP4/ZIP were not recovered; independent audio listening and file-hash verification remain incomplete.\n\nSettings default OFF; CPU mode skips all weights and model execution. Read https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE . Hosting region remained unknown in the observed run, not legal verification. No Context-IR/Regenerate2K. No public tunnel, Drive mount, token access, or custom nodes. Local result export is not a public upload.\n\nRun settings, setup, startup, then shutdown. CPU budget 12 minutes; GPU budget up to 50 minutes from setup start. Do not extend it, supervise progress and verify runtime release. Software gates cannot establish legal eligibility or guarantee shutdown after kernel failure.\n'),
 cell('code',(r/'config.py').read_text()),cell('code',setup_cell),cell('code',start_cell),
 cell('code',"# Release only this notebook's Colab runtime. Verify Manage sessions afterwards.\nsave_result()\nrelease_runtime('operator finished review')\n") ]}
(r/'minimax_h3_author_bf16.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
manifest={'program_source_sha256':source_hash,'files':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['minimax_h3_author_bf16.ipynb','cell_01.py','cell_02.py','minimax_h3_t2v.json']}}
(r/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
