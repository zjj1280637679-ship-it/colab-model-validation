AUTHOR_WORKFLOW = {'6': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_fl2va_bf16.safetensors', 'weight_dtype': 'default'}}, '9': {'class_type': 'BasicScheduler', 'inputs': {'model': ['6', 0], 'scheduler': 'simple', 'steps': 20, 'denoise': 1}}, '10': {'class_type': 'VAEDecode', 'inputs': {'samples': ['14', 0], 'vae': ['11', 0]}}, '11': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}}, '13': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_bf16.safetensors', 'type': 'minimax', 'device': 'default'}}, '14': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'noise': ['15', 0], 'guider': ['16', 0], 'sampler': ['17', 0], 'sigmas': ['9', 0], 'latent_image': ['104', 1]}}, '15': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 42}}, '16': {'class_type': 'BasicGuider', 'inputs': {'model': ['6', 0], 'conditioning': ['104', 0]}}, '17': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'res_multistep'}}, '23': {'class_type': 'VAEDecodeAudio', 'inputs': {'samples': ['14', 0], 'vae': ['24', 0]}}, '24': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'}}, '91': {'class_type': 'CreateVideo', 'inputs': {'images': ['10', 0], 'audio': ['23', 0], 'fps': 24}}, '92': {'class_type': 'SaveVideo', 'inputs': {'video': ['91', 0], 'filename_prefix': 'video/MiniMax_H3_T2V', 'format': 'auto', 'codec': 'auto'}}, '104': {'class_type': 'MiniMaxH3ImageToVideo', 'inputs': {'clip': ['13', 0], 'vae': ['11', 0], 'prompt': 'A cinematic wide shot of a paper lantern floating over a quiet lake at blue hour. Gentle ripples, slow camera push-in. Audio: soft wind, water ambience, and a distant bell.', 'width': 864, 'height': 480, 'length': 124}}}
# Adapted from shinshin86/comfy-agent scripts/colab/02_start_comfyui.py.
# Keep native ComfyUI; replace only public tunnel with Colab's authenticated proxy.
import socket, urllib.request, urllib.error, hashlib, zipfile
PORT = 8188
LOG_FILE = RUN_ROOT / 'comfy.log'

def _wait_port(port, timeout=180.0):
    deadline = min(DEADLINE, time.monotonic() + timeout)
    while time.monotonic() < deadline:
        if comfy_proc.poll() is not None: return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0: return True
        time.sleep(0.5)
    return False

def api(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request('http://127.0.0.1:' + str(PORT) + path, data=data,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

def validate_author_graph(graph, object_info):
    # Structure/types/ranges against the real imported node schemas. This does NOT
    # instantiate models. CPU deliberately skips exactly the 4 missing model files.
    skipped = []
    edges = []
    model_fields = {('6','unet_name'), ('13','clip_name'), ('11','vae_name'), ('24','vae_name')}
    for node_id, node in graph.items():
        kind = node['class_type']
        if kind not in object_info: raise RuntimeError('Missing native node: ' + kind)
        schema = object_info[kind]
        required = schema.get('input', {}).get('required', {})
        allowed = {**required, **schema.get('input', {}).get('optional', {})}
        values = node['inputs']
        if set(required) - set(values): raise RuntimeError('Missing required fields: ' + node_id)
        if set(values) - set(allowed): raise RuntimeError('Unknown node fields: ' + node_id)
        for name, value in values.items():
            spec = allowed[name]
            expected = spec[0]
            if isinstance(value, list):
                if len(value) != 2 or value[0] not in graph or not isinstance(value[1], int):
                    raise RuntimeError('Invalid graph connection')
                source = object_info[graph[value[0]]['class_type']].get('output', [])
                if not 0 <= value[1] < len(source) or source[value[1]] != expected:
                    raise RuntimeError('Graph connection type mismatch: ' + node_id + '.' + name)
                edges.append((value[0], node_id))
            elif (node_id, name) in model_fields and MODE == 'CPU_NO_WEIGHTS':
                skipped.append(node_id + '.' + name + ': file existence SKIPPED_NO_WEIGHTS')
            elif isinstance(expected, list):
                if value not in expected: raise RuntimeError('Unsupported selection: ' + node_id + '.' + name)
            else:
                ok = {'STRING': isinstance(value, str), 'INT': isinstance(value,int) and not isinstance(value,bool),
                      'FLOAT': isinstance(value,(int,float)) and not isinstance(value,bool), 'BOOLEAN': isinstance(value,bool)}
                if expected in ok and not ok[expected]: raise RuntimeError('Wrong primitive type: ' + node_id + '.' + name)
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if isinstance(value, (int,float)) and not isinstance(value,bool):
                    if ('min' in opts and value < opts['min']) or ('max' in opts and value > opts['max']):
                        raise RuntimeError('Numeric input out of range: ' + node_id + '.' + name)
    # Cycle check using Kahn's algorithm.
    remaining = set(graph)
    while remaining:
        ready = {x for x in remaining if not any(dst == x and src in remaining for src,dst in edges)}
        if not ready: raise RuntimeError('Workflow has a cycle')
        remaining -= ready
    return {'node_count':len(graph), 'connection_count':len(edges), 'schema_check':'PASSED',
            'skipped_model_existence_checks':skipped,
            'note':'Real /object_info inspection, not ComfyUI execution validation or inference.'}

if RESULT['status'] != 'SETUP_READY': raise RuntimeError('Run the setup cell successfully first')
try:
    with socket.socket() as check_socket:
        if check_socket.connect_ex(('127.0.0.1', PORT)) == 0:
            raise RuntimeError('Port already in use; will not reuse or stop another session')
    command = [sys.executable, 'main.py', '--dont-print-server', '--listen', '127.0.0.1',
               '--port', str(PORT), '--disable-all-custom-nodes', '--disable-api-nodes']
    command += ['--cpu'] if MODE == 'CPU_NO_WEIGHTS' else ['--bf16-unet', '--bf16-text-enc']
    comfy_proc = subprocess.Popen(command, cwd=WORKSPACE, stdout=LOG_FILE.open('w'), stderr=subprocess.STDOUT)
    if not _wait_port(PORT): raise RuntimeError('Native ComfyUI failed to become ready')
    object_info = api('/object_info')
    RESULT['workflow_validation'] = validate_author_graph(AUTHOR_WORKFLOW, object_info)
    RESULT['workflow_sha256'] = hashlib.sha256(json.dumps(AUTHOR_WORKFLOW,sort_keys=True).encode()).hexdigest()
    RESULT['workflow_parameters'] = {'width':AUTHOR_WORKFLOW['104']['inputs']['width'], 'height':AUTHOR_WORKFLOW['104']['inputs']['height'],
                                     'length':AUTHOR_WORKFLOW['104']['inputs']['length'], 'steps':AUTHOR_WORKFLOW['9']['inputs']['steps'], 'seed':42}
    print('COMFY_STARTUP_LOG\n' + safe_text(LOG_FILE.read_text()), flush=True)
    from google.colab import output
    output.serve_kernel_port_as_window(PORT)
    if MODE == 'CPU_NO_WEIGHTS':
        RESULT.update(status='CPU_NATIVE_STARTUP_AND_GRAPH_SCHEMA_PASSED',
                      skipped=['all H3 weight downloads', 'model file existence checks', 'native /prompt execution validation',
                               'all model loading', 'parameter dtype audit', 'sampling', 'H3 media generation/verification'])
        save_result()
        print('CPU: inspect the authenticated ComfyUI window, then run the shutdown cell. The single deadline remains active.', flush=True)
    else:
        submitted = api('/prompt', {'prompt':AUTHOR_WORKFLOW, 'client_id':'h3-author-single-run'})
        prompt_id = submitted.get('prompt_id')
        if not prompt_id or submitted.get('node_errors'): raise RuntimeError('Native prompt validation rejected the graph')
        started = time.monotonic()
        log_position = 0
        while time.monotonic() < DEADLINE - 30:
            contents = LOG_FILE.read_text(errors='replace')
            if len(contents) > log_position:
                print(safe_text(contents[log_position:]), end='', flush=True); log_position = len(contents)
            history = api('/history/' + prompt_id)
            if prompt_id in history:
                item = history[prompt_id]
                RESULT['execution_status'] = item.get('status')
                RESULT['outputs'] = item.get('outputs')
                if item.get('status',{}).get('status_str') != 'success': raise RuntimeError('Native execution failed; see complete ComfyUI log')
                break
            if comfy_proc.poll() is not None: raise RuntimeError('ComfyUI process exited during inference')
            time.sleep(2)
        else: raise TimeoutError('Single lifecycle budget nearly exhausted')
        RESULT['inference_seconds'] = time.monotonic() - started
        rows = [json.loads(x) for x in (RUN_ROOT/'precision.jsonl').read_text().splitlines()]
        RESULT['precision'] = rows
        if {x['component'] for x in rows} != {'transformer','text_encoder','video_vae','audio_vae'}:
            raise RuntimeError('Missing native component precision evidence')
        RESULT.update(model_loaded=True, inference_executed=True, status='NATIVE_INFERENCE_COMPLETED_MEDIA_REVIEW_REQUIRED')
        # Inspect exactly the original SaveVideo prefix in this fresh workspace.
        videos = list((pathlib.Path(WORKSPACE)/'output'/'video').glob('MiniMax_H3_T2V_*.mp4'))
        if len(videos) != 1 or videos[0].is_symlink(): raise RuntimeError('Expected one original workflow MP4')
        video_path = videos[0]
        import av, numpy as np
        with av.open(str(video_path)) as container:
            vs, aus = list(container.streams.video), list(container.streams.audio)
            if len(vs) != 1 or len(aus) != 1: raise RuntimeError('Expected one video and one audio stream')
            media = {'width':vs[0].width, 'height':vs[0].height, 'fps':float(vs[0].average_rate),
                     'sample_rate':aus[0].rate, 'channels':aus[0].channels}
        frames = 0
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                values = frame.to_ndarray(format='rgb24')
                if not np.isfinite(values).all(): raise RuntimeError('Nonfinite decoded video')
                frames += 1
        total, squares, peak = 0, 0.0, 0.0
        with av.open(str(video_path)) as container:
            for frame in container.decode(audio=0):
                raw = frame.to_ndarray(); values = raw.astype(np.float64)
                if np.issubdtype(raw.dtype, np.integer): values /= max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
                if not np.isfinite(values).all(): raise RuntimeError('Nonfinite decoded audio')
                total += values.size; squares += float(np.square(values).sum()); peak = max(peak,float(np.abs(values).max()))
        media.update(frames=frames, decoded_audio_rms=(squares/total)**0.5 if total else 0, decoded_audio_peak=peak,
                     sha256=sha256_file(video_path), hash_scope='runtime bytes only; no independent download verification')
        if (media['width'],media['height'],frames,media['fps'],media['sample_rate'],media['channels']) != (864,480,124,24.0,32000,2):
            raise RuntimeError('Decoded media does not match the author baseline')
        if media['decoded_audio_rms'] <= 1e-7 or peak <= 1e-6: raise RuntimeError('Silent decoded audio')
        RESULT.update(media=media, status='NATIVE_INFERENCE_AND_MEDIA_STRUCTURE_PASSED_QUALITY_REVIEW_REQUIRED')
        save_result()
        archive = RUN_ROOT/'h3_author_result.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            # Explicit local-only whitelist. No weights, caches, tokens, Drive or public uploads.
            for path in [RUN_ROOT/'result.json', RUN_ROOT/'precision.jsonl', video_path]:
                if path.is_symlink() or not path.is_file(): raise RuntimeError('Unsafe export file')
                bundle.write(path,arcname=path.name)
        from IPython.display import display, Video
        display(Video(str(video_path), embed=True))
        from google.colab import files
        files.download(str(archive))
        print('Result ZIP download requested, not independently verified. Visual/audio quality remains for human review.', flush=True)
        release_runtime('single native inference completed')
except Exception as error:
    RESULT.update(status='FAILED', error_type=type(error).__name__, error=safe_text(error))
    if LOG_FILE.exists(): print('COMPLETE_COMFY_LOG\n' + safe_text(LOG_FILE.read_text(errors='replace')), flush=True)
    save_result()
    release_runtime('startup/validation/execution failure')
    raise
