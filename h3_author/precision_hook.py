# Read-only summaries after native ComfyUI loaders. No tensor movement or dtype change.
def _h3_precision(component, module, options):
    import os, json, collections
    counts, devices, parameter_classes = collections.Counter(), collections.Counter(), collections.Counter()
    quantized = []
    for name, parameter in module.named_parameters():
        counts[str(parameter.dtype)] += parameter.numel()
        devices[str(parameter.device)] += parameter.numel()
        parameter_classes[type(parameter).__module__ + '.' + type(parameter).__name__] += 1
        if getattr(parameter, 'is_quantized', False) or any(x in type(parameter).__name__.lower() for x in ('quantized', 'int8', 'float8', 'fp8', 'nvfp')):
            quantized.append(name)
    quant_modules = [type(m).__name__ for m in module.modules() if any(x in type(m).__name__.lower() for x in ('quantized', 'int8', 'float8', 'fp8', 'nvfp'))]
    row = {'component': component, 'parameter_numel_by_dtype': dict(counts),
           'parameter_numel_by_device': dict(devices), 'parameter_classes': dict(parameter_classes),
           'quantized_parameter_count': len(quantized), 'quantized_module_classes': sorted(set(quant_modules)),
           'loader_options': {str(k): str(v) for k,v in options.items()},
           'scope': 'parameter storage at native-loader return, not autocast/compute dtype or a complete checkpoint audit'}
    print('H3_NATIVE_PRECISION ' + json.dumps(row, sort_keys=True), flush=True)
    with open(os.environ['H3_PRECISION_LOG'], 'a') as f: f.write(json.dumps(row) + '\n')
    allowed = {'torch.float16', 'torch.float32'} if component == 'video_vae' else ({'torch.float32'} if component == 'audio_vae' else {'torch.bfloat16', 'torch.float32'})
    if not counts or set(counts) - allowed or quantized or quant_modules:
        raise RuntimeError('Unexpected parameter precision or quantization: ' + component)
    if component in ('transformer','text_encoder') and not counts.get('torch.bfloat16'):
        raise RuntimeError('No BF16 parameters in ' + component)
