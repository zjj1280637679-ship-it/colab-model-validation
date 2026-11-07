# Non-commercial technical evaluation under the Qwen Research License.
# Official checkpoint; every learned component remains BF16 (no INT8/FP8/NVFP4).
import subprocess, sys, importlib, time, json, collections, hashlib, gc
if "p" in globals():
    del p
    gc.collect()
    torch.cuda.empty_cache()
print("GPU_ALLOCATED_BEFORE_IMAGE_GIB",round(torch.cuda.memory_allocated()/2**30,3))
assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/2**30 > 70, "STOP: A100 80GB required before model download."
DIFFUSERS_COMMIT="84532a058702c74ae5d1dd6ceb05683fc5d5e768"
assert not any(name == "diffusers" or name.startswith("diffusers.") for name in sys.modules), "STOP: diffusers already imported; restart runtime and rerun from bootstrap to avoid stale modules."
subprocess.check_call([sys.executable,"-m","pip","install","-q","git+https://github.com/huggingface/diffusers.git@"+DIFFUSERS_COMMIT])
installed_diffusers = observed_distribution("diffusers")
assert installed_diffusers and installed_diffusers["source"] and installed_diffusers["source"]["commit"] == DIFFUSERS_COMMIT, "STOP: installed Diffusers source commit does not match expected pin."
from diffusers import QwenImage21Pipeline
import diffusers, numpy as np
MODEL_ID=PROFILE["repo"]
REVISION=PROFILE["revision"]
assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
assert torch.cuda.memory_allocated()/2**30 < 2, "STOP: previous model still occupies GPU."
start=time.monotonic()
torch.cuda.reset_peak_memory_stats()
pipe=QwenImage21Pipeline.from_pretrained(MODEL_ID,revision=REVISION,dtype=torch.bfloat16,token=False).to("cuda")
components={}
for name,module in pipe.components.items():
    if isinstance(module,torch.nn.Module):
        dtype_count=collections.Counter()
        device_count=collections.Counter()
        for p in module.parameters():
            dtype_count[str(p.dtype)]+=p.numel()
            device_count[str(p.device)]+=p.numel()
        components[name]={"parameters_by_dtype":dict(dtype_count),"parameters_by_device":dict(device_count),"quantized":bool(getattr(module,"is_quantized",False))}
        assert set(dtype_count)=={"torch.bfloat16"}, f"STOP: unexpected dtype in {name}"
        assert set(device_count)=={"cuda:0"}, f"STOP: off-device weights in {name}"
        assert not components[name]["quantized"], f"STOP: quantization in {name}"
print("IMAGE_COMPONENTS:",json.dumps(components,indent=2))
image_result={"model":MODEL_ID,"revision":REVISION,"diffusers_version":diffusers.__version__,"diffusers_commit":installed_diffusers["source"]["commit"],"diffusers_expected_commit":DIFFUSERS_COMMIT,"diffusers_source":installed_diffusers["source"],"gpu":torch.cuda.get_device_name(0),"components":components,"load_seconds":round(time.monotonic()-start,2),"tests":[]}
steps_seen=[]
def validate_latents(pipeline, step, timestep, callback_kwargs):
    if step%5==0 or step==39:
        assert torch.isfinite(callback_kwargs["latents"]).all().item(), "STOP: non-finite latents"
    steps_seen.append(int(step))
    return callback_kwargs
prompt='A photorealistic red ceramic teapot on a wooden table beside a small green plant, soft morning window light, crisp detail, a white card clearly labeled "A100 BF16 TEST".'
start=time.monotonic()
image=pipe(prompt=prompt,width=2048,height=2048,num_inference_steps=40,true_cfg_scale=1.0,generator=torch.Generator("cuda").manual_seed(42),callback_on_step_end=validate_latents,callback_on_step_end_tensor_inputs=["latents"]).images[0]
torch.cuda.synchronize()
path=REPORT_DIR/"qwen_image_2_1_bf16_2048.png"
image.save(path)
pixels=np.asarray(image)
assert image.size==(2048,2048)
assert np.isfinite(pixels).all() and float(pixels.std())>2, "STOP: invalid/constant image"
image_result["tests"].append({"task":"text_to_image","prompt":prompt,"width":image.width,"height":image.height,"mode":image.mode,"requested_steps":40,"actual_steps":len(steps_seen),"seed":42,"seconds":round(time.monotonic()-start,2),"pixel_std":float(pixels.std()),"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
image_result["peak_allocated_gib"]=round(torch.cuda.max_memory_allocated()/2**30,3)
image_result["peak_reserved_gib"]=round(torch.cuda.max_memory_reserved()/2**30,3)
image_result["status"]="INFERENCE_COMPLETED_PENDING_VISUAL_AND_INDEPENDENT_REVIEW"
(REPORT_DIR/"qwen_image_2_1_bf16.json").write_text(json.dumps(image_result,ensure_ascii=False,indent=2))
print("QWEN_IMAGE_BF16_RESULT:",json.dumps(image_result,ensure_ascii=False,indent=2))
display(image)
del pipe,module,p
gc.collect()
torch.cuda.empty_cache()
print("IMAGE_WEIGHTS_UNLOADED_GPU_ALLOCATED_GIB",round(torch.cuda.memory_allocated()/2**30,3))

