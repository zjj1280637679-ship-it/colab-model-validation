# Main generation path BF16; MTP auxiliary branch is not loaded or enabled.
# Transformers omits the optional MTP branch; audit_checkpoint.py records this limitation.
import os, time, json, gc, collections, importlib.metadata
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText
MODEL_ID=PROFILE["repo"]
REVISION=PROFILE["revision"]
assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_properties(0).total_memory/2**30 > 70
config=AutoConfig.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False,token=False)
assert not getattr(config,"quantization_config",None), "STOP: quantized checkpoint."
print("MODEL:",MODEL_ID,"REVISION:",REVISION)
print("ARCHITECTURE:",config.architectures,"MODEL_TYPE:",config.model_type)
print("VERSIONS:",{p:importlib.metadata.version(p) for p in ["torch","transformers","accelerate","huggingface_hub","safetensors"]})
started=time.monotonic()
torch.cuda.reset_peak_memory_stats()
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,revision=REVISION,dtype=torch.bfloat16,device_map={"":0},trust_remote_code=False,token=False,attn_implementation="sdpa")
model.eval()
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False,token=False)
dtype_counts=collections.Counter()
device_counts=collections.Counter()
for p in model.parameters():
    dtype_counts[str(p.dtype)]+=p.numel()
    device_counts[str(p.device)]+=p.numel()
print("LOADED_PARAMETERS_BY_DTYPE:",dict(dtype_counts))
print("LOADED_PARAMETERS_BY_DEVICE:",dict(device_counts))
print("QUANTIZED:",getattr(model,"is_quantized",False))
assert not getattr(model,"is_quantized",False)
assert set(dtype_counts)=={"torch.bfloat16"}, "STOP: unexpected parameter dtype."
assert set(device_counts)=={"cuda:0"}, "STOP: not fully on A100."
results={"model":MODEL_ID,"revision":REVISION,"gpu":torch.cuda.get_device_name(0),"dtype_counts":dict(dtype_counts),"device_counts":dict(device_counts),"quantized":False,"load_seconds":round(time.monotonic()-started,2),"tests":[]}
for prompt in ["请用一句中文解释什么是 Google Colab。","计算 17 × 23，只输出整数。","Write a Python function named square that returns x multiplied by itself."]:
    encoded=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt").to("cuda")
    t=time.monotonic()
    with torch.inference_mode():
        output=model.generate(**encoded,max_new_tokens=128,do_sample=False,use_cache=True)
    torch.cuda.synchronize()
    generated=output[0,encoded["input_ids"].shape[1]:]
    answer=tokenizer.decode(generated,skip_special_tokens=True)
    row={"prompt":prompt,"answer":answer,"new_tokens":len(generated),"seconds":round(time.monotonic()-t,2)}
    assert answer.strip(),"STOP: empty generation."
    results["tests"].append(row)
    print("TEST:",json.dumps(row,ensure_ascii=False))
results["peak_allocated_gib"]=round(torch.cuda.max_memory_allocated()/2**30,3)
results["peak_reserved_gib"]=round(torch.cuda.max_memory_reserved()/2**30,3)
results["status"]="INFERENCE_COMPLETED_PENDING_INDEPENDENT_REVIEW"
(REPORT_DIR/"qwen3_8_bf16.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
print("QWEN_BF16_RESULT:",json.dumps(results,ensure_ascii=False,indent=2))
del model,tokenizer,encoded,output,generated,p
gc.collect()
torch.cuda.empty_cache()
print("QWEN_WEIGHTS_UNLOADED_GPU_ALLOCATED_GIB",round(torch.cuda.memory_allocated()/2**30,3))

