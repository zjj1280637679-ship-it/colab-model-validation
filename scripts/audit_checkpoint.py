# Read-only checkpoint/header audit. No tensors loaded, no GPU inference.
import json, math, collections, pathlib, gc
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText
audit_repo="huihui-ai/Huihui-Qwen3.8-27B-abliterated"
audit_revision="739e3c5b89849f6c238ce1e5b70008612ae42cdd"
index_path=pathlib.Path(hf_hub_download(audit_repo,"model.safetensors.index.json",revision=audit_revision,local_files_only=True,token=False))
index=json.loads(index_path.read_text())
checkpoint={}
checkpoint_dtype=collections.Counter()
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(str(index_path.parent/shard),framework="pt",device="cpu") as f:
        for key in f.keys():
            sl=f.get_slice(key)
            shape=list(sl.get_shape())
            dtype=sl.get_dtype()
            count=math.prod(shape)
            checkpoint[key]={"shape":shape,"numel":count,"dtype":dtype}
            checkpoint_dtype[dtype]+=count
audit_config=AutoConfig.from_pretrained(audit_repo,revision=audit_revision,local_files_only=True,trust_remote_code=False,token=False)
with init_empty_weights():
    meta_model=AutoModelForImageTextToText.from_config(audit_config,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation="sdpa")
meta_state={k:{"shape":list(v.shape),"numel":v.numel()} for k,v in meta_model.state_dict().items()}
checkpoint_only={k:checkpoint[k] for k in checkpoint.keys()-meta_state.keys()}
model_only={k:meta_state[k] for k in meta_state.keys()-checkpoint.keys()}
shape_mismatch={k:{"checkpoint":checkpoint[k],"model":meta_state[k]} for k in checkpoint.keys()&meta_state.keys() if checkpoint[k]["shape"]!=meta_state[k]["shape"]}
checkpoint_audit={"repo":audit_repo,"revision":audit_revision,"checkpoint_dtype_numel":dict(checkpoint_dtype),"checkpoint_total":sum(v["numel"] for v in checkpoint.values()),"meta_state_dict_total":sum(v["numel"] for v in meta_state.values()),"meta_parameters_total":sum(p.numel() for p in meta_model.parameters()),"meta_parameters_without_deduplication":sum(p.numel() for n,p in meta_model.named_parameters(remove_duplicate=False)),"checkpoint_only_total":sum(v["numel"] for v in checkpoint_only.values()),"checkpoint_only":checkpoint_only,"model_only":model_only,"shape_mismatch":shape_mismatch,"note":"Headers and meta tensors only. No real tensors loaded or inference repeated."}
(REPORT_DIR/"qwen_checkpoint_audit.json").write_text(json.dumps(checkpoint_audit,indent=2))
print("QWEN_CHECKPOINT_AUDIT:",json.dumps(checkpoint_audit,indent=2))
del meta_model,sl
gc.collect()
print("POST_AUDIT_GPU_ALLOCATED_GIB",round(torch.cuda.memory_allocated()/2**30,3))

