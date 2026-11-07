"""Generate two self-contained, output-free Colab notebooks; no GPU use."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def cell(kind,text):
    c={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code': c.update(execution_count=None,outputs=[])
    return c

def build():
    for model in ('qwen3_8','qwen_image_2_1'):
        image=model=='qwen_image_2_1'
        cells=[cell('markdown','# BF16 Colab validation\nA100 80GB required. Read repository README and THIRD_PARTY_NOTICES before running. Qwen Image is research-only. Run cells in order. This notebook is generated from reviewed scripts; original evidence is separate. No Drive or token required.\n')]
        cells.append(cell('code','import threading\nfrom google.colab import runtime\nshutdown_timer=threading.Timer(25*60, runtime.unassign)\nshutdown_timer.daemon=True\nshutdown_timer.start()\nprint("25-minute best-effort runtime shutdown armed. Export results promptly.")\n'))
        cells.append(cell('code','%pip install -q transformers==5.22.1 accelerate==1.15.0 huggingface_hub==1.33.0 safetensors==0.8.0\n'))
        cells.append(cell('markdown','If Colab asks to restart the session after installation, restart and rerun from the timer cell. Keep Colab supplied PyTorch. Validated environment was Python 3.13.16 / torch 2.14.1+cu130; other environments are not validated.\n'))
        for n in ['preflight.py',model+'_bf16.py']+([] if image else ['audit_checkpoint.py']): cells.append(cell('code',(ROOT/'scripts'/n).read_text()))
        cells.append(cell('markdown','Download the ZIP before shutdown. If download fails, use Colab Files to download JSON/PNG manually. Do not leave paid GPU running while troubleshooting export. A PNG filename/hash in JSON does not prove the file was exported.\n'))
        cells.append(cell('code',(ROOT/'scripts/export_evidence.py').read_text()))
        cells.append(cell('code','from google.colab import runtime\nruntime.unassign()\n'))
        for i,c in enumerate(cells):c['id']=f'{model}-{i:02d}'
        n={'cells':cells,'metadata':{'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
        (ROOT/'notebooks'/f'{model}_bf16.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':build()
