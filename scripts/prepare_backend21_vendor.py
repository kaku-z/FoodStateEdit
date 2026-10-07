"""Extract the pinned Qwen 2.1 implementation without unrelated video backends."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]/'outputs/first_bite_structure_20260930'

def main():
    src=ROOT/'backend21_sources';out=ROOT/'backend21_vendor';out.mkdir(exist_ok=False)
    package=out/'videox_fun';(package/'models').mkdir(parents=True);(package/'pipeline').mkdir()
    mapping=[]
    for n in ['models/qwenimage21_transformer2d.py','models/qwenimage21_transformer2d_control.py','models/qwenimage21_vae.py','pipeline/pipeline_qwenimage21_control.py']:
        b=(src/n).read_bytes()
        if n.startswith('pipeline/'):
            b=b.replace(b'from .pipeline_qwenimage import QwenImagePipelineOutput',b'from .output import QwenImagePipelineOutput')
        (package/n).write_bytes(b)
        mapping.append({'file':n,'upstream_sha256':hashlib.sha256((src/n).read_bytes()).hexdigest(),'runtime_sha256':hashlib.sha256(b).hexdigest()})
    (package/'__init__.py').write_text('"""Pinned, minimal Qwen 2.1 inference subset of VideoX-Fun."""\n')
    (package/'models/__init__.py').write_text('''from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
from .qwenimage21_transformer2d import QwenImage21KVCache,QwenImage21Transformer2DModel
from .qwenimage21_transformer2d_control import QwenImage21ControlTransformer2DModel
from .qwenimage21_vae import AutoencoderKLQwenImage21
''')
    (package/'pipeline/__init__.py').write_text('from .pipeline_qwenimage21_control import QwenImage21ControlPipeline\n')
    (package/'pipeline/output.py').write_text('''from dataclasses import dataclass
from typing import List,Union
import numpy as np
from PIL import Image
from diffusers.utils import BaseOutput
@dataclass
class QwenImagePipelineOutput(BaseOutput):
    images: Union[List[Image.Image],np.ndarray]
''')
    (package/'dist.py').write_text('''"""Single-process adapter. Distributed inference is deliberately unsupported."""
import torch
def check():
    assert not torch.distributed.is_initialized(), "Minimal inference subset supports a single process only"
def get_sequence_parallel_rank():check();return 0
def get_sequence_parallel_world_size():check();return 1
def sequence_parallel_all_gather(x,*args,**kwargs):check();return x
class QwenImage21MultiGPUsAttnProcessor:
    def __init__(self,*args,**kwargs):raise RuntimeError("Distributed attention is not enabled in this experiment")
''')
    (package/'models/attention_utils.py').write_text('''"""Exact SDPA fallback used by the pinned upstream attention function."""
import torch
def attention(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,q_scale=None,causal=False,window_size=(-1,-1),deterministic=False,dtype=torch.bfloat16,fa_version=None,attention_type=None,attn_mask=None):
    assert q_lens is None and k_lens is None and softmax_scale is None and q_scale is None and window_size==(-1,-1)
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=attn_mask,is_causal=causal,dropout_p=dropout_p).transpose(1,2).contiguous()
''')
    manifest={'upstream':'https://github.com/aigc-apps/VideoX-Fun','commit':'4b7b6402a1e0f0406bd6801fb66c0a00bd922621','files':mapping,
        'adaptations':['Minimal package initializers exclude unrelated models.','Dataclass output moved into a standalone module.','Single-process distribution interface rejects initialized distributed execution.','Attention uses the upstream exact SDPA fallback; no approximate or sparse attention.'],
        'code_claim':'Inference adapter code, not an original generative model. Base model and control branch are credited to their authors.'}
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'backend21_vendor.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print({'files':len(list(out.rglob('*.py')))})

if __name__=='__main__':main()
