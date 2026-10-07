"""Load the observed-material LoRA without merging or changing backbone tensors."""
import hashlib
import json
from pathlib import Path


def load_material_binding_adapter(transformer, checkpoint, name='material_binding'):
    import torch
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    checkpoint=Path(checkpoint)
    config_path=checkpoint.parent/'frozen_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    modules=[f'control_blocks.{i}.attn.{n}' for i in range(16) for n in ['to_q','to_k','to_v','to_out.0']]
    assert config['modules']==modules
    rank=int(config['rank']);assert int(config['alpha'])==rank
    state=load_file(str(checkpoint),device='cpu')
    expected={f'{module}.lora_{factor}.weight':((rank,4096) if factor=='A' else (4096,rank))
              for module in modules for factor in ['A','B']}
    assert set(state)==set(expected),{'unexpected':sorted(set(state)-set(expected)),
                                      'missing':sorted(set(expected)-set(state))}
    assert all(tuple(state[k].shape)==shape and bool(torch.isfinite(state[k]).all()) for k,shape in expected.items())
    transformer.add_adapter(LoraConfig(r=rank,lora_alpha=rank,target_modules=modules,lora_dropout=0.,bias='none'),adapter_name=name)
    # Match FP32 adapter master weights used during training; matmuls follow the caller's autocast.
    for n,p in transformer.named_parameters():
        if f'.{name}.' in n and 'lora_' in n:p.data=p.data.float()
    result=set_peft_model_state_dict(transformer,state,adapter_name=name)
    assert not result.unexpected_keys,result.unexpected_keys
    assert not [k for k in result.missing_keys if 'lora_' in k],result.missing_keys
    actual=get_peft_model_state_dict(transformer,adapter_name=name)
    assert set(actual)==set(state)
    assert all(torch.equal(actual[k].detach().cpu().float(),state[k].float()) for k in state)
    transformer.set_adapter(name)
    transformer.requires_grad_(False)
    return dict(name=name,rank=rank,alpha=rank,tensors=len(state),
        parameters=sum(v.numel() for v in state.values()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        frozen_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        exact_loaded_tensor_verification=True,base_tensors_in_checkpoint=0,merged=False,
        scope='Self-supervised observed real-photo patch reconstruction; no first-bite paired supervision.')


def set_material_binding_enabled(transformer, enabled=True):
    import torch
    # Accelerate may have moved LoRA tensors while the pipeline was in inference mode.
    # PEFT's enable path temporarily sets requires_grad=True, so toggle in the same context.
    with torch.inference_mode():
        if enabled:transformer.enable_adapters()
        else:transformer.disable_adapters()
        transformer.requires_grad_(False)
