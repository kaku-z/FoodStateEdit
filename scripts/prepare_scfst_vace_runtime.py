"""Copy an E7 runtime and add an opt-in SCFST residual hook.

The source runtime is never modified. The hook is dormant unless a runner
attaches ``state_transfer_adapter`` and aligned conditions to a Wan DiT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


STEP_OLD = '''            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)'''
STEP_NEW = '''            # Inference
            setattr(models["dit"], "_scfst_global_step", global_step)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)'''

REF_OLD = '''    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    freqs = torch.cat(['''
REF_NEW = '''    # Reference image
    state_transfer_memory = None
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        state_transfer_memory = reference_latents
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    # VACE image-to-video runs do not populate ``reference_latents`` above.
    # Their reference image is the first temporal slice of vace_context:
    # channels 0:16 are the VAE source latents, 16:32 are the zero partner,
    # and the remaining channels encode the VACE mask.  Use only the actual
    # source latents as provenance memory; never infer them from generated x.
    if state_transfer_memory is None and vace_context is not None:
        expected_channels = int(getattr(dit, "state_transfer_reference_channels", 16))
        if vace_context.ndim != 5 or vace_context.shape[1] < expected_channels:
            raise RuntimeError("SCFST cannot extract VACE reference latent memory")
        state_transfer_memory = vace_context[:, :expected_channels, 0]

    state_transfer_adapter = getattr(dit, "state_transfer_adapter", None)
    state_transfer_conditions = None
    state_transfer_active = None
    if state_transfer_adapter is not None:
        if use_unified_sequence_parallel:
            raise RuntimeError("SCFST v1 does not support sequence parallelism")
        if state_transfer_memory is None:
            raise RuntimeError("SCFST requires reference or VACE source latents")
        state_transfer_adapter = state_transfer_adapter.to(device=x.device, dtype=x.dtype)
        state_transfer_conditions = getattr(dit, "state_transfer_conditions", None)
        state_transfer_active = getattr(dit, "state_transfer_active", None)
        if state_transfer_conditions is None or state_transfer_active is None:
            raise RuntimeError("SCFST conditions were not attached")
        if state_transfer_conditions.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"SCFST token mismatch: {state_transfer_conditions.shape[1]} vs {x.shape[1]}"
            )
        state_transfer_conditions = state_transfer_conditions.to(x.device)
        state_transfer_active = state_transfer_active.to(x.device)
        if state_transfer_conditions.shape[0] == 1 and x.shape[0] > 1:
            state_transfer_conditions = state_transfer_conditions.expand(x.shape[0], -1, -1)
            state_transfer_active = state_transfer_active.expand(x.shape[0], -1)
        if state_transfer_conditions.shape[0] != x.shape[0]:
            raise RuntimeError("SCFST batch mismatch")
        state_transfer_memory = state_transfer_memory.to(device=x.device, dtype=x.dtype)
        if state_transfer_memory.shape[0] == 1 and x.shape[0] > 1:
            state_transfer_memory = state_transfer_memory.expand(x.shape[0], -1, -1, -1)

    freqs = torch.cat(['''

BLOCK_OLD = '''            # Animate
            if pose_latents is not None and face_pixel_values is not None:'''
BLOCK_NEW = '''            # Optional source-consistent state-transfer residual.
            if state_transfer_adapter is not None and block_id in getattr(dit, "state_transfer_block_ids", ()):
                before = x
                x, diagnostics = state_transfer_adapter(
                    x, state_transfer_memory, state_transfer_conditions,
                    state_transfer_active,
                    strength=float(getattr(dit, "state_transfer_strength", 1.0)),
                )
                trace = getattr(dit, "state_transfer_trace", None)
                if trace is not None:
                    gate = diagnostics["source_gate"]
                    trace.append({
                        "step": int(getattr(dit, "_scfst_global_step", -1)),
                        "block": int(block_id),
                        "branch": str(getattr(dit, "state_transfer_branch", "unknown")),
                        "tokens": int(x.shape[1]),
                        "active_tokens": int(diagnostics["active_tokens"]),
                        "source_gate_nonzero": int((gate > 0).sum().item()),
                        "max_abs_delta": float((x - before).abs().max().item()),
                    })

            # Animate
            if pose_latents is not None and face_pixel_values is not None:'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',type=Path,required=True)
    p.add_argument('--source-run-config',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args(); bundle=a.bundle.resolve()
    base=json.loads(a.base_config.read_text(encoding='utf-8'))
    source=Path(base['runtime']['geoedit_root'])
    target=bundle/'runtime'
    if target.exists():
        raise FileExistsError(target)
    rel=Path('diffsynth/pipelines/wan_video.py')
    code=(source/rel).read_text(encoding='utf-8')
    for old,name in ((STEP_OLD,'step'),(REF_OLD,'reference'),(BLOCK_OLD,'block')):
        if code.count(old)!=1:
            raise ValueError(f'expected one {name} patch site, found {code.count(old)}')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','.git'))
    patched=code.replace(STEP_OLD,STEP_NEW).replace(REF_OLD,REF_NEW).replace(BLOCK_OLD,BLOCK_NEW)
    (target/rel).write_text(patched,encoding='utf-8')
    base['runtime']['geoedit_root']=str(target)
    base['runtime']['geoedit_files'][rel.as_posix()]=sha256(target/rel)
    out=bundle/'scfst_runtime_config.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(base,indent=2),encoding='utf-8')
    run_config=json.loads(a.source_run_config.read_text(encoding='utf-8'))
    run_config['base_config']=str(out)
    run_config['base_config_sha256']=sha256(out)
    run_config['claim_limit']=(
        'SCFST zero-initialized wiring run on the frozen E7 cake case. '
        'This validates execution only and cannot show an algorithmic gain.'
    )
    run_out=bundle/'scfst_run_config.json'
    if run_out.exists():
        raise FileExistsError(run_out)
    run_out.write_text(json.dumps(run_config,indent=2),encoding='utf-8')
    receipt={'source_config':str(a.base_config.resolve()),'source_runtime':str(source),
             'source_pipeline_sha256':sha256(source/rel),'patched_pipeline_sha256':sha256(target/rel),
             'derived_config_sha256':sha256(out),'run_config_sha256':sha256(run_out),
             'source_run_config':str(a.source_run_config.resolve()),
             'preparer_sha256':sha256(Path(__file__)),
             'default_behavior':'unchanged unless state_transfer_adapter is attached'}
    (bundle/'runtime_derivation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
