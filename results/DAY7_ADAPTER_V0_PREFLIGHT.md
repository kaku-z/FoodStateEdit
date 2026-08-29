# Day 7 VACE-LoRA adapter v0 preflight

## Decision

Freeze a two-sample, high-noise VACE-LoRA run as an infrastructure smoke test.
It may establish that the existing offline model can load the FoodStateEdit
paired-control format, backpropagate through the VACE branch, and save a LoRA
checkpoint. It cannot establish visual improvement, action generalization, or
photo realism because its targets are identical to deterministic proxy controls.

## Trainer selection

- Framework: official DiffSynth-Studio Wan2.2 VACE-LoRA trainer.
- Frozen commit: `899d2cd5740f50f7b6a1ec2cf7360a20897b1191`.
- Source archive SHA-256:
  `68344f19db71b4e8b0616d0a3aec436fddcc545d9ce03396ad797f29a1b32228`.
- Training entry-point SHA-256:
  `4a8c9c27c23b217f7438b37c71213df58414c39cb2b6998d3d75a4dd7992bc41`.
- Selection reason: the newest inspected upstream snapshot failed under the
  existing PyTorch 2.4 environment because it imported
  `torch.nn.attention.flex_attention`. The frozen earlier commit retains the
  official Wan2.2 VACE-LoRA path and passed an offline `--help` import smoke.
- Existing GeoEdit worktree was not modified.

## Frozen dataset

The dataset contains two 21-frame plumbing-only samples:

| Sample | Family/action | Utensil | Target policy |
| --- | --- | --- | --- |
| `ramen_chopsticks_001_proxy_identity_smoke` | strand/lift | chopsticks | proxy target equals VACE control |
| `pasta_fork_001_proxy_identity_smoke` | strand-contact/twirl-and-lift | fork | proxy target equals VACE control |

- Dataset manifest SHA-256:
  `d250726a2d6a752dc934b5d7492f58cd9387eff7ecf9c6f97d32dd3cde8dc9d9`.
- Metadata SHA-256:
  `9e4c4fb663a01707cf91aa5bdb24c0b2b392e802f47dea37d73958df257c7652`.
- Every media file records a byte size and SHA-256 in the manifest.
- Preflight requires metadata paths to agree with the manifest, validates every
  media size and hash, and confirms that each identity-smoke target/control pair
  is byte-identical.
- A failed initial materialization caused by selecting the wrong manifest root
  is preserved in the ignored local artifact directory
  `artifacts/day7_adapter_v0_smoke_dataset_failed_missing_manifest_20260829`.

## Frozen run and safety gate

- Base: existing audited `Wan2.2-VACE-Fun-A14B` high-noise model; downloads are
  disabled.
- Adapted branch: `vace`; LoRA modules `q,k,v,o,ffn.0,ffn.2`; rank 8.
- Schedule: one epoch, two samples, one repeat, one GPU, BF16, two optimizer
  steps expected.
- Output: new-only
  `/tmp/foodstateedit_day7_adapter_v0_high_noise_lora_smoke_v1`.
- GPU gate: at least 48,000 MiB free, at most 5% utilization, and no compute
  process on the selected A6000.
- Host-memory gate: at least 80,000 MiB available.
- Any preflight or training failure is preserved; no existing directory or
  other user's process may be changed.

## Initial resource observation

On 2026-08-29, all eight A6000 GPUs were occupied by another user's eight-GPU
job at roughly 44.5 GiB per card and near 100% utilization. No adapter training
was launched. A fresh remote preflight is required after the exact dataset and
code snapshot are transferred.

## Generalization gate after smoke

A successful two-step run only opens the data pipeline. The first publishable
adapter candidate requires a disjoint paired-photo dataset covering all four
families (liquid, granular, strand, strand-contact), multiple dishes and camera
conditions per utensil/action, held-out food instances, and separate reporting
of action correctness, photo realism, source removal, and protected-region
preservation. Deterministic geometry and final 2D protection remain outside the
LoRA.
