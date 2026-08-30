# Day 7 FoodStateEdit VACE-LoRA adapter v0 checklist

## Scientific scope

- [x] Keep deterministic geometry, contact topology, utensil count, and final 2D protection outside the learned model.
- [x] Use LoRA only on the VACE control branch for the first experiment.
- [x] Label the two-anchor identity-target run as infrastructure smoke, not quality or generalization evidence.
- [ ] Build a separate paired-photo training set before any publishable fine-tuning claim.

## Frozen implementation

- [x] Reuse the existing audited `Wan2.2-VACE-Fun-A14B` files; downloads are disabled.
- [x] Preserve the dirty GeoEdit checkout and use a separate DiffSynth source snapshot.
- [x] Freeze compatible DiffSynth commit `899d2cd5740f50f7b6a1ec2cf7360a20897b1191`.
- [x] Freeze a two-case strand/fork dataset contract and deterministic builder.
- [x] Require a read-only GPU/process/memory/model/dataset/trainer preflight.
- [x] Refuse to reuse output or preflight-report paths.

## Execution gate

- [x] Materialize and hash the local two-case smoke dataset.
- [x] Transfer it to a new remote directory and verify every hash.
- [x] Enforce one A6000 with at least 48,000 MiB free, <=5% utilization, and no compute process.
- [x] Run exactly one epoch over two samples with rank-8 high-noise VACE LoRA.
- [x] Preserve command, log, preflight, failure state, and checkpoint hashes.
- [x] Load the resulting LoRA in an offline validation smoke before any visual comparison.

The loadability smoke is frozen in `scripts/validate_adapter_lora_checkpoint.py`.
It requires complete finite rank-8 LoRA A/B pairs and a positive tensor-update
count from the official `pipe.load_lora(pipe.vace, ...)` path. This does not
count as a visual result.

## Current resource state

At the initial 2026-08-29 preflight, 51/52 checks passed. All eight A6000 GPUs were occupied by `chen-q` at approximately 44.5 GiB per card and 99--100% utilization, so `gpu_gate` was the sole failed check. No training was started and the frozen output directory remains absent.

The host pool was expanded to `gp38`--`gp42`. `gp39` had eight idle A6000s
and passed all 52 original checks, but the upstream trainer then failed before
model loading because it instantiated an unused audio operator and `librosa`
is not installed. The full v1 failure is preserved under
`results/day7_adapter_v0_gp39_failure_missing_librosa_v1/`; it produced no
checkpoint and did not occupy the GPU. The correction series uses a hash-frozen
no-audio wrapper and new output directories rather than installing a package or
overwriting a failure.

The first complete wrapper run is also preserved as v2: `LoadAudio.__init__`
looked up `librosa.load`, so a sentinel with only `ModuleSpec` was insufficient.
It again failed before model loading and produced no checkpoint. V3 supplies a
forbidden `load` callable that raises if audio is ever processed; the frozen
metadata contains no audio key, so the VACE path never calls it.

V3 passed preflight and loaded the frozen DiT, VACE, T5, and VAE components,
then failed at the first dataloader item. ImageIO 2.37.2 selected its PyAV
backend, whose container-level metadata calculation multiplied a missing
duration by the stream time base. Frame-level decoding itself remains valid.
The full failure is preserved under
`results/day7_adapter_v0_gp39_failure_pyav_metadata_v3/`. The first V4 decoder
smoke then showed that ImageIO's PyAV `LegacyReader` also omits the
`count_frames()` method expected by the frozen upstream trainer; it did not
start training or create an output directory. V5 supplies both metadata and
frame count from the same PyAV stream while leaving `get_data` unchanged, and
preflight now decodes the first and last frame of every training/control video
before model loading.

V5 completed both optimizer steps in 99.1 seconds on gp39 GPU 0 and saved
`step-1.safetensors` and `step-2.safetensors`. The final checkpoint contains
160 finite BF16 tensors forming 80 complete rank-8 LoRA A/B pairs. The offline
official `pipe.load_lora(pipe.vace, ...)` path updated 80 tensors. This closes
the infrastructure smoke only; the identity targets provide no action or photo
quality evidence.
