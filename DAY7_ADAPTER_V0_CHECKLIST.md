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
- [ ] Transfer it to a new remote directory and verify every hash.
- [ ] Require one A6000 with at least 48,000 MiB free, <=5% utilization, and no compute process.
- [ ] Run exactly one epoch over two samples with rank-8 high-noise VACE LoRA.
- [ ] Preserve command, log, preflight, failure state, and checkpoint hashes.
- [ ] Load the resulting LoRA in an offline validation smoke before any visual comparison.

## Current resource state

At the initial 2026-08-29 preflight, all eight A6000 GPUs were occupied by another user's eight-GPU training job at approximately 44.5 GiB per card and near 100% utilization. No training was started.
