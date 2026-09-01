# Day 11 phase-action overfit training

Date: 2026-09-01

## Outcome

The frozen phase-varying overfit training completed on gp40 GPU 0. The runner
used only the existing Wan2.2 high-noise VACE, T5, and VAE files with downloads
disabled. Its independent and runner-owned preflights both passed, and the
uploaded 19-file dataset was byte-identical to the local frozen dataset.

All 64 optimizer steps completed in 1,441.50 seconds. Checkpoints were saved at
steps 16, 32, 48, and 64; each is 15,354,160 bytes and has a distinct recorded
SHA-256. The output and preflight paths were new, so no earlier experiment was
overwritten.

The step-64 checkpoint also passed the offline structural and official-loader
gate: 160 finite BF16 tensors form 80 complete rank-8 A/B pairs, and the
official single-model VACE loader updated all 80 targets.

## Claim boundary

This is a training and checkpoint-validity result only. The two training
sequences are deterministic translated-patch pseudo-motion made from synthetic
ImageGen targets; they are not physical motion or real-video supervision. No
action, contact, photo-realism, generalization, or adapter-benefit claim follows
from successful optimization.

The blind fork case remains prohibited. The next gate is the preregistered
same-seed sweep over LoRA disabled plus steps 16, 32, 48, and 64 on both seen
phase-varying samples. Approach, contact, payload acquisition, lift, final hold,
and photo realism must be reviewed separately before considering broader data.

Machine-readable evidence:
`results/day11_phase_action_overfit_training_result_v1.json`.
