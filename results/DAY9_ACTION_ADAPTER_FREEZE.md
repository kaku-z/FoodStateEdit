# Day 9 action-supervised adapter pilot freeze

Date: 2026-08-31

## What is frozen

The next experiment is a mechanism pilot, not formal training. It uses two
non-identity, built-in-ImageGen pseudo-targets: one chopsticks/udon target and
one spoon/broth target. There are no real-photo action targets in the training
set. Pasta/fork is absent from every training row and remains the blind family.

Each target and its VACE control contain 21 static frames. The target, control,
and no-utensil reference are identical outside a bounded action alpha:

| sample | edit support | target/control changed pixels | target/control RGB MAE | outside max |
| --- | ---: | ---: | ---: | ---: |
| udon + chopsticks | 5.086% | 4.858% | 1.254 | 0 |
| broth + spoon | 6.961% | 6.391% | 3.101 | 0 |

The noodle control is the deterministic topology-authoritative edit. The spoon
control is a deliberately simplified, Day-8-like rigid/material proxy. Targets
are oracle pixels projected only inside the audited action support; raw oracle
backgrounds are never used as supervision.

## Rejected builds

- v1 stopped before its manifest because of a Python boolean typo. The partial
  directory is preserved and is not eligible for upload or training.
- v2 completed technically but changed roughly 96%--99.9% of pixels because it
  used the raw oracle frame. It failed the data contract and is not eligible.
- v3 is the first eligible dataset: both target and control have zero change
  outside their declared edit supports and their video hashes are non-identical.

## Training contract

- Existing offline Wan2.2 VACE-Fun A14B and frozen DiffSynth snapshot only;
- high-noise VACE LoRA, rank 8;
- learning rate `5e-5`, two samples, repeat 4, two epochs;
- 16 expected optimizer steps, checkpoints at steps 8 and 16;
- one idle NVIDIA RTX A6000 with at least 48,000 MiB free, utilization at most
  5%, no compute process, and at least 80,000 MiB host memory;
- no model downloads, no overwrite, and technical failures remain preserved.

## Blind evaluation contract

After checkpoint validation, run the unchanged Day 8 `pasta_fork_001` control
with the same seed, prompt, steps, VACE scale, TTM setting, selected frame, and
projection alpha. Only LoRA off/on may differ. Review action/contact separately
from photo realism. Improvement on one held-out synthetic-control anchor is
still not evidence of universal cooking-edit generalization.

Frozen config: `configs/adapter_action_pseudo_v1.json`.
Machine-readable dataset summary: `results/day9_action_pseudo_dataset_v3/summary.json`.
