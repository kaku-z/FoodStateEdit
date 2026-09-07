# Day 15 frozen SAM3 strand/utensil observer pilot

Date: 2026-09-07. Status: complete negative prerequisite-gate result.

## Bottom line

The existing frozen SAM3 checkpoint ran successfully on the Day 13 seen udon
source, synthetic target, five existing VACE final-hold outputs and three
counterfactual controls. It is **not yet a valid observer for the proposed
contact/visibility guidance**. Only 3 of 9 frozen automated gates passed.

SAM3 found a strand and a wooden-chopstick candidate in the relative-3D
topology-weighted output, but that isolated fact is insufficient. The frozen
"lifted strand" prompts assigned almost the same high score to the unedited
source and segmented most of the entire noodle bowl. The synthetic target
received no chopstick detection. Removing the frozen strand or utensil region
did not lower the corresponding score, and a solid noodle-colored block was
only 0.0157 below the target strand score rather than the required 0.05.

No VACE inference was launched. This result does not alter the Day 13 negative
effectiveness conclusion.

## Frozen protocol

- Direct SAM3 image model through the preserved `Evol-SAM3` source tree;
- source commit `29b46c21c31f892224360c0d6fd1f8038412bf30`;
- existing 3.45 GB `sam3.pt`, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`;
- no Qwen model, evolutionary agent, model download or weight update;
- four prompts frozen before inference: two lifted-strand variants and two
  wooden-chopstick variants;
- confidence threshold 0.25, positive threshold 0.35 and contrast margin 0.05;
- one seen synthetic sample only; masks are candidate observations, not ground truth.

The final config hash is
`1303826625989007684996384407edc6d02b77b06734b4f93ded8e9479786181`.
The completed runner hash is
`272ecc5bc46f056de2701a4d1f215bce1694188ab073bb62dd6b2184c9dca7db`.

## Automated result

| Frozen gate | Pass |
|---|:---:|
| Target has strand candidate | yes |
| Target has utensil candidate | **no** |
| Target strand score exceeds source by 0.05 | **no** |
| Target utensil score exceeds source by 0.05 | **no** |
| Strand erasure lowers strand score by 0.05 | **no** |
| Utensil erasure lowers utensil score by 0.05 | **no** |
| Solid block is worse than target strand by 0.05 | **no** |
| Weighted output has strand candidate | yes |
| Weighted output has utensil candidate | yes |

Selected group scores are the maximum returned instance confidence across the
two frozen prompt variants:

| Image | Strand | Utensil |
|---|---:|---:|
| Unedited source | 0.9239 | 0.0000 |
| Synthetic target | 0.9177 | 0.0000 |
| Planar LoRA off | 0.9180 | 0.8916 |
| Planar uniform step 32 | 0.9190 | 0.9009 |
| Relative-3D LoRA off | 0.9146 | 0.5019 |
| Relative-3D uniform step 32 | 0.9161 | 0.7254 |
| Relative-3D topology weighted step 32 | 0.9150 | 0.3805 |
| Strand restored to source | 0.9210 | 0.0000 |
| Utensil restored to source | 0.9189 | 0.0000 |
| Solid noodle-colored block | 0.9020 | 0.0000 |

These values must not be used to rank VACE conditions because the observer
failed its prerequisite gates.

## Technical visual audit

The overlay sheets confirm the automated failure. Both strand prompts color
nearly the full bowl of noodles, scallions included, rather than isolating the
lifted strand. For generated images the chopstick prompts return one mask
instance or none; they do not establish that two sticks were separately
recognized. The topology-weighted output receives a 0.3805 wooden-chopsticks
score, while the planar and relative-3D-uniform outputs receive larger scores,
but this is diagnostic only because semantic validity is not established.

This inspection is an agent technical audit, not either of the two independent
human reviews required for a semantic claim.

## Resource and failure evidence

The initial gp40 attempt was blocked safely when another user's process occupied
GPU 0 between the read-only audit and preflight. No process was terminated or
preempted. The first gp38 attempt exposed a missing Triton dependency in the
old `Evol-SAM3` environment. A second gp38 attempt completed inference but a
runner boolean typo prevented final JSON creation. All failed directories and
preflight reports were preserved; no path was overwritten.

The authoritative run used gp38 physical GPU 0 after a fresh 0%-utilization,
48,539 MiB-free, zero-compute-process preflight. It used the already installed
`sam3_env` with PyTorch 2.4.0+cu121. The model released the GPU normally after
completion.

The complete remote output is
`/host/space0/guo-z/tf-ufi/outputs/day15_sam3_observer_pilot_gp38_retry_v3_20260907T1050Z`.
Its result JSON SHA-256 is
`e8f0ddbb2869048342fe37df68994f9e28fb7a9acbfd6bc44400b58a9dce4f37`.
After transfer, all 95 files covered by the remote manifest matched in size and
SHA-256.

Local overlays, masks, fixtures, raw JSON and all four preflight reports are in
`artifacts/day15_sam3_observer_pilot_20260907`.

## Decision and next gate

Do not connect this observer to the VACE scheduler. The next observer must use
the relative-3D projection to restrict the candidate region or provide
box/point prompts, and it must be evaluated against independent noodle
centerline, two-stick instance and contact labels. The no-edit, object-erasure,
solid-block, broken-strand, empty-grasp, fused-stick and true-occlusion controls
must remain. Passing that observer gate would permit a separately frozen latent
guidance pilot; it would still not prove generalization or photo realism.
