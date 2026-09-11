# Day 11 post-outage gp40 checkpoint-sweep reproduction

Date reviewed: 2026-09-12

## Outcome

The frozen seen-synthetic sweep was restored after the server outage and ran
serially on idle gp40 physical GPU 4 (NVIDIA RTX A6000). Both samples completed
all five predeclared conditions with one pipeline load per sample, 21 decoded
frames per condition, and exact preservation outside the declared support.
All 45 downloaded artifact and supporting-evidence files matched their remote
SHA-256 values.

The positive gate remains closed: `0/2` samples show a clear phase-action or
contact improvement over LoRA-off, and `0/2` show a clear photo-realism
improvement. This reproduces the earlier decision, although individual output
bytes are not claimed to be identical.

## Fixed checkpoint comparison

| Sample | LoRA-off | Step 16 | Step 32 | Step 48 | Step 64 | Numeric best | Clear semantic gain |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Udon / chopsticks | 13.6511 | 13.4942 | 12.6833 | 12.1341 | 11.9998 | step 64 | no |
| Clear broth / spoon | 21.2355 | 20.9574 | 21.3260 | 25.7896 | 23.8928 | step 16 | no |

Values are mean RGB errors inside the frozen target support. They measure
closeness to the seen synthetic target, not physical success or photographic
quality.

## Manual review

- Udon: the conditions are nearly indistinguishable. The lower numerical error
  does not create a clearer pinch, payload acquisition, bowl-connected lift, or
  stable high-lift hold. The contact remains soft/composited.
- Broth: step 16 resembles LoRA-off. Two scallion rings remain in the bowl while
  two also appear in the spoon, so the sequence depicts duplication rather than
  conserved transfer. Steps 48 and 64 add a pale rim and weaken spoon realism.

## Integrity and claim boundary

- successful sample directories: 2; failed sample directories: 0;
- conditions per sample: `lora_off`, `step_16`, `step_32`, `step_48`, `step_64`;
- seed 1, 21 frames, 20 inference steps, VACE scale 1, TTM disabled;
- pipeline loads: one per sample;
- maximum outside-support pixel difference: 0;
- downloaded files rehashed: 45; mismatches: 0;
- run-manifest hashes: udon
  `e1eeb79ae5e32eeb0ff09571ba010489b9f08a72b57f709e5bc718bc3d746ec7`,
  broth
  `99352f6c7b1bb770e19aca0fbaa495a87f78b82831ed572217c41bae998fa60f`.

This is a seen synthetic overfit-capacity diagnostic only. It does not establish
held-out generalization, real-data performance, physical correctness, or
paper-level photo realism. Balanced expansion and blind-fork evaluation remain
prohibited. Machine-readable evidence is in [result.json](result.json).
