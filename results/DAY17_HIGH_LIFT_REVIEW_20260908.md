# Day 17 high-lift pilot: technical completion, projected action failure

## Day 18 causal correction (2026-09-08)

The disappearance described below applies to the **old projected output**, not
necessarily to native VACE generation. A CPU-only ablation reusing the identical
Day 17 raw video shows raised sticks reappear with expanded final alpha. The
old support omitted 1,409 of 4,421 final-hold stick-body pixels (31.87%); mean
old alpha on the stick body was only 47.27/255. Bounding-box containment did
not establish actual support coverage or opacity. This is a concrete masking
and compositing defect, not evidence that VACE alone deleted the utensil.

The original projected artifact and original negative visible-action review
remain valid and are preserved. The reprojected sticks retain malformed edges;
neither this correction nor the original run establishes complete action or
photorealism. Diagnostic evidence:
`artifacts/day18_high_lift_swept_support_v1/review/expanded_projection.png`.

## Original artifact review

Reviewed 2026-09-08. Run finished 2026-09-07T12:14:30Z on gp38.

The user requested a larger noodle lift. The new scaffold moves the pinch
anchor from normalized (0.755, 0.405) to (0.660, 0.170). Contact-to-final
screen displacement is 150.314 px versus 64.181 px for the original scaffold.
The original dataset and outputs were preserved.

One LoRA-off VACE inference completed with seed 1, 21 frames, 20 steps,
VACE scale 1 and TTM disabled. Pipeline load count was 1. The reported
outside-support maximum difference is zero for projected frames before video
encoding; this does not establish native generator preservation or lossless
MP4 preservation. The prompt and negative prompt also changed to request a
larger lift, so comparison with Day 13 is not an amplitude-only ablation.

Agent inspection of the six-frame contact sheet shows a taller noodle-like
segment at final hold, but the chopsticks disappear by the final two reviewed
frames. Visible pinch contact and two-stick persistence therefore fail.
The desired complete high-lift action is not established. The lifted segment
also remains soft and artificial-looking; no photorealism success is claimed.
This technical inspection is not independent human review.

All six artifacts listed by the run manifest were pulled and verified in size
and SHA-256 with zero mismatches. Local evidence:
`artifacts/day17_high_lift_vace_gp38_20260907_v1`.
Remote evidence:
`/host/space0/guo-z/tf-ufi/outputs/day17_high_lift_vace_gp38_20260907_v1`.

Before another inference, inspect the full swept edit support: the scaffold
review itself shows clipping along the raised chopstick handles. A support
bounding box alone is insufficient to prove that the whole trajectory fits.
The copied low-lift target video is historical source data, not a valid
high-lift target or training label. No high-lift training was performed.
