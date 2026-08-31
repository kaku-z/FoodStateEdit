# Day 10 seen-family action-LoRA diagnostic

Date: 2026-08-31

## Outcome

Both same-seed LoRA-off/on comparisons completed on gp39. Each resident worker
loaded the Wan2.2 pipeline once, decoded 21 frames per condition, injected all
80 trained high-noise VACE branches for LoRA-on, left `vace2` untouched, and
changed no pixel outside the frozen edit support.

The result closes the immediate diagnostic: the current adapter is underfit;
the failure cannot be explained only as chopsticks/spoon-to-fork transfer. The
LoRA produces numerical changes but no visible semantic gain even on its two
training families.

## Seen-sample comparison

| sample | target-support MAE off | target-support MAE on | relative change | semantic LoRA gain |
| --- | ---: | ---: | ---: | --- |
| udon / chopsticks | 21.8543 | 21.8879 | 0.15% worse | no |
| clear broth / spoon | 53.1147 | 52.9740 | 0.27% better | no |

For udon, both conditions show one blurred gray utensil-like head rather than
two separate chopsticks. No noodle is pinched, lifted, or kept connected to the
bowl. Strict action, contact, and photo-realism gates fail in all six reviewed
frames (0, 5, 10, 15, 18, and 20).

For broth, both conditions produce a coherent metal spoon containing broth.
This is a useful one-sample static-state success of the base VACE renderer plus
control, because it already occurs with LoRA disabled. LoRA-on does not change
the visible action/contact semantics, so the small MAE reduction is not an
adapter success and cannot support a LoRA claim.

## Why the current training failed

The checkpoint is valid and its 80 branches are active at inference, so this is
not another loader failure. The pilot has only two synthetic pseudo-targets and
16 optimizer steps. More importantly, every training row repeats one final
image for all 21 frames. It therefore teaches a static local appearance target,
not the intended approach--contact--lift transition. The outputs confirm that
failure mode: all six reviewed frames retain essentially the same state.

## Next gate

Do not run another blind fork evaluation yet. First run a deliberately bounded
overfit sanity experiment using phase-varying targets for approach, contact,
payload acquisition, lift, and final hold. Save intermediate checkpoints and
measure both support-restricted target error and explicit semantic events. The
new gate is passed only if LoRA-on visibly improves its training cases over the
same-seed LoRA-off controls. Only then should the dataset expand across liquid,
strand, granular, and mixed foods with spoon, chopsticks, and fork families.

This result is a diagnostic, not evidence of generalization, real-data
performance, or paper-level photo realism. Machine-readable evidence:
`results/day10_seen_action_lora_result_v1.json`.
