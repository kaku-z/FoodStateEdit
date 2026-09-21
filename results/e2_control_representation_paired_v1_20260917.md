# E2 paired control-representation diagnostic (2026-09-17)

## Outcome

The experiment completed without a technical failure, but the proposed full-frame Canny arm failed the visual gate. Under the current VACE local-edit interface, all six Canny outputs retained a large black artifact inside the editable support. The RGB-proxy arm generated visible approach/contact/lift motion, whereas the Canny arm did not preserve photo realism.

This is a negative diagnostic result. It must not be reported as evidence that the proposed 3D method is effective.

## Frozen design

- Cases: cake and noodle.
- Seeds: 1, 2, 3.
- Arms: `appearance_laden_rgb` and `fullframe_canny`.
- Both arms use one identical reference slot, the same prompt, negative prompt, mask, trajectory, frame count, step count, VACE scale, and seeded noise shape.
- 21 frames, 20 inference steps, VACE scale 1.0, LoRA off, TTM off.
- One pipeline load per case; no retry or seed replacement.

## Execution and integrity

- Host: gp40; physical A6000 GPU 0 for cake and GPU 1 for noodle.
- Both preflights passed with 48,539 MiB free VRAM, 0% utilization, zero compute processes, and more than 245 GiB available host memory.
- 12/12 conditions completed; 21/21 raw PNG and 21/21 projected PNG frames exist for every condition.
- `pipeline_load_count == 1` for each case.
- No `failure.txt` or `resource_stop.txt` was created.
- 570 remote files and 570 local files have identical relative-path SHA-256 sets.
- The two preflight reports also match remote SHA-256 values.
- The lossless compositor preserves every protected pixel exactly (`outside_support_max_pixel_difference == 0` in all 12 conditions).

Local evidence: `artifacts/e2_control_representation_results_gp40_20260917T031505Z`.

## Internal visual screen

This screen was performed with arm identities visible; it is not an independent blind evaluation.

| Arm | Cake (3 seeds) | Noodle (3 seeds) | Conservative conclusion |
| --- | --- | --- | --- |
| Appearance-laden RGB | Visible fork approach/contact/lift. One sample has a plausible gap; other seeds show residual or detached fragments and seed-sensitive utensil geometry. | Chopsticks and a lifted strand are visible, but strict source depletion/material transfer is not demonstrated. | Useful motion prior, but appearance leakage remains a confound. |
| Full-frame Canny | Severe black editable-region artifact in 3/3 seeds. | Severe black editable-region artifact in 3/3 seeds. | Photo gate fails in 6/6; do not advance this representation. |

The Canny arm's raw outside-support MAE is much larger than the RGB arm's, but this is a diagnostic of the unprojected generator output and is not itself a photo-quality or action-success metric. The final projection restores protected pixels exactly while leaving the black artifact inside the allowed support.

## Failure localization

The result does **not** establish that Canny conditioning is universally unsuitable. It falsifies a narrower implementation hypothesis: sparse black-background edge video cannot directly replace the dense RGB proxy in the frozen local-mask VACE call used here. The consistent black support suggests a control-representation/interface mismatch before 3D trajectory quality can be evaluated.

## Decision

- Do not use the Canny outputs in the paper or presentation as successful results.
- Do not claim 3D superiority from the RGB outputs because the proxy contains target appearance.
- The next controlled experiment should retain the same trajectory and mask but use a reference-anchored **dense** control representation that removes target texture without removing scene context. Only after that arm passes a photo-realism gate should control scheduling or 3D-vs-2D comparisons resume.

## Claim boundary

Two development cases and an unblinded internal screen do not establish generalization, source conservation, physical correctness, publication-level realism, or superiority over VACE. Formal success-rate claims still require independent blind scoring.
