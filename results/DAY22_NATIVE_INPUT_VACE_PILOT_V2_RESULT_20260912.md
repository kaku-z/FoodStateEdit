# Day 22 Native-Input VACE Pilot v2 Result

## Material Passport

- Experiment ID: `day22_native_input_vace_pilot_v2_soup`
- Artifact type: experiment result
- Verification status: `EXECUTED_AND_HASH_VERIFIED`
- Execution host: `gp40.cs.uec.ac.jp`
- Physical GPU: `0` (`NVIDIA RTX A6000`)
- Execution date: `2026-09-12`
- Scientific scope: one same-seed real-image soup development ablation; not evidence of generalization, physical correctness, superiority, or paper-level photo realism

## Frozen condition

- Condition: `native_input_text_mask_only`
- Control: the original soup reference image repeated for all 21 VACE frames
- Mask: the frozen edit alpha repeated for all 21 frames
- Excluded: planar proxy, relative3D proxy, and any utensil-food motion proxy
- Seed: `1`
- Frames: `21`
- Inference steps: `20`
- VACE scale: `1.0`
- LoRA: disabled
- TTM: disabled
- Config SHA-256: `a2ac9b4960c5acb506c8df2b2d0204b8e7a95beae7f80fda2e1a1dae4e18ee29`
- Runner SHA-256: `d34a92bc9fed505374ba4afb0fb383755a4e27f6080a9fc927dff7bc4abb2516`

## Execution result

- Exit code: `0`
- Preflight: passed
- Pipeline load count: `1`
- Wall time: `251.114 s`
- Completed condition wall time: `218.595 s`
- Outside-support maximum pixel difference: `0`
- Remote-to-local SHA-256 verification: exact match for all 11 output files and the preflight report

## Preliminary visual review

This is a non-blind single-reviewer development review and must not be reported as the formal human evaluation.

| Gate | Result | Observation |
|---|---|---|
| Visible spoon | Fail | No spoon is visible in the six frozen review frames or the final hold frame. |
| Visible supported soup payload | Fail | No spoon or lifted soup payload appears. |
| Source surface change | Fail | No visible scooping/removal event is established. |
| No duplicate payload | Not applicable | No payload was generated. |
| Photo realism | Pass for preservation only | The output remains visually close to the input photograph because the requested action is absent. |
| Non-edit preservation | Pass | Outside-support maximum pixel difference is exactly zero. |
| Action Success | Fail | The requested scoop action is absent. |
| Strict End-to-End Success | Fail | Action Success fails. |

## Matched development comparison

The frozen same-seed relative3D comparator visibly introduces a spoon, establishes contact with the soup, carries soup in the spoon, and changes spoon height across the sequence. Its utensil and payload remain visually synthetic, so this observation supports only the need for explicit interaction control in this development case; it does not establish final photo realism or generalization.

## Conclusion

The native photograph, prompt, and edit mask alone did not elicit the requested soup-scooping action in this frozen case. This negative control is therefore useful evidence that explicit utensil-food motion guidance is necessary for the matched development example. The result must remain paired with the relative3D condition and must not be generalized beyond this single case.

## Evidence paths

- Local output: `artifacts/day22_native_input_vace_pilot_v2_soup/`
- Local preflight: `artifacts/day22_native_input_vace_pilot_v2_soup_preflight.json`
- Remote output: `/host/space0/guo-z/tf-ufi/outputs/day22_native_input_vace_pilot_v2_soup`
- Remote preflight: `/host/space0/guo-z/tf-ufi/outputs/day22_native_input_vace_pilot_v2_soup_preflight.json`
- Matched relative3D evidence: `artifacts/day19_multimaterial_gp40_recovered_20260909_v1/soup__relative3d/`
