# Day 18: high-lift swept-support repair

## Question and scope

Can a complete swept edit support retain the raised chopsticks and noodle in
the Day 17 high-lift pilot? This is a repair/control-following diagnostic on
one seen synthetic udon image, not a new learned method or evidence of
generalization. There is no new training, blind fork or model download.

The high-lift scaffold remains fixed: contact-to-final pinch displacement is
150.314 px, versus 64.181 px for the earlier low-lift scaffold. These are
**specified control displacements**, not measured generated motion.

## Defect isolated without new inference

The Day 17 projected final hold hid chopsticks which were present in its raw
VACE output. Reusing the exact raw video while changing only final alpha
restores visible raised sticks, although their edges remain malformed.
The previous conclusion that the projected action failed is retained; an
attribution of disappearance solely to VACE is not supported.

| Coverage diagnostic | Old support | Repaired support |
|---|---:|---:|
| Nonzero alpha area / image | 16.1877% | 17.9857% |
| Final rendered stick-body pixels outside support | 1,409 / 4,421 | 0 / 4,421 |
| Mean alpha on final stick bodies | 47.27 / 255 | 255 / 255 |
| Full-opacity coverage of swept rendered changes | incomplete | 35,309 / 35,309 |

The repair is derived from the complete rendered geometry, not from selecting
successful generated pixels. It uses the swept union, an 8 px full-opacity
margin and an 8 px feather ramp, then takes the maximum with old alpha.
The fixed source is unchanged outside the expanded support before encoding.
This mask contract is not a physical-occlusion proof.

## Frozen fresh inference

- Same prompts, negative prompt, seed 1, reference, geometry and model/runtime
  bytes as Day 17; 21 frames, 20 steps, VACE scale 1, LoRA off and TTM off.
- **Three things change together:** raster support in the control video,
  the VACE edit mask, and final projection alpha. The fresh run cannot isolate
  each stage's contribution; only the same-raw CPU ablation isolates final alpha.
- New package is inference-only. The historical low-lift target is not copied
  as a valid high-lift target or used for training.
- Config: `configs/day18_high_lift_swept_support_gp40_v1.json`, SHA-256
  `b497aaeac4d595c2938a632f6ac8dfe9bcc5c7a211de4dd84825b57f154136db`.
- Dataset manifest SHA-256:
  `920fe644ccaf366d1ff128a66e6ce3f044ce17831bc225105a32115f94917e52`.
- Preflight SHA-256:
  `daa42990b3b245ec0b05ae04dee9978ea3c81fd22f5d6c5717e5f9eb38bdcb3f`.

The gp38 output-path template was not run. Its gp40 variant changes only the
output path and was frozen before launch. All 12 uploaded package files and
both runtime/config files matched local SHA-256.

## Resource safety

Read-only gp38--gp42 audit excluded occupied gp38/gp39 A6000s, gp41 A40s and
gp42 Blackwell GPUs. On gp40, only physical GPU 5 qualified. Fresh runner
preflight at 2026-09-08T06:54:59Z recorded RTX A6000, 48,539 MiB free, 0%
utilization, zero compute processes, and 247,313 MiB available host memory.
This is a point-in-time gate, not an exclusive reservation. Other users'
processes were not changed. The first short-timeout SSH probe failed; a longer
connection timeout succeeded before any upload or inference launch.

## Execution and visual review

The single inference completed at 2026-09-08T07:07:40Z in 761.072 seconds,
with one pipeline load and 21 decoded frames. All eight remote output files
(six manifested artifacts plus manifest and completion marker) and the
preflight matched local SHA-256. All six preserved Day 17 artifacts were also
reverified against their recorded hashes.

Agent technical inspection of native and projected frames 0, 3, 6, 10, 15 and
20, plus final-hold contact crops, found a visible repair: the raised utensil
remains at frames 15 and 20, and the ragged/cut-out handles of the same-raw
Day 17 reprojection become smooth, continuous handles after fresh inference.
An elongated noodle-like body is visible. This is an improvement in this
specific artifact, not a statistically established effect.

The strict action and photo gates are **not passed**. Two separate stick
instances are not unambiguous at the pinch, the precise gripping relationship
is weak, and the noodle's attachment to the existing bowl noodles is unclear.
The strand remains overly straight and soft, with weak contact shading.
Native and final outputs both retain these limitations. This is agent image
inspection, not independent human evaluation or a continuous-video quality
guarantee.

The native output changes pixels outside support (MAE to source 6.6960).
Compositing restores exact zeros outside support **before encoding**. The
decoded lossy projected MP4 has outside-support MAE 3.8977 and maximum
difference 94; therefore the MP4 itself must not be called pixel-exact.
Inside-support change MAE is 15.4156 native and 14.4079 after recompositing.
These are preservation/change diagnostics, not semantic quality scores.

Decision: retain the support repair, but do not claim full manipulation,
photorealism, learned effectiveness or algorithmic novelty. No blind fork or
balanced expansion is unlocked. The next design should isolate contact
visibility and two-stick separation with frozen controls, not search seeds.

Machine-readable result: `results/day18_high_lift_support_repair_20260908.json`.
Local comparisons and verification:
`artifacts/day18_support_result_verification_20260908_v1`.

## Evidence and reproduction

- Input package: `artifacts/day18_high_lift_swept_support_v1`.
- Preserved Day 17 result: `artifacts/day17_high_lift_vace_gp38_20260907_v1`.
- Remote fresh output:
  `/host/space0/guo-z/tf-ufi/outputs/day18_swept_support_vace_gp40_20260908_v1`.
- Remote package:
  `/host/space0/guo-z/tf-ufi/outputs/day18_high_lift_swept_support_v1`.
- Offline runtime:
  `/host/space0/guo-z/tf-ufi/runtime/day18_swept_support_v1`.
- Builder: `scripts/prepare_high_lift_support_pilot.py` (one-shot, refuses
  existing output root and named config).
- Pull verification and diagnostic comparisons:
  `scripts/verify_high_lift_support_result.py` (new output root required).

The bundled Python environment initially lacked the repository's optional
`jsonschema` test dependency. It was installed into a local ignored validation
cache without changing the remote inference environment. The initial suite
reported two import errors; these must not be confused with algorithm failure.
After adding the missing dependency and regression checks, the complete suite
passed **171 tests, zero failures/errors/skips**. Validation command:

```powershell
& 'C:\Users\kaku\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "import sys,unittest; sys.path.insert(0,'cache/day18_validation_deps'); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests')); sys.exit(not r.wasSuccessful())"
```

No independent human review, measured generated lift distance, multiple-seed
robustness, unseen-image performance or physical correctness is established.
Automatic execution remains paused.
