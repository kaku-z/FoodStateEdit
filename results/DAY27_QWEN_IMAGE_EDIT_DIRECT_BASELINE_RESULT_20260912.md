# Day 27 Qwen-Image-Edit direct-input baseline result

## Material Passport

- Experiment ID: `day27_qwen_image_edit_direct_baseline_v1`
- Artifact type: strong external diffusion editor development baseline
- Verification status: `EXECUTED_HASH_VERIFIED_INTERNAL_NON_BLIND_REVIEW`
- Execution host: `gp40.cs.uec.ac.jp`
- Physical GPU: `6` (`NVIDIA RTX A6000`)
- Execution date: `2026-09-12`
- Scientific scope: four selected development inputs and three fixed seeds;
  cake is synthetic supplementary evidence; not a held-out benchmark

## Completed execution

The frozen Qwen-Image-Edit-2511 runner completed all 12 single-candidate
conditions in 2422.839 seconds. The preflight passed all 28 checks with 48,539
MiB free GPU memory, zero utilization, zero compute processes and 251,994 MiB
available host memory. The model was loaded once, no model was downloaded, no
seed was replaced and no failure marker was created.

- Config SHA-256: `d2a8a144bbfa1bf48e370dc7f7c399c92df3c6812da2cf1e31e87b8caa9cab7a`
- Runner SHA-256: `4a3a63d2873cb77c0061747a0f57f0e7e06bfeae67a51861d23ecb8543a539fb`
- Expected and completed outputs: `12/12`
- Pipeline loads: `1`
- Remote output files rehashed locally: `62`, mismatches: `0`
- Additional preflight/log/config/runner files rehashed locally: `4`, mismatches: `0`
- Repository validation: `220` tests passed; `655` JSON files parsed; frozen
  override hashes passed

## Which image is the Qwen baseline

`raw_qwen.png` is the actual Qwen baseline endpoint. These images are highly
photographic and contain recognizable chopsticks, spoon, serving spatula and
fork interactions.

The runner also produced `final.png` by applying the source-space edit support
to the globally edited Qwen image. This is not a valid Qwen endpoint. Qwen
changes the frame dimensions and slightly reframes the scene; the old support
therefore no longer aligns with the generated utensil and payload. The hard
composite visibly truncates chopsticks, spoon handles and fork handles and can
create rectangular seams. It is retained as a transparent post-processing
failure diagnostic, not reported as Qwen quality and not credited as autonomous
preservation.

## Preliminary internal visual review

This is one non-blind author-side diagnostic, not the frozen independent human
evaluation. Applying the existing yes/no rubric conservatively to the raw Qwen
outputs gives:

| Endpoint | Pass | Total | Interpretation |
| --- | ---: | ---: | --- |
| Action Success | 3 | 12 | Soup seeds 1 and 3 and cake seed 1 visibly satisfy the full endpoint; other outputs fail a task constraint or leave source correspondence uncertain. |
| Photo Success | 12 | 12 | All raw Qwen outputs are photographic under the internal rubric. |
| Preservation Success | 0 | 12 | Raw outputs globally re-render/reframe the scene and fail the exact protected-pixel gate. |
| Strict End-to-End Success | 0 | 12 | No output passes action, photo and exact preservation together. |

The important result is a trade-off, not a simple failure. Qwen is far stronger
than the local ChordEdit baseline in static appearance and produces plausible
final-state food manipulation. However, ramen commonly lifts several strands
instead of one conserved noodle; fried-rice source reduction is obscured by
global reframing and shallow depth of field; cake seed 2 lacks a clear matching
source gap; and cake seed 3 generates two forks.

## Machine preservation diagnostic

Each raw output was resized to the corresponding input size with LANCZOS, then
compared outside the frozen source-space support without registration. Every raw
output failed exact preservation. The maximum channel difference ranged from
208 to 250; mean absolute difference ranged from 11.837889 to 69.292492; and
the fraction of outside-support pixels whose maximum channel difference exceeded
10 ranged from 0.309664743 to 0.971492793. These values include resizing and
global reframing and must not be presented as perceptual similarity scores.

The support-locked composites have exact zero outside-support difference by
construction, but the evaluation protocol explicitly forbids treating such a
post-hoc composite as evidence that the generator itself preserved the scene.

## Scientific interpretation

The named Qwen-Image-Edit comparison is now executed on the matched four-case,
three-seed development set. It establishes Qwen as a strong final-image realism
baseline and removes it from the list of missing editors. It also sharpens the
FoodStateEdit contribution target: the relevant advantage cannot be claimed as
raw photographic quality on these examples. It must be tested as controllable
3-D-guided interaction sequence, utensil-payload contact, source-payload
correspondence and bounded scene preservation.

FLUX Kontext, ChronoEdit, held-out generation and independent blinded scoring
remain incomplete. No superiority or generalization claim is supported.

## Evidence paths

- Machine-readable result: `results/day27_qwen_image_edit_direct_baseline_result_v1.json`
- Local artifacts: `artifacts/day27_qwen_image_edit_direct_baseline_v1/`
- Raw Qwen review grid: `artifacts/day27_qwen_image_edit_direct_baseline_v1/review_grid_raw_qwen.png`
- Masked-composite diagnostic grid: `artifacts/day27_qwen_image_edit_direct_baseline_v1/review_grid_masked_composite_diagnostic.png`
- Remote output root: `/host/space0/guo-z/tf-ufi/outputs/day27_qwen_image_edit_direct_baseline_v1`
- Remote preflight: `/host/space0/guo-z/tf-ufi/outputs/day27_qwen_image_edit_direct_baseline_v1_preflight.json`

## Claim boundary

This is a selected four-case, three-seed development result with one internal
non-blind review. Cake is synthetic supplementary evidence. It does not
establish superiority, held-out generalization, physical correctness,
independent human preference or paper-level benchmark performance.
