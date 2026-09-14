# Day 26 ChordEdit direct-input baseline result

## Material Passport

- Experiment ID: `day26_chordedit_direct_baseline_v1`
- Artifact type: external diffusion editor development baseline
- Verification status: `EXECUTED_HASH_VERIFIED_INTERNAL_NON_BLIND_REVIEW`
- Execution host: `gp40.cs.uec.ac.jp`
- Physical GPUs: ramen `2`, soup `3`, rice `4`, cake `5`
- GPU type: `NVIDIA RTX A6000`
- Execution date: `2026-09-12`
- Scientific scope: four selected development inputs and three fixed seeds; cake is synthetic supplementary evidence; not a held-out benchmark

## Frozen method

ChordEdit with the locally available SD-Turbo package edited the original image
directly from a source-to-target prompt difference. It did not receive the
planar proxy, relative-3-D proxy, VACE output, or any method-specific best-frame
selection. Every seed produced one candidate. The final composite restored the
source image exactly outside the predeclared edit support.

- Config SHA-256: `a2acaeb05fb2823386e17d08d630f88f36f9db85b92e66602e1b08a8bcfdde0a`
- Runner SHA-256: `7d1a977ab19384efa585c8a0024f7f6e169f2757e160a3d6b0003bb5246f87dc`
- Frozen seeds: `1`, `2`, `3`
- Expected and completed outputs: `12/12`
- Pipeline loads: exactly one per case
- Model downloads: none

## Technical verification

All four fresh preflights passed the RTX A6000 resource gate. The four workers
completed without replacement seeds or retries. Seventy-two remote files were
rehashed after download; every local SHA-256 matched. For all 12 candidates,
the maximum pixel difference outside the declared edit support was exactly
zero.

| Case | GPU | Seeds complete | Pipeline loads | Outside-support max difference |
| --- | ---: | ---: | ---: | ---: |
| Ramen | 2 | 3/3 | 1 | 0 |
| Soup | 3 | 3/3 | 1 | 0 |
| Fried rice | 4 | 3/3 | 1 | 0 |
| Cake (synthetic) | 5 | 3/3 | 1 | 0 |

## Preliminary visual review

This review is an internal, non-blind diagnostic and is not the formal human
evaluation. The frozen ballot must still be completed by independent reviewers.

| Endpoint | Pass | Total | Interpretation |
| --- | ---: | ---: | --- |
| Action Success | 0 | 12 | No output established the requested correct utensil, supported payload, contact/lift and source correspondence together. |
| Photo Success | 1 | 12 | Only rice seed 1 remained photographically plausible, largely because the requested action was absent. |
| Preservation Success | 12 | 12 | Exact outside-support compositing passed for every output. |
| Strict End-to-End Success | 0 | 12 | Every candidate failed Action Success. |

The main failures were missing or truncated utensils, incorrect utensil type,
duplicated fork/chopstick fragments, handleless spoon bowls, unsupported food,
and visible support-boundary or local diffusion artifacts. Ramen did not show a
valid two-stick pinch with a lifted connected noodle. Soup did not show one
complete spoon holding lifted soup. Fried rice did not show one complete serving
spatula supporting a coherent scoop. Cake did not show one fork lifting the
pre-cut bite without duplication.

## Result for publication-gap item 3

The experiment adds one reproducible, same-input external diffusion editor
baseline. It is useful negative evidence: direct final-state editing did not
solve the structured food-manipulation task under the frozen supports. However,
it does not fully close the strong-model comparison requirement. Qwen-Image-Edit,
FLUX Kontext and ChronoEdit remain unavailable as complete frozen local
runtimes, and no result is fabricated for them.

The held-out test remains locked. Before item 4, the Day 25 same-condition
ablation must finish and the unique material-adaptive/rollback rule must be
frozen without looking at held-out outputs.

## Evidence paths

- Machine-readable result: `results/day26_chordedit_direct_baseline_result_v1.json`
- Local artifacts: `artifacts/day26_chordedit_direct_baseline_v1/`
- Review grid: `artifacts/day26_chordedit_direct_baseline_v1/review_grid.png`
- Remote outputs: `/host/space0/guo-z/tf-ufi/outputs/day26_chordedit_direct_baseline_v1_{ramen,soup,rice,cake}`
- Remote preflights: `/host/space0/guo-z/tf-ufi/outputs/day26_chordedit_direct_baseline_v1_{ramen,soup,rice,cake}_preflight.json`

## Claim boundary

This four-case, three-seed development result supports reproducible failure
analysis for one available external editor. Cake is synthetic supplementary
evidence. It does not establish superiority, generalization, physical
correctness, fully automatic editing, or paper-level photo realism.
