# Day 30 held-out input readiness audit

## Outcome

The frozen 40-image real-data test split is available on `gp40` and every
canonical input matches the SHA-256 recorded before method development. The
split remains balanced at 10 images each for liquid, granular, strand and
strand-contact. This audit read hashes only; it did not open images, create
controls or generate any held-out method output.

Evidence: `results/day30_heldout_input_readiness_gp40_v1.json` (SHA-256
`ce27b6cdfd1d599febabe0073093d25d0a5d52003e99cd1ef78c39ba63bf1a60`).

| Readiness gate | Result |
|---|---:|
| Frozen manifest hash | pass |
| 40 unique real test images | pass |
| 10 images per family | pass |
| Canonical files present on gp40 | 40/40 |
| Canonical SHA-256 match | 40/40 |
| Versioned annotations/controls complete | 0/40 (`pending`) |
| Day 25 development ablation complete and reviewed | pending |
| Material-level Ours rule frozen before test output | pending |
| Held-out generation allowed | **no** |

## Why generation is still locked

The test set cannot be used to choose the VACE strength, rollback behavior,
geometry, prompt or frame. Rice and cake Day 25 workers have only passed
preflight and entered pipeline loading; completion has not been verified. The
40 test cases also still need versioned annotations/controls. Generating now
would turn the held-out set into development data.

## Exact next gate

1. Verify and review all four Day 25 development cases.
2. Freeze one material-family strength rule and rollback contract without
   reading held-out outputs.
3. Produce and hash-lock annotations/controls for all 40 frozen cases.
4. Freeze identical seeds (1, 2, 3), prompts, frame selection, endpoints and
   method configurations, then run the complete test without replacement or
   output-dependent retry.

## Claim boundary

This is input-integrity and readiness evidence only. It contains no held-out
model result and supports no effectiveness, superiority, realism, physical
correctness or generalization claim.
