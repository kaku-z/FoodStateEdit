# E3 Canny mask-contract correction probe

## Result

The E3 probe completed on `gp40` physical GPU 0 without retries or technical failures. Both cases used seed 1, 21 frames, 20 inference steps, VACE scale 1.0, LoRA off and TTM off. Each case loaded the pipeline once.

The identified interface error is corrected. E2 passed the local final-compositing alpha to the VACE condition encoder. For the sparse black-background Canny video, that routed the black background into VACE's inactive/preserved branch outside the local support. E3 instead passes `vace_video_mask=None`, which the frozen runtime converts to an all-one full-generation mask. The frozen local alpha is used only after generation.

| Case | E2 black fraction | E3 black fraction | E2 raw MAE to reference outside support | E3 raw MAE to reference outside support | Final outside-support max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| cake | 0.618313 | 0.000000 | 145.0523 | 17.8524 | 0 |
| noodle | 0.261903 | 0.0000029 | 96.4524 | 10.4704 | 0 |

The large inversion in raw-image similarity is consistent with the diagnosis: E2 raw outputs were close to the black Canny control outside the local support, whereas E3 raw outputs are much closer to the reference photograph.

## Internal visual screen

- **Cake:** the fork approaches, contacts and lifts a separated cake piece. The black failure is gone. A dark mask-shaped support/shadow region remains visible, so strict photo success is not established.
- **Noodle:** the chopsticks approach, contact and lift a noodle strand. The black failure is gone. The source bowl remains visually full and source depletion is unclear, so source conservation and non-duplication are not established.

This is an unblinded internal screen, not a formal human evaluation.

## Integrity

- Config SHA-256: `15186033393ba30e24917bd001cf66ab95ea9c56ec7ed35b1cb549008726a8a7`
- Runner SHA-256: `f5058b565ddc86e453bb66c6418687ee844bcd0f60ca43a4a2ce66ff65210a2e`
- Frozen controls manifest SHA-256: `1a2e23251b50580e3ea4735d6a881d049280255e41053436be21133e4158eec6`
- Remote/local evidence: 102 files, all SHA-256 exact matches
- Local evidence: `artifacts/e3_canny_mask_contract_probe_gp40_20260917T043645Z`
- Remote output: `/host/space0/guo-z/tf-ufi/outputs/e3_canny_mask_contract_probe_v1_20260917T043645Z`

## Decision

The corrected VACE mask contract is accepted for this bounded seed-1 diagnostic. The next experiment should be a separately frozen multi-seed confirmation. It should also test a per-frame or softly feathered final projection support and an explicit source-depletion constraint. No claim of 3D superiority, generalization, physical correctness, source conservation or publication-level photo realism is supported by E3 alone.
