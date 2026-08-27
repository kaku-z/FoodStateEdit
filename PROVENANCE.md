# Sprint-start provenance

Snapshot date: 2026-08-26 (Asia/Tokyo)

## Historical workspace

- Local experiment root:
  `C:\Users\kaku\Downloads\TF_UFI_EXPERIMENT_20260806\noodle_image_benchmark_v1`
- Remote experiment root: `/host/space0/guo-z/tf-ufi`
- The historical local experiment root was not a Git repository at snapshot time.
- This `paper_release` directory is intentionally a separate lightweight repository.

## GeoEdit source

- Local path: `third_party/GeoEdit`
- Remote path: `/host/space0/guo-z/GeoEdit`
- Upstream/base commit: `fa7583c60913d05754b6aa430b579d09dbc11ac8`
- Snapshot was dirty in four known files. The exact Day 3 baseline copies are
  retained under `results/day3_geoedit_overrides_v1/GeoEdit/`; later staged
  development continues under `vendor_overrides/GeoEdit/`.

| File | SHA-256 at freeze |
| --- | --- |
| `geoedit/inference.py` | `f78488216536e70744e839c0294319f334d541adc36f9e76136c14e7ffd328fa` |
| `diffsynth/pipelines/wan_video.py` | `28a1b5e3939983139e2dcd4a7f19d085fcec0017434793bac163767b7ae29c00` |
| `diffsynth/utils/data/__init__.py` | `fe056b4a675a345cf02d6c76d327e8434103ccf2a4066ff208ce41368771e360` |
| `tests/test_masks.py` | `57176ea2595f236846e1d38a79f58e3d586a99e67ce7ef0df4747af768a982da` |

Known intent of the modifications:

- preserve the pre-existing video-save and first-frame-result fixes;
- expose controlled GeoEdit inference locally;
- add staged rigid/material proxy masks and separate injection endpoints;
- add argument validation and tests.

## FoodStateEdit four-layer runtime v2

The original remote GeoEdit worktree remains untouched. A detached, untracked
runtime was composed from the upstream commit plus the four files below at
`/host/space0/guo-z/tf-ufi/runtime/foodstateedit_geoedit_v2_20260827`.

| File | SHA-256 |
| --- | --- |
| `geoedit/inference.py` | `d093ae7e2245884ccb7fc664b78a919ea5443088e77aa1a53c6a033f1e8c8bf1` |
| `diffsynth/pipelines/wan_video.py` | `50bb12685ef9ccb6b89d6600c27ad0cd2f12b5c67c5301fc51ea3f0eff02bfb5` |
| `diffsynth/utils/data/__init__.py` | `fe056b4a675a345cf02d6c76d327e8434103ccf2a4066ff208ce41368771e360` |
| `tests/test_masks.py` | `fc6400e66c94a22e5048fbf7fc37058e6bac099c8c2024086b7ea408e640f421` |

Version v2 adds explicit contact and hole masks/endpoints to the existing
rigid/material interface. Active semantic masks are composed by union over
half-open windows `[tweak_index, endpoint)`. Ten offline unit tests passed in
the existing GeoEdit environment before stochastic inference.

## Model inventory

Models are reused in place. Downloads are disabled in formal launchers.

| Remote file | Size in bytes | Recorded mtime | SHA-256 completed 2026-08-27 |
| --- | ---: | --- | --- |
| `/host/space0/guo-z/models/PAI/Wan2.2-VACE-Fun-A14B/high_noise_model/diffusion_pytorch_model.safetensors` | 34,675,325,000 | 2026-08-19 22:30:11 +0900 | `66c61b736c5674deeeef17861e494d3652cc9b1463a9656bf18c2c72d2c5f007` |
| `/host/space0/guo-z/models/PAI/Wan2.2-VACE-Fun-A14B/low_noise_model/diffusion_pytorch_model.safetensors` | 34,675,325,000 | 2026-08-20 14:44:17 +0900 | `0bf791adfb8330d451d2f5c03577b2a8fb780453f8ef05e23a8fa91f27a2134d` |
| `/host/space0/guo-z/models/PAI/Wan2.2-VACE-Fun-A14B/models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | 2026-08-20 22:20:48 +0900 | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` |
| `/host/space0/guo-z/models/PAI/Wan2.2-VACE-Fun-A14B/Wan2.1_VAE.pth` | 507,609,880 | 2026-08-20 22:23:16 +0900 | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |

The one-time full-file audit is retained remotely at
`outputs/paper_sprint_day3_20260827_model_hash_v1/wan_model_sha256.txt`.
Its tracked text copy is `results/day3_model_hash_v1/wan_model_sha256.txt`
(SHA-256 `4c255c04811a2bc1484db5dae01bdc7352ebd4ad0bd7802b13cebe0bc669d34b`).
Launchers validate that audit and current file sizes instead of rereading
81.2 GiB before every run. Formal manifests record all four audited hashes.

## Key result anchors

| Result | Local path relative to historical root | SHA-256 |
| --- | --- | --- |
| staged spoon scoop, rigid18/material8 | `outputs/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/staged_rigid18_material8/evaluation_v0_3_regions/edited_2d.png` | `94aad130d59de6194068864ec773cca1d18d09fcbdb1576541de370ed0b2f667` |
| controlled liquid fill, no warm | `outputs/foodstateedit_liquid_fill_v0_2_4_garnish_lock_median/clear_broth_low_to_high/geoedit_no_warm/evaluation_v0_2/edited_2d.png` | `7d749ef1ea21c16dd6f42e7c35de22f58e1d1242ef5e943599ff7c3bcd5a3ed4` |
| ramen/chopsticks v3 last frame | `outputs/geoedit_3d_v3_telea_exact/uec256_noodle_20_1897_insert/geoedit/return_2d_last80/edited_2d.png` | `c0104c32d04b81a0eeed1b618c1ad808043be40ad303155113e2d3599c7ce99b` |

These anchors document sprint-start evidence; they are not the frozen 60-case
benchmark and must not be silently promoted into test-set statistics.
