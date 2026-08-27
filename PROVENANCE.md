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

## Day 4 staged-v1 execution

- Paper-release commit: `eb0c0b01174f3d873db295f1168c5bfc14e45489`.
- Exact launcher directory:
  `/host/space0/guo-z/tf-ufi/outputs/paper_sprint_day4_20260827_launchers_v1`.
- `run_geoedit_anchor.py` SHA-256:
  `4fcc4baca473dd5b7a3b116af7536cdfbd0f0472b0ca2e3c53dfd00f50e4ccf2`.
- `run_geoedit_resident_batch.py` SHA-256:
  `1f36f28696409f48a939c1ece0530ebc77913545d7c1487404397e99446ae1cb`.
- Completed worker roots:
  `/tmp/foodstateedit_day4_staged_v1_gpu0` and
  `/tmp/foodstateedit_day4_staged_v1_gpu1`.
- Both workers processed two cases with one pipeline load. All four cases
  completed at seed `1`, and exact projection changed no protected pixel.

| Anchor | Final edited-image SHA-256 |
| --- | --- |
| `soup_spoon_001` | `bd6df07ac228529b76a053d7671e77336e27ceaff898126a19c6ac6299d04931` |
| `fried_rice_spatula_001` | `67594aa2bdb000769dee82c3a783fb09a845aa4392324198cf85ea6274d45e6a` |
| `ramen_chopsticks_001` | `cf4d5a440598776943ef9b9b5fc80cebff5da69f0695f9be92c3423f0174bd8e` |
| `pasta_fork_001` | `69e5eb631ab0d4d85e4dd9f47b8477fd9573e4fc36bbcf91d81f07ddb9ac3c86` |

The first space0 launch was stopped before image generation after the second
worker log hit the user's disk quota. The partial directory remains at
`outputs/paper_sprint_day4_20260827_staged_v1_gpu0`; its command and manifest
are copied under `results/day4_staged_v1/preflight_quota_abort/`. The completed
workers used new `/tmp` roots and did not overwrite that incident record.

The staged-v1 visual review is a negative pilot (`0/4` provisional action
success). A post-run audit found that overlapping masks shadow shorter semantic
windows, including 100% of soup contact/material pixels. See
`results/DAY4_STAGED_REVIEW.md`; do not promote these pilot results into formal
test statistics.

## Day 4 exclusive-mask follow-up

The ownership policy was frozen at commit
`f1170db8481cd4bcb56067816c52c4ff77a7d204`, the deterministic mask packages
were hash-locked at `91f4ea625c5c594f8cd2b01c29dec8cf04ad0f58`, and stochastic inference began
only afterward. The exact worker SHA-256 was
`b59f4abb30f55d29a9f180b195b3412940eebe4e65e403eac109ab99a5a2fede`.

- Mask root: `/tmp/foodstateedit_day4_exclusive_masks_v2`.
- Worker roots: `/tmp/foodstateedit_day4_staged_exclusive_v1_gpu0` and
  `/tmp/foodstateedit_day4_staged_exclusive_v1_gpu1`.
- Model files, common proxy, seed, frames, steps, endpoints, and exact 2D
  protection were unchanged from staged v1.
- Both workers processed two cases with one pipeline load; all four cases
  completed without a technical failure or seed replacement.

| Anchor | Exclusive edited-image SHA-256 |
| --- | --- |
| `soup_spoon_001` | `0c18c3f18c57568f2e028f61bfa1848887c58af1d0500d752e107dda027b6fce` |
| `fried_rice_spatula_001` | `64fcca50986871dbd0090aec242c075e08365af258a9b1e12fad0337a7e23c70` |
| `ramen_chopsticks_001` | `0cf9ccd69431d1bc6806fc6206f7c26d35754d29540ba0de0f8e604e65914d03` |
| `pasta_fork_001` | `d4d9cc2b64aa2efa98aa07021adf713713d339a413e8de6cd768b0a099938f78` |

All four hashes differ from staged v1, while the exact protected region remains
unchanged. The internal action/photo outcome remains `0/4`; this is a closed
negative pilot, not a formal test result. Full build and comparison evidence is
under `results/day4_exclusive_masks_v2/` and
`results/day4_staged_exclusive_v1/`.

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
