# Day 3 baseline executability audit

Audit date: 2026-08-26 (Asia/Tokyo)

## Outcome

The formal comparison set is frozen to three primary controls/baselines plus
the proposed method: `input_no_edit`, `vanilla_geoedit`,
`geoedit_unified_action_mask`, and `foodstateedit_staged`. A direct VACE
same-proxy baseline is conditional on a small offline launcher. The Day 3 batch
gate remains open until `common_proxy_v1` exists for all four anchors; engine
readiness must not be confused with completed image inference.

## Reusable assets verified

- Eight NVIDIA RTX A6000 GPUs were visible and idle at audit time.
- The existing Wan2.2-VACE-Fun-A14B high-noise, low-noise, T5, VAE, and tokenizer
  assets are present; no model was downloaded or rerun during the audit.
- The remote GeoEdit override hashes exactly match `PROVENANCE.md`.
- `python -m geoedit.inference --help` imports successfully with downloads
  disabled.
- The six GeoEdit mask/schedule tests pass through their built-in `unittest`
  entry point. The environment does not contain `pytest`; no package was
  installed solely for this audit.
- Historical result files prove that the same GeoEdit/Wan backend has completed
  noodle, spoon, and liquid runs. They are development evidence, not results for
  the newly frozen anchors.

## Inclusion decisions

| Method | Decision | Day 3 evidence | Remaining prerequisite |
| --- | --- | --- | --- |
| Input / no edit | frozen primary control | deterministic batch runner | none |
| Vanilla GeoEdit | frozen primary baseline | CLI import + six unit tests + historical runs | four common RGB/depth proxies |
| GeoEdit + unified action mask | frozen primary baseline | same tested engine; union-mask contract frozen | four common RGB/depth proxies |
| VACE direct/static proxy | conditional secondary | all Wan assets present | common proxies and offline launcher |
| FoodStateEdit staged | frozen proposed method | prior two-layer spoon run and tested staged arguments | four-layer integration and common proxies |

The common proxy is deliberately a prerequisite. Running flat colored shapes or
method-specific emergency proxies would make the comparison uninterpretable.

## Audited exclusions

- **PhysicEdit:** code and a 2,401,645,656-byte fine-tuned checkpoint are local,
  but the Qwen-Image-Edit-2509 base directory is empty, DINOv2 weights are
  absent, and the existing Python environment fails to import the required
  `Qwen2_5_VLModel`. Its checkpoint SHA-256 is
  `649bf4cd3c28561701d9463b3a5c945e292a8be3022db281278a6bbcbeff839f`.
  The official model is a Qwen-Image-Edit-2509 fine-tune, so the small local
  checkpoint alone is insufficient ([official model card](https://huggingface.co/metazlb/PhysicEdit)).
- **FreeFine:** the official ICCV 2025 code is public, but no local repository,
  environment, or model set existed at audit time. It is excluded instead of
  introducing a late multi-model download ([official repository](https://github.com/CIawevy/FreeFine)).
- **ObjectMorpher:** it is a relevant deformable 3DGS reference, but there is no
  local installation and its audited official repository states that detailed
  local generative-composition inference will be released later
  ([official repository](https://github.com/Panaoxuan/ObjectMorpher)).
- **Strong cloud instruction editor:** retained only as a possible qualitative
  oracle. It is not a quantitative baseline without a frozen endpoint/version
  and an explicit compliant policy for uploading restricted UECFOOD256 images.

GeoEdit itself is retained because its lift-manipulate-render and dual-branch
denoising structure directly matches the project interface
([paper](https://arxiv.org/abs/2606.30003),
[official repository](https://github.com/Heey731/GeoEdit)).

## Gate rule

The gate closes only after the three frozen primary controls/baselines produce
non-overwritten run manifests for all four anchors. At this checkpoint the
contract-ready count is three, but the four-anchor batch-complete count is zero.
The next dependency is `common_proxy_v1`, not a prompt or seed sweep.
