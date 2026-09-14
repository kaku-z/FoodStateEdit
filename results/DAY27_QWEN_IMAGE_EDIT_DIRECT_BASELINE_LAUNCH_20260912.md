# Day 27 Qwen-Image-Edit direct baseline launch

## Material Passport

- Experiment ID: `day27_qwen_image_edit_direct_baseline_v1`
- Artifact type: launch record; not an experiment result
- Verification status: `LAUNCHED_PREFLIGHT_PASSED_NOT_YET_REVIEWED`
- Execution host: `gp40.cs.uec.ac.jp`
- Physical GPU: `6` (`NVIDIA RTX A6000`)
- Launch date: `2026-09-12`
- Process ID observed after launch: `1015452`

## Frozen scope

- Model: local `Qwen-Image-Edit-2511` through `QwenImageEditPlusPipeline`
- Inputs: the same ramen, soup, fried-rice and synthetic-cake development images used by Days 25 and 26
- Seeds: `1`, `2`, `3`
- Candidates: exactly one per case/seed; no best-of selection or failed-seed replacement
- Inference: 40 steps, true CFG `4.0`, BF16, model CPU offload
- Projection: exact source restoration outside each predeclared edit support
- Expected outputs: `12`
- Model downloads: none

## Frozen evidence

- Config: `configs/day27_qwen_image_edit_direct_baseline_v1.json`
- Config SHA-256: `d2a8a144bbfa1bf48e370dc7f7c399c92df3c6812da2cf1e31e87b8caa9cab7a`
- Runner: `scripts/run_qwen_image_edit_direct_baseline.py`
- Runner SHA-256: `4a3a63d2873cb77c0061747a0f57f0e7e06bfeae67a51861d23ecb8543a539fb`
- Model root: `/host/space0/guo-z/projects/multimodal_food/models/qwen-image-edit-2511`
- Runtime root: `/host/space0/guo-z/tf-ufi/outputs/day27_qwen_image_edit_direct_baseline_runtime_v1`
- Output root: `/host/space0/guo-z/tf-ufi/outputs/day27_qwen_image_edit_direct_baseline_v1`
- Preflight: `/host/space0/guo-z/tf-ufi/outputs/day27_qwen_image_edit_direct_baseline_v1_preflight.json`
- Log: `/host/space0/guo-z/tf-ufi/outputs/day27_qwen_image_edit_direct_baseline_runtime_v1/day27_qwen_image_edit_direct_baseline_v1.log`

The preflight passed the GPU type, free-memory, utilization, zero-process,
host-memory, input-hash and model-file hash checks. Pipeline loading completed
and the first 40-step generation was observed in progress. Per user request,
the job is left running without continuous monitoring.

Writing the optional runtime PID sidecar failed with a user disk-quota warning;
this did not prevent the log, preflight, output directory or Python worker from
being created. The authoritative launch PID was obtained from the live process
table and is recorded above. No existing file was deleted to address the quota.

## Claim boundary

This file records only a successful launch and one observed in-progress
generation. It is not evidence that any output completed or that Qwen achieves
action, photo, preservation, generalization or superiority success. Completion,
hash retrieval and blinded evaluation remain pending.
