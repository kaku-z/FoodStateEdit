# Day 25 rice/cake resource-preflight failure

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-13
- Verification Status: HASH_VERIFIED_NO_INFERENCE
- Version Label: day25_same_condition_ablation_gp40_resource_failure_v1

## Existing completed scope

The preserved Day 25 outputs already contain complete ramen (`15/15`) and soup
(`10/10`) case runs. Each loaded the pipeline once and records exact
outside-support preservation for every completed condition.

## New attempt

Rice was submitted to gp40 physical GPU 0 and cake to physical GPU 1 after a
read-only audit reported both GPUs as idle RTX A6000 devices. Before the runner's
own fail-closed preflight completed, another user's process occupied all gp40
GPUs. Both preflights rejected the jobs on free-memory, utilization and
zero-compute-process checks.

- Rice output root created: no; model inference: zero.
- Cake output root created: no; model inference: zero.
- No process was terminated, modified or preempted.
- No automatic retry was performed.
- Four downloaded evidence files match their remote SHA-256 values.

## Required next step

The v1 preflight paths now exist as failure evidence and must never be
overwritten. A retry requires an explicitly frozen v2 config with new rice/cake
output roots, new preflight paths and a freshly audited safe RTX A6000. The
40-image held-out benchmark remains locked until the full Day 25 development
ablation is reviewed and the single Ours rule is frozen.

## Evidence

- Machine-readable record: `results/day25_same_condition_ablation_gp40_resource_failure_v1.json`
- Local evidence: `artifacts/day25_same_condition_ablation_resource_failure_20260913_gp40_v1/`

## Claim boundary

This is a resource-preflight failure, not a model failure or ablation result. It
supports no effectiveness, held-out, realism, superiority or generalization
claim.
