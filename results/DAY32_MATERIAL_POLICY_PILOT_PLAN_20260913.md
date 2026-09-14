# Day 32 material-policy pilot execution plan

Date: 2026-09-13

## Frozen scope

Day 32 applies the Day 31 family-template controls to all 20 development-pilot
images.  It is a pilot rule-freeze experiment, not held-out evaluation.  The
four reported arms are native-input VACE, planar-proxy VACE, fixed scale-1.0
relative-3D VACE, and the candidate material-adaptive relative-3D arm.  The
candidate scale is 1.0 for liquid/granular and 0.8 for strand/strand-contact;
when it equals 1.0, the fixed output is reused rather than generated twice.

The execution config is `configs/day32_material_policy_pilot_v1.json`, SHA-256
`370f45994b400b0cb7762d7b0951d635e1eea1e40b00e54b1709f1a46a143cd9`.
The runner SHA-256 is
`7bd11c8fce21c41a5ebe3446164dd77032ffa9562919646fbd364e372419ec19`.
The four non-overlapping shards contain 51, 51, 54 and 54 generations,
respectively: 210 total, with seeds 1/2/3, 21 frames, 20 steps, TTM off and no
LoRA.  The runner records runtime, CUDA peak allocation/reservation, output
hashes and exact outside-support preservation for every job.

## 2026-09-13 resource event

A full read-only audit excluded occupied gp38, A40-only gp41 and Blackwell gp42.
gp39 initially had eight eligible idle A6000s, so the three frozen runtime files
were uploaded and rehashed.  Before either worker was submitted, another user
occupied all eight gp39 GPUs.  The fresh launch-time check showed 29,320 MiB
free and 100% utilization on GPUs 0/1 with foreign compute processes.  No Day32
preflight report, output root or model process was created, and no foreign
process was modified.  This is a resource race, not a technical or scientific
result.

After Day 25 released gp40 GPUs 0/1, a new full host audit admitted both cards.
`shard_b` passed its own preflight on physical GPU1 and entered pipeline
loading.  `shard_a` did not start: the delayed gp39 submission had already
created the shared v1 preflight path, and the gp40 runner therefore failed
closed when opening that report with exclusive creation.  No shard-a output
root exists.  The preserved gp39 report records the actual resource rejection;
the shared `shard_a_launcher.log` was unfortunately truncated by the later
shell redirection and now records only the exclusive-path failure.  Any
shard-a retry requires a new config, output, preflight and launcher-log path and
must not be automatic.

## Next gate

Only a fresh complete gp38--gp42 audit may admit a new A6000 target.  At most two
workers may run per host.  A successful launch must use the frozen paths and
hashes above; any failed worker is preserved and may be retried only in a new
versioned output path.  The 40-image test split remains locked until Day32 is
complete and its material-level decision is recorded without per-image tuning.

## Claim boundary

The config and resource record establish readiness and safe non-execution only.
They do not provide a Day32 output, method effectiveness, relative-3D gain,
held-out generalization, physical correctness or photo-realism evidence.
