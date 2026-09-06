# Day 13 post-outage recovery

Date: 2026-09-06 UTC.

All five hosts were reached after allowing 40 seconds for the multihop SSH
handshake. gp38, gp39 and gp40 each reported eight idle RTX A6000 GPUs with
48,539 MiB free, 0% utilization, and no compute processes. Available host memory
was 253,557 / 252,829 / 253,593 MiB respectively. gp41 contains A40 and gp42
contains Blackwell, so neither is an eligible execution target.

The original gp40 Day 13 dataset, bundle v4, all three selected training roots,
and weighted v3 nohup log/exit file under `/tmp` were absent. A bounded search
under the user's persistent TF-UFI directory returned no Day 13 backup. Earlier
records of two completed arms remain historical observations; their checkpoint
bytes have not been recovered. No semantic conclusion can be drawn from the
interrupted/lost weighted run. We did not delete any remote evidence.

The Python environment, frozen training and inference runtimes, base models,
and model hash audit remain in persistent storage. The retry uses the new root
`/host/space0/guo-z/tf-ufi/outputs/day13_recovery_20260906T0115Z`, confirmed absent
before creation. Dataset, training outputs and evidence are now placed there.

Recovery execution config:
`configs/flexible_completion_execution_recovery_20260906_v1.json`

SHA-256: `bd17a92e7cfdfe25186949b5e32a0cc3385cfbd2ac7a424c4e90d60dcdf63a7b`.

The training settings, implementation hashes, model bytes, original dataset
manifest, geometry, resource gates and scientific comparison remain unchanged.
Only the dated recovery metadata and dataset/training/evaluation storage paths
change. All three arms restart with the original seed and initialization policy.
The original execution config is preserved byte for byte. Each arm must still
pass its own fresh resource preflight, run serially, and use an unused root.
Automatic heartbeat remains paused.

Upload archives:

- dataset.tar: `8370cea8a7463454f928f74c328cfc46e1fe380009a0481cb1b586fcc1872f9a`
- bundle.tar: `997d56ae870c5477300eac497e723da22ebf8f44191d2270006f6f5276bd6f28`

Recovery does not establish method effectiveness. Checkpoint validation,
cross-arm random-trace matching, five-condition evaluation, and independent
blinded review are required before judging the proposed contribution.

## Posttraining schema compatibility

`scripts/prepare_flexible_completion_posttraining.py` is an independent
preparation utility, not a replacement for any frozen trainer or evaluator.
It runs the frozen cross-arm verifier and validates both step 16 and step 32
for all three arms. Each raw validation report must bind to the exact checkpoint
path and SHA-256, contain 160 finite LoRA tensors, and report 80 official-loader
updates before it can be used to freeze evaluation.

The existing checkpoint validator requires its historical Day 9 config schema.
The utility makes explicitly labelled compatibility configs without altering
the validator. The raw validator's historical method and two-pseudo-target
boilerplate are not Day 13 dataset provenance: this experiment still uses one
unique synthetic sample duplicated into two rows.

The old inference runtime template stores `model_hash_audit` as a string,
whereas the Day 13 preflight expects a path/size/SHA-256 record. The utility
normalizes only this representation after verifying the same model audit,
and checks all frozen Day 13 inference parameters before configuration creation.
No loss weights, seeds, steps, model files, controls or scientific gate change.

Local verification: 125 tests run, 122 passed and 3 skipped because the local
interpreter lacks the frozen geometry runtime. Day 13-specific tests: 20 passed.
