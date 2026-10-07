# gp40 execution, 2026-09-27

The synthetic material-transport oracle was executed successfully on
`gp40.cs.uec.ac.jp` using `/host/space0/guo-z/envs/geoedit/bin/python`.

Remote experiment directory:
`/host/space0/guo-z/tf-ufi/material_oracle_e40c5d5937a0`.

Retrieved report and images:
`outputs/material_server_execution_20260927_v1/server_results/`.
Deployment record: `outputs/material_server_execution_20260927_v1/remote_run.json`.

Six unit tests passed before the pilot. Sixteen seeded oracle cases finished in
27.379 seconds. The seeds vary cake height within a small five-height fixture
family; this is an implementation check, not sixteen independent real scenes.
All cases have zero modeled-volume error, duplicate/missing IDs, original-location
residual cells and destination correspondence errors.

The copy negative control increased modeled volume by 1/6 and duplicated all
selected IDs. The shuffle negative control preserved volume and occupancy but
corrupted every selected voxel's material ID. This establishes that conservation
alone does not detect identity errors. These are deliberately broken controls,
not comparisons with state-of-the-art image editors.

Archive and code hashes were verified, and all case images were retrieved.
The pilot used CPU only. At preflight, all eight A6000 GPUs were occupied by
another user's jobs with 99–100% utilization. No job was terminated or modified.

This run validates only the complementary partition, integer rigid transport,
shared canonical appearance, discrete food-collision checks and first-hit ray
renderer. It does not validate learned single-image reconstruction, actual mass,
spoon contact or photorealistic editing. Next implementation gate is finite tool
geometry and reachable/unreachable contact, followed by an audit of real
calibrated data before launching a learned scene encoder.
