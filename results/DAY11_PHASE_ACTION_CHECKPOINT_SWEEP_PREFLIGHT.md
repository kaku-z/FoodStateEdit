# Day 11 phase-action checkpoint-sweep execution status

## Outcome

The same-seed, five-condition seen-sample sweep is frozen but has not yet
produced an evaluable condition. The first launch was stopped by two independent
GPU resource races on gp40. Neither event is evidence about action learning,
photo realism, or checkpoint quality.

## Preserved evidence

- Broth/spoon requested physical GPU 1. Its preflight passed 70 of 71 checks
  and failed only `gpu_gate` after a `chen-q` compute process appeared. The
  runner stopped before creating its sample output directory. The tracked
  report hash is
  `e719b13ce9218ebdff89a21905e246556ebdc6b6ded3d046bab88cce516c2015`.
- Udon/chopsticks requested physical GPU 0. All 71 preflight checks passed, but
  a separate `chen-q` process then grew to roughly 43.5 GiB while the pipeline
  was starting. The worker failed before completing `lora_off`, recorded zero
  conditions, and preserved `technical_failure_preserved`. The tracked
  preflight hash is
  `d5359018d9d875a51fca201da6ba8333fe228ed5f1e215ef5ecf02132a59d0a2`;
  the local ignored failure manifest is retained under
  `artifacts/day11_phase_action_checkpoint_sweep_failure_gp40_v1/`.

The copied files were SHA-256 checked against gp40 byte for byte. No process
belonging to another user was stopped or modified.

## Retry gate

Retries must be serial, use new output and preflight paths, and pass the frozen
A6000 gate: at least 48,000 MiB free, at most 5% utilization, no compute
process, and at least 80,000 MiB available host memory. Existing failure paths
must not be overwritten. The held-out fork run and balanced expansion remain
prohibited until both seen samples have complete checkpoint sweeps and separate
phase-event and photo-realism reviews show a clear semantic gain.
