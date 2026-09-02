# Day 12 single-sample phase-action isolation preflight

Date: 2026-09-02

## Question

The shared Day 11 adapter reduced target-support MAE for udon but produced no
clear semantic improvement, while later checkpoints degraded the spoon render.
This experiment asks whether cross-sample gradient interference is the limiting
factor by training one dedicated adapter per seen synthetic sample.

## Controlled comparison

Both dedicated datasets contain two identical metadata rows for one unique
sample. The underlying target video, VACE control, reference image, prompt,
support mask, final keyframes, and phase schedule are copied byte for byte from
Day 11.

The optimizer and model settings remain unchanged. With two identical rows,
dedicated step 32 contains 32 exposures to the selected sample. It is compared
against shared step 64, where the balanced two-sample run supplied approximately
32 exposures per sample. Dedicated step 64 is reported separately as a larger
overfit-capacity probe and cannot be used as the matched interference test.

## Frozen arms

| arm | config SHA-256 | dataset manifest SHA-256 | metadata SHA-256 |
| --- | --- | --- | --- |
| udon-only | `c6e11696bf55528655677b40d9ebbd8c53fa7d65c4841b34eb8576da9a1e49f3` | `2297d29d301b395663fd6b67f6c21a79e19a72d8f66583791b7f4ae217c02c54` | `ff19012d6d610d3ba5a248c9525bf7824dab03d9b203ca7488214953f41b7f3a` |
| broth-only | `9255aa36bcd5d79d4f8c2c1c6e48db43b377df467fed7f8c27d1d453d12b9112` | `5bc398202fc2636ecdab9d77916fbf52c233af5650601810b4dc4bbc068fbe2c` | `2937d0bc49dccca22132221feb5a5e298a4b66d49845b25c389a6a5e2498c728` |

The isolated dataset builder LF-normalized SHA-256 is
`3f32618cb32cfb61239d855238cf7184e458db8769db8c8db626531bcd1016fe`.
The deterministic configuration builder LF-normalized SHA-256 is
`e7e1f4f00bd961ada1eee2efc7e0644f925686344d2ba3b6b35563ef5f28e13f`.

## Execution gate

Run the two training arms and their evaluations serially. A selected GPU must
be an NVIDIA RTX A6000 with at least 48,000 MiB free, at most 5% utilization,
zero compute processes, and at least 80,000 MiB available host memory. Do not
stop or modify another user's process, do not use A40 or Blackwell hardware, do
not download models, and do not overwrite any existing path.

## Decision rule

- If dedicated step 32 visibly improves a sample over both LoRA-off and the
  shared step-64 result, matched exposure supports a cross-sample-interference
  diagnosis.
- If only dedicated step 64 improves, the evidence supports additional
  single-sample capacity, not a clean interference conclusion.
- If neither dedicated adapter improves its own seen sample semantically, the
  bottleneck remains the supervision/loss/render-control interface.
- Numeric MAE improvement alone never unlocks the blind fork. Both dedicated
  samples must show clear phase-action/contact gains before any blind run.

## Training execution evidence

Both arms completed serially on gp38 GPU0 after fresh resource gates. Udon
completed 64 steps in 1,378.907 seconds; broth completed 64 steps in 937.095
seconds with the model bytes retained in the system page cache. Each arm
produced step-16/32/48/64 checkpoints of 15,354,160 bytes. Both step-64 files
passed the official VACE loader check with 160 tensors, 80 LoRA pairs, rank 8,
and 80 updated linear modules.

## Frozen checkpoint sweeps

The deterministic checkpoint-sweep builder has LF-normalized SHA-256
`3b63cafdedfe4e0bb851f20f5a78c7d59d0ee83bc6ed88877fa936fd154a192c`.
It generated the following byte-frozen corresponding-sample configurations:

| arm | sweep config | SHA-256 |
| --- | --- | --- |
| udon-only | `configs/vace_phase_action_isolated_udon_checkpoint_sweep_v1.json` | `7e40d783f94677b6bcbba3da886f059b03bed3c90d22e61bf5327a9b31c86387` |
| broth-only | `configs/vace_phase_action_isolated_spoon_checkpoint_sweep_v1.json` | `91ae2ca1942905352b5904cd2470513c2ef7656248e42dd4384014d49315b170` |

Each sweep fixes LoRA-off and steps 16/32/48/64, seed 1, 21 frames, 20
inference steps, VACE scale 1, TTM disabled, one corresponding seen sample, and
the previously frozen model/runtime bytes. Inference remains gated on a fresh
resource audit and new non-overwriting output and preflight paths.
