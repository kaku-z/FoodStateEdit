# Day 12 phase-action single-sample isolation result

## Outcome

Both dedicated single-sample checkpoint sweeps completed, but neither passes the
visible semantic gate. The result is `0/2` clear phase-action/contact gains over
both required baselines and `0/2` clear photo-realism gains over LoRA-off.
Cross-sample interference is therefore not supported as the explanation for the
negative shared Day 11 adapter, and blind fork evaluation remains blocked.

This is a seen synthetic single-sample isolation diagnostic. It is not evidence
of held-out generalization, real-data performance, physical correctness, or
paper-level photo realism.

## Execution and integrity

- Host: `gp38.cs.uec.ac.jp`, physical GPU 0, NVIDIA RTX A6000.
- Both jobs ran serially after a read-only resource audit and passed the frozen
  A6000, free-memory, utilization, process, and host-memory gates.
- Conditions were LoRA-off and dedicated steps 16, 32, 48, and 64, with seed 1,
  21 frames, 20 inference steps, VACE scale 1, and TTM disabled.
- Each sample loaded the pipeline once, decoded all 21 frames for every
  condition, and changed zero pixels outside the declared support.
- All 40 pulled run-package files, all 36 manifest-recorded outputs, and all 14
  supporting training/preflight evidence files matched the remote SHA-256
  values; mismatch count was zero.
- No model was downloaded. No remote output path was reused or overwritten.
- The client SSH connection observing the spoon run reset after step 32, but the
  remote worker continued normally. Its fresh directory contains all five
  conditions, a complete manifest, and `COMPLETE`; this is not classified as a
  technical failure.

## Matched exposure comparison

Dedicated step 32 and shared Day 11 step 64 each saw the selected sample 32
times. Dedicated step 64 is reported only as an additional capacity probe.

| Sample | Off MAE | Dedicated 16 | Dedicated 32 | Dedicated 48 | Dedicated 64 | Shared Day 11 step 64 | Dedicated 32 minus shared 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Udon/chopsticks | 13.6511 | 13.6332 | 12.7542 | 12.4349 | 12.0944 | 11.9998 | +0.7544 |
| Broth/spoon | 21.2355 | 20.6891 | 20.1368 | 22.4480 | 20.1830 | 23.8928 | -3.7560 |

Lower target-support MAE is descriptive only. It does not pass the gate without
a visible improvement in approach, contact, acquisition, lift, and hold.

## Phase-action and contact review

### Udon/chopsticks

The dedicated conditions remain nearly identical to LoRA-off and shared Day 11
step 64. The same chopstick entry, ambiguous contact, and final noodle geometry
remain visible; there is no clearer pinch, payload acquisition, bowl-connected
lift, or hold. The target pseudo-sequence contains a much more explicit rising,
connected strand. Dedicated step 32 is also numerically worse than the matched
shared checkpoint.

### Broth/spoon

Dedicated step 32 is numerically closer and visually more coherent than shared
Day 11 step 64, but the same spoon approach, broth payload, and final hold are
already present with LoRA disabled. The adapter therefore adds no clear phase
or contact event. The target contains a cleaner metal spoon and simpler payload
than the generated conditions.

## Photo-realism review

Udon remains broadly food-photographic but retains soft, ambiguous
utensil-contact and lifted-noodle geometry. No dedicated checkpoint clearly
improves it over LoRA-off.

Broth step 32 is more coherent than the degraded shared Day 11 step 64 result,
but is not clearly better than LoRA-off. Dedicated steps 48 and 64 introduce a
conspicuous pale or translucent rim around the spoon bowl and are less
photographic.

## Decision

- Matched cross-sample-interference gate: **failed**.
- Dedicated single-sample overfit-capacity gate: **failed**.
- Balanced multi-family expansion: **blocked**.
- Blind fork evaluation: **blocked**.
- Next diagnostic target: the supervision, loss, or render-control interface,
  under a separately frozen bounded experiment.

Machine-readable evidence and every reviewed contact-sheet hash are recorded in
`results/day12_phase_action_isolation_result_v1.json`.
