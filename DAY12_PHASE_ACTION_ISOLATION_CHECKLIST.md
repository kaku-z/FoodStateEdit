# Day 12 phase-action isolation checklist

- [x] Close the shared Day 11 sweep before defining another experiment.
- [x] Keep the two synthetic pseudo-motion targets, phase schedule, prompts, controls, supports, and model bytes unchanged.
- [x] Build one udon-only and one broth-only dataset by duplicating the selected sample into exactly two metadata rows.
- [x] Verify every copied selected-sample file and the phase schedule byte for byte against Day 11.
- [x] Keep the held-out fork occurrence count at zero.
- [x] Keep rank, learning rate, weight decay, epochs, repeat count, timestep range, offload settings, and 64-step total budget unchanged.
- [x] Freeze checkpoints at steps 16, 32, 48, and 64 for both dedicated adapters.
- [x] Declare dedicated step 32 versus shared step 64 as the matched 32-exposure comparison.
- [x] Reserve dedicated step 64 as an additional single-sample overfit-capacity probe, not a matched comparison.
- [x] Require new, non-overwriting remote dataset, training, validation, preflight, and sweep paths.
- [ ] Run both dedicated training arms serially on a conflict-free RTX A6000.
- [ ] Validate both step-64 checkpoints offline through the official VACE loader.
- [ ] Run LoRA-off/step-16/step-32/step-48/step-64 sweeps on the corresponding seen sample with seed 1.
- [ ] Rehash all pulled outputs and review phase action/contact separately from photo realism.
- [ ] Diagnose cross-sample interference only from the matched-exposure comparison.
- [ ] Permit a blind fork run only if both dedicated samples show a clear semantic gain.

This is a seen synthetic single-sample isolation diagnostic. It cannot establish
generalization, real-data performance, physical validity, or paper-level photo
realism. A lower support MAE without a visible approach/contact/acquisition/lift
gain does not pass the semantic gate.
