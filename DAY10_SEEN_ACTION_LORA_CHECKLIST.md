# Day 10 seen-family action-LoRA checklist

- [x] Freeze the two training-family samples without changing prompts, controls, seed, or checkpoint.
- [x] Require one resident pipeline load per worker and LoRA-off before LoRA-on.
- [x] Inject all 80 trained high-noise VACE branches and keep `vace2` untouched.
- [x] Run only on two safe idle RTX A6000 GPUs without model downloads.
- [x] Preserve both preflight reports and both run manifests with verified hashes.
- [x] Decode 21 frames per condition and preserve source pixels outside edit support exactly.
- [x] Compare target-support MAE under identical seeds.
- [x] Review frames 0, 5, 10, 15, 18, and 20 for action/contact and photo realism.
- [x] Attribute the broth/spoon success to the base condition because it exists with LoRA disabled.
- [x] Close the current adapter as underfit: zero semantic LoRA gains on two seen families.
- [x] Prohibit a generalization or adapter-benefit claim.
- [x] Require a phase-varying overfit sanity gate before another blind fork run.
