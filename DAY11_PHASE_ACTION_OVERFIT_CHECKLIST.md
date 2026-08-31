# Day 11 phase-action overfit checklist

- [x] Close the Day 10 seen-family comparison before changing training.
- [x] Keep the fork/pasta family absent from training.
- [x] Reuse only the two audited synthetic final targets and no downloaded models.
- [x] Replace repeated final frames with explicit source, approach, contact, lift, and final-hold phases.
- [x] Mark the translated-patch sequence as pseudo-motion, not physical or real-video ground truth.
- [x] Make frame 0 equal the no-utensil source and final hold equal the audited Day 9 keyframe.
- [x] Preserve all pixels outside the phase motion union exactly before video encoding.
- [x] Keep each motion union below 20% of the image.
- [x] Freeze 64 optimizer steps with checkpoints at 16, 32, 48, and 64.
- [x] Require a seen-family checkpoint sweep before any new blind fork run.
- [ ] Upload and byte-verify the frozen dataset and runtime on one candidate A6000 host.
- [ ] Run fail-closed preflight on a safe idle A6000 with sufficient host memory.
- [ ] Train without downloads or overwriting any existing remote path.
- [ ] Validate checkpoint structure and official VACE loadability.
- [ ] Compare LoRA-off and all four checkpoints on both seen samples using the same seed.
- [ ] Review phase events and photo realism separately.
- [ ] Continue to balanced multi-family training only if seen semantics improve clearly.
