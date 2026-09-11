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
- [x] Upload and byte-verify the frozen dataset and runtime on one candidate A6000 host.
- [x] Run fail-closed preflight on a safe idle A6000 with sufficient host memory.
- [x] Train without downloads or overwriting any existing remote path.
- [x] Validate checkpoint structure and official VACE loadability.
- [x] Compare LoRA-off and all four checkpoints on both seen samples using the same seed.
- [x] Review phase events and photo realism separately.
- [x] Apply the expansion gate; no clear seen semantic gain was found, so keep balanced training and the blind fork blocked.

Execution note (2026-09-01): the first sweep launch did not produce an
evaluable condition. The broth/spoon worker was blocked by the fail-closed GPU
gate after another user's process appeared on the requested card. The udon
worker passed preflight, but another process then occupied the card during
pipeline startup and the worker preserved a pre-condition OOM failure. Both
events are resource evidence, not model results. Retries must use new output
and preflight paths on a conflict-free A6000; the blind fork remains blocked.

Completion note (2026-09-01): a serial retry completed both samples on gp39
GPU0 under new output and preflight paths. All 10 conditions decoded 21 frames,
loaded the pipeline once per sample, preserved pixels outside support exactly,
and passed local SHA-256 verification after pullback. Udon's numeric best was
step 64 and broth's was step 16, but contact-sheet review found `0/2` clear
phase-action/contact improvements and `0/2` clear photo-realism improvements.
The seen synthetic overfit-capacity gate is closed negative; balanced expansion
and another blind fork run remain prohibited.

Post-outage reproduction note (2026-09-12): the frozen assets were restored to
gp40 after `/tmp` loss and all 35 upload files were byte-verified. A new output
root and new preflight reports were used on idle RTX A6000 GPU4. Both samples
completed serially across all five conditions with one pipeline load, 21 frames,
and exact outside-support preservation. All 45 pulled files matched their remote
SHA-256. Manual review again found `0/2` clear phase-action/contact gains and
`0/2` clear photo-realism gains, including a broth/spoon conservation failure
where garnish remained in the bowl while also appearing in the spoon. The
negative gate is reproduced; no balanced expansion or blind fork is authorized.
