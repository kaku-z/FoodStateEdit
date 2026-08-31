# Day 9 action-adapter checklist

- [x] Re-audit historical action results under the current visible-binding gate.
- [x] Exclude state-only, failed, photo-partial teacher, and held-out fork artifacts.
- [x] Disclose the only two eligible targets as synthetic ImageGen pseudo-labels.
- [x] Keep `pasta_fork_001` absent from all training rows.
- [x] Reject raw-oracle v2 because it changes nearly the full frame.
- [x] Build v3 with non-identity target/control pairs and exact protected pixels.
- [x] Freeze offline trainer/model/data/resource/claim contracts.
- [x] Add fail-closed preflight and launcher.
- [x] Pass the Day 9 tests and the full 75-test regression suite.
- [ ] Upload and hash-verify a new immutable runtime bundle and v3 dataset.
- [ ] Run only on a safe idle RTX A6000; preserve any failure directory.
- [ ] Validate the final checkpoint with the official offline VACE LoRA loader.
- [ ] Freeze and run the same-seed LoRA-off/on fork comparison.
- [ ] Review action/contact and photo realism separately.
- [ ] Update the evidence inventory without claiming generalization.
