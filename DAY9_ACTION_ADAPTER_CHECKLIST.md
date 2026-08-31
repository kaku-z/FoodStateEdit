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
- [x] Preserve the v1 remote preflight blocked by a 62-character copied tokenizer hash; correct it to the previously verified 64-character Day 7 hash without creating a training output.
- [x] Upload and hash-verify a new immutable runtime bundle and v3 dataset.
- [x] Run only on a safe idle RTX A6000; preserve any failure directory.
- [x] Validate the final checkpoint with the official offline VACE LoRA loader.
- [x] Preserve the first fork comparison after its runtime log proved that the
  stock low-VRAM hot-loader patched zero VACE tensors.
- [x] Correct the wrapped-block injection without changing base model files;
  require all 80 high-noise VACE LoRA pairs and keep `vace2` untouched.
- [x] Freeze and run the same-seed LoRA-off/on fork comparison.
- [x] Review action/contact and photo realism separately across six key frames.
- [x] Close the blind pilot negative: numerical effect, no semantic action gain.
- [x] Update the evidence inventory without claiming generalization.
