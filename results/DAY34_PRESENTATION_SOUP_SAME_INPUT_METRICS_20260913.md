# Day 34 same-input soup presentation metrics

All three methods use the same selected synthetic soup input and seed 1.
Pixel metrics measure input fidelity, not whether the requested action is correct.

| Method | Full PSNR ↑ | Full SSIM ↑ | Outside SSIM ↑ | Outside MAE ↓ | Outside exact ↑ | Action | Photo | Preservation | Strict |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| ChordEdit | 31.78 | 0.982 | 1.000 | 0.000 | 100.0% | ✗ | ✗ | ✓ | ✗ |
| Qwen-Image-Edit (raw) | 15.73 | 0.628 | 0.639 | 18.192 | 0.3% | ✓ | ✓ | ✗ | ✗ |
| FoodStateEdit controlled VACE | 20.68 | 0.965 | 1.000 | 0.000 | 100.0% | ✓ | ✓ | ✓ | ✓ |

## Interpretation

- ChordEdit obtains exact protected-pixel preservation through local compositing but does not produce the requested action.
- Raw Qwen-Image-Edit produces the strongest photographic action result in this single case, while globally changing the framing/background.
- FoodStateEdit is the only strict pass under this internal case-level rubric because it combines a successful static spoon state with exact projected background preservation.
- This is one selected synthetic example and cannot support a statistical superiority or generalization claim.
