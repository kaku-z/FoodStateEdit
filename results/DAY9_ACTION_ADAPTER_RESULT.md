# Day 9 action-adapter result

Date: 2026-08-31

## Outcome

Training and checkpoint validation succeeded, but the blind fork action did
not. The rank-8 high-noise VACE LoRA completed 16 optimizer steps on two
bounded, non-identity synthetic pseudo-targets. Its step-16 checkpoint contains
160 finite, nonzero tensors in 80 A/B pairs, and the official single-model
loader matched all 80 targets.

The first full Wan2.2 evaluation was invalid as a LoRA-on comparison. The
frozen low-VRAM runtime wraps complete VACE attention blocks, while its stock
hot-loader searches only for exposed `AutoWrappedLinear` modules. The log
reported `0 tensors are patched by LoRA`, and the MP4, raw selected frame, and
projected edit were byte-identical to Day 8. This preserved run is useful
implementation-failure evidence, not adapter-quality evidence.

The corrected v2 runner injects the 80 low-rank branches into the inner Linear
modules of the wrapped high-noise VACE blocks. It does not alter the base model
files, fuse tiny updates into BF16 weights, or touch the untrained low-noise
`vace2`. The final manifest records exactly 80 injected branches, one pipeline
load, one adapter load, 21 decoded frames, and zero changed pixels outside the
frozen motion support.

## Same-seed comparison

| item | Day 8 LoRA-off | Day 9 corrected LoRA-on |
| --- | --- | --- |
| fork training occurrences | 0 | 0 |
| seed / frames / steps | 1 / 21 / 20 | 1 / 21 / 20 |
| VACE scale / TTM | 1.0 / off | 1.0 / off |
| selected frame | 18 | 18 |
| injected LoRA branches | 0 | 80 |
| outside-support maximum difference | 0 | 0 |
| visible fork | yes | yes |
| wrapped and lifted spaghetti | no | no |
| strict action success | no | no |
| strict photo success | no | no |

The corrected adapter has a measurable numerical effect. At frame 18, its raw
output differs from LoRA-off by RGB MAE 2.554. After exact projection, global
MAE is 0.407 and edit-support MAE is 2.539; 90.82% of support pixels change.
Those changes are small appearance perturbations, not a topology change.

Reviewing frames 0, 5, 10, 15, 18, and 20 gives the same action verdict. The
fork enters from the right and presses into sauce near the plate edge. No
continuous spaghetti wraps the tines, no payload rises with the fork, and no
lifted strand remains connected to the plate. The local fork/contact rendering
also remains malformed enough to fail the strict photo-realism gate.

## Interpretation and next gate

The infrastructure issue is solved: the trained adapter now reaches the
intended high-noise VACE computations. The learning problem is not solved. Two
synthetic pseudo-targets and 16 steps do not support blind chopsticks/spoon to
fork transfer.

The next experiment must separate underfitting from failed cross-family
transfer by evaluating LoRA-off/on on the two seen families. If the adapter
cannot improve its own udon and broth targets, increase supervision and/or
optimization before any new blind test. If it improves seen families but not
fork, expand balanced action/contact data, including fork examples in training
and reserve different fork scenes—not the entire utensil family—for testing.
Neither outcome permits a generalization or paper-level photo-realism claim.

Machine-readable evidence: `results/day9_action_adapter_result_v1.json`.
