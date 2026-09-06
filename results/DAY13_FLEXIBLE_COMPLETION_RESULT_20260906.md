# Day 13 relative-3D topology-weighted flexible-completion result

Date: 2026-09-06. Status: complete negative primary-gate result, preserved.

## Bottom line

The experiment executed correctly, but it does **not** prove that the proposed
topology-weighted adapter is effective. Against the matched relative-3D uniform
baseline, topology-weighted training reduced topology-weighted-volume MAE by
2.22% and pinch-contact ROI MAE by 2.83%. The precommitted gate required at
least 5% on both metrics plus a clear blinded semantic preference. The numeric
gate therefore failed before human review, and technical inspection found no
clear visible difference between the two conditions.

This is still useful: the proposed loss moved all recorded relative-3D metrics
in the intended direction, but the effect is too small to support a positive
claim. The defensible pre-defense statement is **weak directional evidence,
not demonstrated effectiveness**.

## Execution validity

- gp40 RTX A6000 GPU 0 passed a fresh 107-check preflight for each later arm;
- all three arms ran 32 optimizer steps and saved step 16 and step 32;
- initial trainable aggregate SHA-256 was identical across arms:
  `c16056a55c89ec92318674358ea519a2e3425dcb4ff20d80b59b996279f6231e`;
- the matched training-row/timestep/noise sequence was identical:
  `b291714145168af742fd58b9c3f53b8042c993bed6c58c8c4b15e802d04226b7`;
- all six checkpoints contained 160 finite LoRA tensors and the official
  loader updated 80 VACE layers;
- evaluation used one resident pipeline, seed 1, 21 frames, 20 steps,
  VACE scale 1, and TTM off for all five conditions;
- every output decoded to 21 frames and all pixels outside support were
  preserved exactly.

The frozen Flash-Attention runtime warns that its backward kernel is
nondeterministic. Seeds, initialization and sampled noise were matched, but
bitwise training reproducibility is not claimed.

## Metrics

| Condition | Support MAE | Topology MAE | Pinch MAE | Connection MAE |
|---|---:|---:|---:|---:|
| planar, LoRA off | 13.651 | 21.797 | 28.235 | 15.901 |
| planar, uniform step 32 | 12.971 | 21.745 | 29.187 | 14.626 |
| relative-3D, LoRA off | 15.272 | 33.004 | 38.968 | 24.430 |
| relative-3D, uniform step 32 | 15.077 | 31.063 | 38.053 | 22.146 |
| relative-3D, topology-weighted step 32 | 15.008 | 30.373 | 36.974 | 21.931 |

Primary weighted-versus-uniform comparison:

- topology MAE improvement: **2.22%** (required >=5%);
- pinch-contact MAE improvement: **2.83%** (required >=5%);
- result: **numeric gate fail**.

Context only: topology-weighted versus relative-3D LoRA-off improved support,
topology, pinch, and connection MAE by 1.73%, 7.97%, 5.12%, and 10.23%,
respectively. These comparisons mix adapter training with the proposed loss and
are not the primary causal test. Relative-3D uniform had 16.23% worse global
support MAE than planar uniform in this one sample.

## Visual finding and review status

The generated contact sheets show a continuous noodle bend near the chopsticks,
but neither the uniform nor weighted relative-3D result clearly depicts a large,
unambiguous lifted strand with a visibly improved pinch and final hold. Their
appearance is nearly indistinguishable in the six frozen review frames.

Matched relative-3D uniform baseline:

![relative-3D uniform step 32 review](day13_flexible_completion_20260906/relative3d_uniform_step32_review.png)

Proposed relative-3D topology-weighted result:

![relative-3D topology-weighted step 32 review](day13_flexible_completion_20260906/relative3d_topology_weighted_step32_review.png)

These displayed files are byte-identical copies of the pulled evaluation
artifacts. Their SHA-256 values are `5cba7b...94d7` and `d57006...31b1`.

The precommitted R1--R5 package is ready for two independent human reviewers at
`artifacts/day13_blind_review_20260906_v1`. The mapping is excluded from that
package. Because the numeric gate already failed, reviewer preference cannot
convert this run into a positive-gate pass; reviews remain useful diagnostic
evidence for the pre-defense.

## Evidence integrity

Pulled evidence is under
`artifacts/day13_recovery_results_20260906T0115Z`. Remote and local inventories
both contain 60 files and 118,922,008 bytes: 0 missing, 0 extra, and 0 SHA-256
mismatches. Evaluation manifest SHA-256:
`35ed63ce96fa0cff6a61df3cd2808accf4395565223f9866ca555f7a2c307427`.

## Decision

Preserve this result as a clean negative mechanism diagnostic. Do not run the
blind fork or balanced family expansion. The next experiment, if any, needs a
new frozen design targeting a larger semantic lift signal; it must not be
presented as a rescue selected after seeing this output.

This experiment uses one seen synthetic sample duplicated into two rows. It
does not establish held-out generalization, real-data performance, physical
correctness, or paper-level photo realism.
