# Day 13 3D-guided flexible completion experiment design

## Goal

Produce the smallest causal experiment that could support the proposed thesis
contribution: explicit 3D control fixes the action scaffold, while diffusion
completes the flexible food appearance without discarding pinch contact,
connectivity, lift, or final payload hold.

The pilot uses only the previously audited seen synthetic udon sequence. It
does not evaluate the blind pasta/fork anchor and cannot establish
generalization.

Frozen design config: `configs/flexible_completion_mechanism_pilot_v0.json`,
SHA-256 `6eb599dfbb1b7b2b9bdad955f70c9bc4e5d496407408d9f3717cb3e6b76280b5`.

## Why the previous experiment was insufficient

The frozen DiffSynth FlowMatch objective computes one global uniform MSE over
the predicted and target flow tensors. Thin strands, the pinch point, and the
bowl-connected endpoint occupy only a small fraction of the latent volume, so
their errors can be dominated by the surrounding photographic region.

Day 8 already showed that a relative-3D control can preserve fork identity and
trajectory while VACE discards the wrapped spaghetti. Day 11 and Day 12 showed
that more training and single-sample isolation change pixels and reduce some
MAE values without producing a clearer flexible action. The next experiment
therefore changes the supervision interface, not the number of samples or the
blind evaluation policy.

## Three-arm causal design

| Arm | Control | Training objective | Purpose |
| --- | --- | --- | --- |
| planar-uniform | frozen Day 12 planar pseudo-motion | stock uniform FlowMatch MSE | contextual baseline |
| relative3d-uniform | new relative-3D udon scaffold | stock uniform FlowMatch MSE | isolate the control representation |
| relative3d-topology-weighted | exactly the same new relative-3D scaffold | proposed topology-weighted FlowMatch MSE | isolate flexible completion supervision |

All three arms start from identical LoRA initialization and use the same data
order, timestep draws, diffusion noise, precision, optimizer state, 32-step
budget, model bytes, target, reference, prompt, and support. The primary causal
comparison is the third arm against the second. The first arm is not allowed to
substitute for that comparison.

## Relative-3D udon scaffold

The normalized-pinhole scaffold contains two rigid chopsticks and one
continuous flexible curve. One curve endpoint stays fixed in the bowl. From
contact onward, the other endpoint follows the moving pinch anchor between the
chopsticks. Depth-aware projection must encode the pinch crossing instead of
using a fixed 2D paint order.

The same 3D state produces four synchronized outputs:

1. the RGB VACE control video;
2. a flexible-strand mask video;
3. a pinch-contact mask video; and
4. a source-connection mask video.

This remains relative 3D, not recovered scene depth. Frame zero and pixels
outside the frozen support must equal the source exactly.

## Proposed objective

For binary latent-aligned masks `M_strand`, `M_contact`, and
`M_source_connection`, construct

`W_raw = 1 + 3 M_strand + 7 M_contact + 5 M_source_connection`.

Each full-resolution mask is reduced to the exact prediction volume with
adaptive max pooling so thin structures survive latent compression. The
non-first-frame weight volume is divided by its own mean, and the loss is

`sum(W * squared_error) / sum(W)`.

Mean normalization prevents the proposed arm from receiving a larger global
learning-rate-like scale. Overlapping masks add, so the pinch and connection
regions remain the highest-priority flexible structures.

## Frozen evaluation

The primary checkpoint is step 32. Inference uses seed 1, 21 frames, 20 steps,
VACE scale 1, and TTM disabled. Conditions are anonymized before review.

Machine checks include per-phase target-support MAE, topology-weighted MAE,
pinch-ROI MAE, connection-ROI MAE, decoded frame count, output hashes, and
exact outside-support preservation. These metrics do not replace visual
semantics.

At least two independent reviewers must separately judge two chopsticks,
pinch contact, one continuous strand, visible return to the bowl, payload lift,
final hold, and photo realism.

## Positive-contribution gate

The proposed arm passes only if all of the following hold:

- both reviewers prefer it over the relative3d-uniform arm for pinch,
  connected lift, and final hold;
- it visibly improves over relative3d LoRA-off;
- topology-weighted target MAE and pinch-ROI MAE each improve by at least 5%;
- neither reviewer judges photo realism worse than relative3d-uniform;
- every condition preserves all pixels outside support exactly; and
- every expected artifact and SHA-256 verifies.

A numerical-only change fails. Step 16 cannot replace step 32 as the selected
winner, and another seed cannot rescue failure after output inspection.

If only relative3d-uniform improves over planar-uniform, the result supports a
3D representation contribution but not the proposed flexible-completion
contribution.

## Current execution status

The design is frozen but deliberately non-executable. The relative-3D udon
builder, seeded training entry points, preflight, validator, evaluation runner,
and topology evaluator must be implemented, tested, and hash-frozen in a new
execution configuration before any GPU launch. All remote paths must be new,
and the existing no-download, no-preemption A6000 resource gate remains in
force.
