# Day 9 non-identity action-target audit

Date: 2026-08-31

## Decision

The current archive contains **zero real-photo action targets** suitable for
formal adapter training. It contains exactly two usable synthetic
pseudo-targets for a claim-limited mechanism pilot:

1. chopsticks visibly lifting one bowl-connected udon strand;
2. one spoon visibly containing clear broth and a garnish ring.

Both were produced with built-in ImageGen and are therefore labeled as
synthetic pseudo-supervision. They may test whether non-identity supervision
helps VACE preserve utensil-food binding, but they cannot support claims about
real-data training, generalization, or publication-level photo realism.

The pasta/fork family is excluded from training and remains blind. No fork
image, failed fork result, or fork appearance oracle enters the dataset.

## Important re-audit

The older case-20 VACE frame was described as the strongest natural result.
Under the stricter current gate it is not an eligible action target: compared
with its oracle, the pronounced lifted loop collapses to a short contact near
the bowl edge. Using that frame as supervision would teach the exact thin-food
loss we are trying to repair. The geometry-authoritative deterministic edit
still passes its recorded topology checks; this re-audit concerns the natural
VACE appearance frame only.

The staged spoon result remains a semantic/structural pass, but its soft handle
neck, metal reflection, and contact shadow make it unsuitable as the clean
teacher for the first adapter pilot. Its ImageGen oracle is used instead and
is disclosed as a pseudo-label.

## Exclusions

- The strict liquid-fill image changes liquid level but does not show scooping.
- The fried-rice result is a state/mass trace with no rendered action target.
- The Day 8 fork output has no wrapped or lifted plate-connected spaghetti.
- Failed or partial method outputs are never promoted to primary targets.

## Frozen next question

Train a small high-noise VACE-LoRA on the two pseudo-targets, then apply it to
the unchanged Day 8 pasta/fork 3-D control with the same seed and inference
settings. The only meaningful positive outcome is improved fork-food binding
without degrading fork identity or protected pixels. A two-sample pilot is not
evidence of a universal cooking editor, even if the held-out image improves.

Machine-readable audit: `day9_action_target_audit_v1.json`.
