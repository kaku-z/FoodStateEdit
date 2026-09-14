# Day 31 family-template control pilot

Date: 2026-09-13

## Outcome

The deterministic control builder completed all 20 **pilot** cases: five each
for liquid, granular, strand and strand-contact food.  Every case contains one
21-frame planar control and one 21-frame relative-3D control, plus final frames,
review sheets, edit support and geometry traces.  No held-out test image was
opened for annotation or used for generation.

The remote manifest SHA-256 is
`99a8b4de01edc0d76083c8fd8c9e7d5102ce3fd7f2de41af2ab01ede933cd977`.
All 180 manifest-recorded files (88,384,357 bytes) were copied to
`artifacts/day31_family_template_controls_pilot_v1` and rehashed locally with
zero SHA-256 or size mismatch.

## What the pilot establishes

- one fixed family-level rule can place a utensil, contact region and large
  lift trajectory without per-image parameter tuning;
- planar and relative-3D variants use the same source, timing and support;
- all changed control pixels lie inside the recorded edit support, so protected
  pixels remain exact;
- the builder is frozen provisionally at SHA-256
  `c61b1342ab6479c950686e39623acdba54f41c412fc08d4de670203961fb741d`.

## Internal visual review

The control geometry and lift direction are readable across the four families.
The controls are intentionally schematic rather than photographic.  The pasta
fork is particularly line-like, and solid-food payloads are simple sampled-color
shapes.  They are therefore admitted only as VACE intervention inputs, not as
final images, targets, annotations, or evidence that a food interaction is
physically correct.

The four family grids are stored under
`artifacts/day31_family_template_controls_pilot_review_v1`.  Their SHA-256
values are recorded in `day31_family_template_controls_pilot_v1.json`.

## Decision

Keep the 40-image test split locked.  Test control generation requires the
Day 25 rice/cake workers to finish, a review of all four development cases, and
a frozen material-level VACE-scale rule.  Only subsequent model outputs—not
these proxy images—may be evaluated as generated results.

## Claim boundary

This pilot proves deterministic control construction and evidence integrity.
It does **not** establish VACE success, relative-3D superiority, photo realism,
physical correctness, or held-out generalization.  The visual judgment above is
internal and unblinded; it is not an independent human evaluation.
