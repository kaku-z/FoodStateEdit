# FoodStateEdit blind evaluation protocol v1

## Scope

This protocol evaluates fixed final images and six-frame process contact sheets
without revealing method names. It separates action correctness, photographic
appearance and protected-scene preservation. Reviewers must not infer a pass
from visual attractiveness alone.

## Blinding package

Each review item receives a random opaque ID. The package contains only:

- the source image;
- the requested utensil and action in neutral text;
- one fixed final output, or one fixed six-frame contact sheet;
- no method name, seed, scale, filename, metadata or preferred answer.

The mapping from opaque ID to method, case and seed is held separately until
all ballots are locked. Output order is randomized independently for every
reviewer. Reviewers may use `uncertain`; uncertainty is not a pass.

## Final-image questions

Answer `yes`, `no` or `uncertain` unless a scale is specified.

1. Is the required utensil type recognizable?
2. Is the required utensil count correct?
3. Is a food payload visibly present in the requested manipulated state?
4. Is the payload visibly supported, gripped or contained by the utensil?
5. Is the food source region changed consistently with the removed payload?
6. Is there no duplicated payload or impossible floating food?
7. Is local material appearance photographic rather than graphic/proxy-like?
8. Are local lighting, reflection, occlusion and shadow plausible?
9. Is the source-region repair natural and free of an obvious hole/artifact?
10. Overall photo realism: integer 1 (clearly synthetic) to 5 (photographic).

`ActionSuccess` requires yes on questions 1-6. `PhotoSuccess` requires yes on
questions 7-9 and a photo-realism rating of at least 4. `StrictE2ESuccess`
additionally requires the machine-verified preservation gate. Any `uncertain`
component prevents the corresponding composite pass.

## Process questions

For a six-frame contact sheet, answer `yes`, `no` or `uncertain`:

1. Frame 0 represents the unmanipulated source state.
2. The utensil approaches before contact.
3. Payload motion starts only after visible contact/support.
4. Contact/support persists during lift.
5. The final hold is stable and preserves utensil identity.
6. The source change is temporally consistent with the lifted payload.

`TemporalActionSuccess` requires yes on all six questions.

## Reviewer eligibility and independence

- Minimum two reviewers; three preferred.
- Reviewers must not have produced the outputs or seen the method mapping.
- Record relevant vision/graphics expertise, but do not collect unnecessary
  personal data.
- Resolve institutional requirements before treating evaluations as a public
  human-subject study. This protocol is not ethics approval.

## Adjudication and statistics

- Preserve every raw ballot before resolving disagreement.
- Report per-question counts, composite counts and family-stratified results.
- Report reviewer agreement; never silently majority-vote away uncertainty.
- Primary method comparisons are paired by source image and clustered by image
  across seeds.
- The method mapping is revealed only after ballot locking and checksum.

## Prohibitions

- No output-dependent frame selection per method.
- No replacing failed seeds.
- No showing only successful cases.
- No changing the rubric after test outputs are visible.
- No treating exact final compositing as autonomous generator preservation.
- No treating scaffold-derived masks as independent ground truth.
