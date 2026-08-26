# Scope freeze: FoodStateEdit paper sprint

Freeze ID: `foodstateedit-scope-v1-20260826`  
Freeze date: 2026-08-26 (Asia/Tokyo)

## Working title

**FoodStateEdit: Training-Free Physically-Constrained Food Manipulation Editing
with Material-Aware Staged Projection**

## Frozen task interface

Required inputs:

1. a source food image without the target utensil manipulation;
2. a structured action type and utensil type;
3. target contact anchors, pose/direction, or an equivalent sparse control;
4. food/container masks generated automatically or explicitly corrected by the user.

Required output:

- a 2D image showing the requested manipulation;
- a complete case/run/metric manifest;
- exact source pixels outside the declared edit alpha;
- independent automatic and manual acceptance decisions.

## Frozen primary hypothesis

Rigid utensils, contact regions, deformable food payloads, and vacated source
regions require different constraint lifetimes during denoising. A structured
food-state solver plus material-aware staged projection should improve joint
action correctness and physical plausibility over a unified-mask GeoEdit
baseline without sacrificing protected-scene fidelity.

## Frozen contribution candidates

1. A shared food-manipulation state/action representation spanning rigid,
   liquid, granular, and strand materials.
2. A layered proxy with rigid, contact, material, source-hole, and protection
   regions, projected through material-aware denoising schedules.
3. FoodManipBench-P: a constraint-driven evaluation protocol with material
   metrics, exact preservation, blind physical-realism questions, and failures.

## In-scope families

| Family | Dish | Utensil | Action |
| --- | --- | --- | --- |
| liquid | soup | spoon | scoop liquid |
| granular | fried rice | spoon/spatula | scoop payload |
| strand | ramen/udon | chopsticks | lift strand |
| strand-contact | pasta | fork | twirl and lift strand |

## Explicit non-claims

- not text-only;
- not fully automatic while masks or anchors require correction;
- not a universal cooking editor;
- not trained end-to-end;
- not universally photorealistic;
- no main-venue-level generalization claim from the current pilots;
- no proxy similarity metric is treated as physical ground truth.

## Success definitions

`ActionSuccess` requires correct utensil count/type, valid payload, valid
contact/support, correct source reduction/removal, and no forbidden duplicate.

`PhotoSuccess` requires local material realism, plausible reflection/light/shadow,
an invisible source-hole repair, and no obvious diffusion artifact.

`PreservationSuccess` requires a zero maximum pixel difference outside the
declared edit alpha and an accepted feather-band transition.

`StrictE2ESuccess = ActionSuccess ∧ PhotoSuccess ∧ PreservationSuccess`.

State-only success must never be reported as end-to-end image-edit success.

## Benchmark target and frozen split policy

- target: 60 real inputs, 15 per family;
- per family: 5 pilot and 10 frozen test inputs;
- schedules are selected on pilot only;
- test cases are not removed after observing failures;
- stochastic formal methods use three predeclared seeds;
- generated and real inputs are reported separately.

## Sprint gates

- Day 3: at least three baselines batch successfully;
- Day 8: at least three of four anchors pass ActionSuccess and two pass PhotoSuccess;
- Day 14: the full method improves ActionSuccess in at least three families and
  has stable positive StrictE2E gain over the strongest reproducible baseline.

If a gate fails, the claims and scope must be reduced immediately. Negative or
failed cases remain in the benchmark.

## Change control

Any change to task inputs, success definitions, material families, split policy,
formal seeds, or primary metrics requires a new freeze ID and a short rationale
in `CHANGELOG.md`. Case-specific source-code changes after the test split freezes
are prohibited.
