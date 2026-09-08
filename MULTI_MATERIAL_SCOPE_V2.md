# FoodStateEdit: multi-material pilot scope v2

Scope ID: `foodstateedit-multimaterial-pilot-v2-20260908`.
Requested by the user on 2026-09-08: cover noodles, soup, fried rice and cake,
not just increasingly tuned noodle examples.

This is an exploratory scope/design addendum, not a frozen executable run or
an assertion of multi-family effectiveness. It does not overwrite the v1
scope, its 60-image split, historical pasta family, or any negative gate.
The new user-requested study is not a success-triggered expansion of Day 13/18.

## Four required demonstration classes

| Material | First action | Required material behavior | Current evidence |
| --- | --- | --- | --- |
| Strand: ramen/udon | Two chopsticks lift a connected strand | Continuous visible strand, pinch, plausible occlusion and bowl attachment | Day 18 support repair improved utensil retention on one seen synthetic image; strict action/photo gates did not pass |
| Liquid: soup | Spoon scoops and lifts liquid | Liquid contained by spoon, plausible surface and meniscus; no rigid source crater | Historical controlled spoon action and fill results exist; broad real-image performance not established |
| Granular: fried rice | Spoon/spatula lifts a rice payload | Distinct grains, utensil support, coherent ingredients and corresponding source reduction | State solver, real pilot anchor and Day 3 render baseline records exist; quality/effectiveness not established |
| Cohesive soft solid: cake | Fork lifts one already pre-cut bite | Block remains attached to fork; plausible exposed crumb and matching source gap; no duplicated bite | New class: no frozen cake source, mask, geometry, adapter or evaluated output found in current experiment assets |

For cake, start with an already pre-cut bite in the source. Cutting an intact
cake, fracture, cream deformation and crumbs are separate later tasks. This
assumption must be visible in source selection and report captions.

## Shared method, material-specific constraints

Working description: **Food manipulation image editing with 3D action control**.
The research target is a shared interface, not a claim that all foods deform
like noodles.

Common inputs: source image, action/utensil specification, material label,
food/container masks and contact/motion anchors. Common outputs: an edited
image plus a temporal control-following diagnostic and auditable provenance.

Shared stages: relative/reconstructed 3D action -> material-specific state and
contact update -> depth-aware 2D control and full swept support -> existing
VACE appearance completion -> protected-region compositing -> evaluation.

- Strand preserves continuity except modeled occlusion.
- Liquid preserves modeled transferred volume and container support, not a
  strand skeleton or persistent source hole.
- Granular preserves modeled component accounting and payload support; local
  grain rearrangement is allowed.
- Cake permits a deliberate connectivity change at a pre-cut boundary; a
  noodle-style global connectedness loss would be wrong for this action.

State conservation is an internal solver check, not proof of physical mass or
volume conservation inferred from an RGB image. Relative 3D is not measured
scene geometry. SAM3 is not yet a validated universal observer across these
classes, and no observer is automatically installed as diffusion guidance.

## Next experiment sequence

1. Audit/recover the existing soup and rice pilot evidence before spending
   another inference. Preserve all failed or weak outputs.
2. Select one pilot source for each of the four required classes; record source
   license/provenance and whether it is real, synthetic, previously used or new.
   Use existing pilot-only noodle/soup/rice assets where appropriate. Cake needs
   a newly audited source; do not repurpose a frozen test image.
3. Build/review material-appropriate controls and full swept masks for all four
   before concentrating further tuning on noodles. Check source change and
   destination contact together. No GPU run is unlocked by this document.
4. Freeze a two-arm pilot: simple 2D action control + VACE versus material-aware
   3D control + VACE. Use a no-edit reference as a CPU sanity check. Across the
   two arms within each image, hold prompts, source, model, seed, steps, output
   size, editing support and final-compositing policy fixed. Different controls
   are the intended intervention. If other factors cannot be matched, label
   the comparison confounded rather than claiming a controlled ablation.
5. Initial proposed budget: one image/class, seed 1, 21 frames, 20 steps,
   VACE scale 1, LoRA off, TTM off, two inference arms: **eight new inferences**
   if none can be reused byte-for-byte under the new freeze. This is a proposed
   budget, not a launch instruction or a statistical effectiveness study.
6. Review every class under the same action/photo/preservation rubric, with
   material-specific checks. Publish a four-class panel including failures.
   Existing historical images are context, not retroactively matched baselines.
7. Only after feasibility review, predeclare additional images, seeds and a
   held-out evaluation under a separate execution freeze. Do not select the
   best seed or discard a difficult class after seeing results.

Before execution: freeze sources, masks, control builders, runtime/config
hashes, conditions, exact thresholds and output paths. Reaudit gp38--gp42;
use only a qualifying idle RTX A6000 (>=48,000 MiB free, <=5% utilization,
zero compute processes, >=80,000 MiB host memory). Remain offline for models,
use serial inference, preserve failures, and never preempt other users.
Automatic heartbeat execution remains paused.

## Reporting contract

- Report four separate material rows and sample counts, not just a pooled
  score that lets noodle performance hide cake/rice failures.
- Action: correct utensil/payload, support/contact, source update and no
  duplication, plus the class-specific requirements above.
- Photo: texture, boundary, lighting and contact shadow, independently of action.
- Preservation: exact pre-encoding support-exterior pixels; separately report
  native output changes and lossy-video encoding errors.
- A single good picture per class is a breadth demonstration, not proof of
  generalization, a universal editor or a novel diffusion algorithm.
- Primary contribution remains a hypothesis: shared material-aware 3D action
  control may improve food manipulation editing. It requires matched evidence
  across classes, not just model reuse or assembling four attractive pictures.

Evidence entry points: `results/DAY18_HIGH_LIFT_SUPPORT_REPAIR_20260908.md`,
`results/CURRENT_RESULTS_INVENTORY.md`, `benchmark/anchor_manifest_v1.csv`,
`results/day3_unified_geoedit_smoke_v1/runs/fried_rice_spatula_001.json`.
