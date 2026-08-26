# Anchor annotation protocol v1

The four primary anchors are real pilot cases selected before method output is
observed. Their normalized vector controls are frozen in
`anchor_specs_v1.json`; deterministic dense layers and case manifests are in
`anchors_v1/`.

## Five semantic layers

1. `rigid`: the target utensil geometry. Utensil count, pose, and separation
   are hard constraints and must not be delegated to the appearance model.
2. `contact`: the local support/grip interface shared by utensil and payload.
3. `material`: liquid, granular payload, or deformable strands at the target
   state.
4. `hole`: the source footprint that becomes visible after material moves and
   therefore requires material-aware repair.
5. `protect`: all pixels outside the nonzero feathered edit support. These
   pixels must be copied exactly from the canonical input.

`edit_alpha.png` is a derived feathered union used for compositing; it is not a
sixth physical layer. Colors in the ignored review overlays are red `rigid`,
yellow `contact`, green `material`, and blue `hole`.

## Frozen anchors

| Anchor | Source case | Action | Key hard constraint |
| --- | --- | --- | --- |
| `soup_spoon_001` | `soup_002` | spoon scoops potage | one spoon, contained liquid, self-leveled source surface |
| `fried_rice_spatula_001` | `rice_007` | spatula lifts rice | supported granular payload and one source reduction |
| `ramen_chopsticks_001` | `noodle_001` | chopsticks lift udon | two separated chopsticks, valid grip, continuous length-conserved strand |
| `pasta_fork_001` | `pasta_006` | fork twirls spaghetti | one four-tine fork, wrapped contact, continuous strand to plate |

The noodle vector centerline check gives a target/source length ratio of
`1.005146`, within the frozen `[0.95, 1.05]` tolerance. Dense-mask audits also
require nonempty rigid/contact/material/hole/protect layers and nonzero contact
intersection with both rigid and material masks.

## Change control

The anchors may be tuned only on the pilot split. Any spatial change after a
formal result is inspected requires a new annotation version and rationale.
Test inputs never receive case-specific action geometry derived from their
outputs.
