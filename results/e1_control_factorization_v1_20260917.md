# E1 control-factorization diagnostic (2026-09-17)

## Outcome

The frozen two-case experiment completed on gp40 without retries or resource-policy violations. Both cases used one pipeline load, seed 2, 21 frames, 20 steps, VACE scale 1.0, LoRA off, and TTM off. All six conditions completed, all expected frames exist, and no failure file was produced.

This is a failure-localization diagnostic, not an effectiveness result. The evidence supports a mixed conclusion: separating structure from appearance changes the failure mode, but a sparse RGB scribble is still treated partly as an appearance teacher by VACE. Adding a frozen payload crop did not show a clear semantic benefit.

## Conditions

| Arm | Control | References |
| --- | --- | --- |
| `appearance_laden_rgb` | Existing lossless relative-3D RGB proxy | Full input |
| `structure_scribble` | Binary geometry scribble | Full input |
| `structure_scribble_payload_reference` | Same geometry scribble | Full input + frozen payload crop |

All other inference and compositing settings were held fixed.

## Visual findings

### Cake

- `appearance_laden_rgb`: contact and lift occur, but the bite breaks into implausible residual pieces and contains source/payload artifacts.
- `structure_scribble`: the source gap and lifted bite are more coherent, but the fork becomes conspicuously outline-like/doubled, consistent with control-style leakage.
- `structure_scribble_payload_reference`: cake texture remains plausible, but the fork/contact artifact remains; there is no clear overall gain over the structure-only arm.

### Noodle

- `appearance_laden_rgb`: chopsticks contact and lift a strand, but the local contact and strand shape are thin and awkward.
- `structure_scribble`: the approach-contact-lift order is readable and the lifted strand is clearer.
- `structure_scribble_payload_reference`: visually similar to the structure-only arm, without a clear additional material-identity benefit.

The review was performed by one labeled reviewer and is not a blind human study.

## Pixel diagnostics

| Case | Arm | Final outside max | Final outside MAE | Final inside MAE | Raw outside MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Cake | appearance-laden RGB | 0 | 0.000 | 25.655 | 5.058 |
| Cake | structure scribble | 0 | 0.000 | 28.262 | 4.609 |
| Cake | structure + payload reference | 0 | 0.000 | 30.806 | 3.718 |
| Noodle | appearance-laden RGB | 0 | 0.000 | 20.229 | 4.988 |
| Noodle | structure scribble | 0 | 0.000 | 31.504 | 5.153 |
| Noodle | structure + payload reference | 0 | 0.000 | 31.080 | 5.271 |

All final images are byte-exact outside the shared edit support because of the deterministic compositor. The raw VACE frames are not exact outside support, so preservation must not be attributed to VACE itself. Pixel-change values do not measure action success or photo quality.

## Decision

Do not scale this configuration or present it as a positive result. The next experiment should change how structure enters the generator—such as a representation or time-dependent conditioning schedule—rather than simply adding more RGB reference appearance. A matched small diagnostic should precede any multi-food benchmark.

## Evidence

- Local artifact root: `artifacts/e1_control_factorization_results_gp40_20260917T013752Z`
- Remote artifact root: `/host/space0/guo-z/tf-ufi/outputs/e1_control_factorization_v1_20260917T013752Z`
- Remote/local comparison: 168 files, all SHA-256 hashes identical
- Control manifest SHA-256: `d7f715252f25ccb193eb7199b34846d832053a878727f8354bfe5fe0f550e98e`
- Runner SHA-256: `89ff8bba28da4345f7250d2edd30a73b85b42253853b323d0677fa0d8e5df0d0`
- Cake run manifest SHA-256: `57cfc4fdeef9bccc6b86ec2220fba3be92d29510783578e0a61328574b751bce`
- Noodle run manifest SHA-256: `db26759b86238878eaea568a8b12b81b4053164de8919661736baca8ec51c75c`

Two cases and one seed cannot establish superiority, generalization, physical correctness, source conservation, or publication-level effectiveness.
