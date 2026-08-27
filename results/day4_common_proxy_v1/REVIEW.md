# Common proxy v1 review

Review date: 2026-08-27 (Asia/Tokyo)

## Decision

`common_proxy_v1` passes the internal smoke-inference gate for all four frozen
anchors. This is a geometry/control proxy, not a claimed final edited image.
Its deliberately synthetic utensil shading and warped source texture are inputs
that the same generation backend must refine.

## Checks

- 4/4 anchors exist at aspect-preserving long-side 736 resolution.
- Width and height are divisible by 16.
- The declared relative layer depth does not depend on a learned depth model.
- Maximum pixel difference outside `edit_alpha` is 0 for every anchor.
- Soup: one spoon, contained liquid, and a separate surface-repair support.
- Fried rice: one spatula, supported granular payload, and a source-reduction support.
- Udon: exactly two separated chopsticks, grip contact, a continuous lifted strand,
  and explicit rear-strand-front ordering.
- Pasta: one four-tine fork, twirl contact, returning strands, and source repair.

The source-embedded review boards remain under the ignored `artifacts/` tree and
must not be committed or redistributed. Machine-readable manifests are retained
under this directory.
