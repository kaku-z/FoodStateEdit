# Day 5 qualitative method grid

Artifact: `artifacts/day5_paper_figure_v1/day5_method_grid.png`

SHA-256: `483b88b14b8dbd2a1df7635f75b28fdcf66c09a6ce00dc6d3257ca6b88166cef`

## Draft paper caption

**Topology preservation and appearance remain distinct failure axes.** Rows show
the four pilot action families; columns show the unedited source, the shared
explicit geometry proxy, unified-mask GeoEdit (seed 1), staged GeoEdit with
exclusive semantic masks (seed 1), deterministic geometry locking, and the
topology-preserving 2.5D material renderer. All edited outputs preserve source
pixels exactly outside the same `edit_alpha` support. The two stochastic
GeoEdit variants soften procedural texture but lose or break utensil identity,
payload, or contact topology. Geometry locking retains the intended relations;
material rendering further improves rigid metal/wood appearance, but warped
food texture and fine utensil-food occlusion remain visibly non-photographic.

## Usage note

This is a diagnostic pilot figure, not a final user-study result. The action and
photo verdicts are single internal, non-blind reviews. Keep the source hashes in
`results/day5_paper_figure_v1/figure_manifest.json` with any submitted version.
Add the independently frozen VACE-direct column only after its four-anchor batch
and visual review are complete.
