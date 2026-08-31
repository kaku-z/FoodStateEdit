# Day 8 fork relative-3-D VACE comparison review

## Outcome

The frozen same-seed run completed technically on gp39 GPU 0, but it did not
solve the requested fork-twirling edit. The relative-3-D control improved rigid
utensil identity and trajectory over the Day 6 planar control. It did not cause
VACE to render wrapped or lifted spaghetti.

This is a positive representation diagnostic and a strict action/photo
failure. It is not eligible for more seeds or the frozen test set.

## Frozen execution

- Anchor: `pasta_fork_001`; seed: `1`; frames: `21`; selected frame: `18`.
- Inference: 20 steps, VACE scale `1.0`, TTM disabled.
- Host/GPU: gp39, physical A6000 GPU 0.
- Passing preflight: 36/36 checks; SHA-256
  `ff4597e26707d64f32f18e243e6c5a995f4b2e68d422b030d380ca6585b20e10`.
- Pipeline loads: `1`; decoded frames: `21`; wall time: `958.789 s`.
- Generated video SHA-256:
  `dbac7be2a7ffb4c1cb996f960e04f799542f1875e7fd16d231155af863b045c4`.
- Selected raw frame SHA-256:
  `a0c3091e8e70935400fb6cb190b55f38deb3f45d1cbac50c4a0753408522df1f`.
- Exact-projection frame SHA-256:
  `727071883fc3727d3597b67e3e9ba55da3de8f21428a3bc87daee4491b476bf8`.
- Maximum difference outside motion support: `0`.

All pulled output hashes match the remote run manifest. The video remains
outside Git; its hash is retained in the tracked evidence manifest.

## Action/contact review

The raw-frame comparison is primary because the Day 6 and Day 8 final
projection supports differ. The projected frames confirm the preservation
contract but are not used to attribute the representation improvement.

Day 6 renders a single utensil whose head collapses into a broad spoon-like
metal blob with small fused prongs. Day 8 renders one much clearer fork and a
coherent approach/lift trajectory across frames 0--20. This supports the narrow
claim that depth-aware 3-D conditioning improves rigid utensil topology.

Day 8 still fails the defining food-action criteria:

- no multiple spaghetti strands visibly wrap the tines;
- no lifted strand remains connected to the plate;
- no source spaghetti is visibly reduced into a lifted bundle; and
- the contact area becomes a translucent local smear rather than bound food.

Strict `ActionSuccess` is therefore false for both controls.

## Photo-realism review

The unedited dish remains photographic and exact source protection passes.
The Day 8 fork is more recognizable than the Day 6 metal blob, but its tine
attachment, contact occlusion, local halo, and shadow are not physically
coherent. Most importantly, the requested food payload is absent. Strict
`PhotoSuccess` is false for both controls.

## Interpretation and next gate

The 3-D solver has done its job: the control contains fork motion, helical
strands, depth crossings, and plate connections. The learned renderer preserves
the coarse rigid utensil signal but discards the thin deformable payload. That
isolates the present bottleneck to action-conditioned appearance rendering and
utensil-food binding, not the absence of a 3-D motion representation.

The next eligible experiment is a non-identity VACE adapter with explicit
action targets containing fork/strand binding. More hand-designed geometry or
more stochastic seeds is not justified until that supervision exists.
