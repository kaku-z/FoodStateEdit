# Day 8 relative-3D fork projection pilot

## Question

Can the frozen pasta/fork anchor be represented as an actual 3-D rigid plus
deformable motion, including front/back depth crossings, and projected into a
deterministic 2-D temporal control without claiming unavailable scene depth?

## Result

Yes at the geometry-control level. A normalized pinhole camera backprojects the
fork support and sparse contact into relative 3-D. The fork follows one
approach/contact/lift trajectory. Four spaghetti curves form 2.25-turn helices
around the fork axis and remain connected to four fixed plate anchors. Each
helix crosses the fork depth, so rear and front segments are chosen by depth,
not by a fixed 2-D paint order.

- Frames: 21 at 15 fps; frozen selected frame: 18.
- Final fork reprojection maximum error: `1.2711e-13` pixels.
- Fork relative depth: `0.82`.
- Selected helix depth range: `[0.80128, 0.83872]`.
- 3-D strands/tails: `4`; helix samples per strand: `160`.
- Maximum pixel difference outside motion support: `0`.
- Motion-support fraction: `0.160321`.

## Reproducibility record

- Builder SHA-256: `1a3cf2af199d4adf473d5ebaf507a061ce5dbe86f3c9d2f3e5e00ab1a061f459`.
- Projection module SHA-256: `181727cb57c11a4253007268686ff8975695c7eafb683e160720cbb2b52176e0`.
- Frozen config SHA-256: `e737caa96490edee7e37babd5ee11af699c249a8e639235289e9fbe5f1be681a`.
- Run manifest SHA-256: `7a22f15e93f595193dd295f43812c4fe841eb442a6698e843e41d9206ffaab15`.
- Review board SHA-256: `61a0d57d44b39faa235b233415aa53eadfe394956869e8e1acd384c6ab77e03d`.
- Stored 3-D arrays SHA-256: `f73effbf63d89c0977a00e31c5ead8f278d231c177cd0dea1947a325edbdfbb0`.

Generated videos remain outside Git under the repository large-file policy;
their byte sizes and hashes are recorded in the tracked run manifest.

## Visual decision

The control clearly contains one fork, a lifted multi-strand bundle, plate
connections, and alternating front/back contact around the fork. It is a
procedural geometry proxy: strand colors, fork shading, sauce removal, and local
shadow are not photo-real. Therefore this is a geometry pass and a photo-result
non-evaluation, not `ActionSuccess` or `PhotoSuccess`.

## Claim boundary

The available VGGT checkpoint used by the historical ramen case is no longer
present locally or on gp39/gp40. This pilot therefore records
`relative_3d_normalized_pinhole`, not reconstructed monocular depth. No model was
downloaded. The next controlled experiment is one same-seed VACE render against
the Day 6 2-D dynamic proxy; only that output can test whether explicit 3-D
contact improves rendered action topology.
