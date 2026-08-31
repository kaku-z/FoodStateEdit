# Day 8 structured 3-D motion projection checklist

## Method boundary

- [x] Make 3-D motion and camera projection the explicit method core.
- [x] Distinguish reconstructed 3-D from relative 3-D in every manifest.
- [x] Keep VACE/LoRA outside the geometry solver.
- [x] Do not claim photo realism from a procedural control image.

## Fork pilot

- [x] Build one pinhole camera shared by rigid and deformable geometry.
- [x] Lift the final fork support into relative 3-D and animate approach/contact/lift.
- [x] Construct four 3-D spaghetti helices and four plate-connected tails.
- [x] Require every helix to cross the fork depth for front/back contact.
- [x] Project all geometry into a deterministic 21-frame 2-D VACE control.
- [x] Preserve frame zero exactly and source pixels outside motion support.
- [x] Save 3-D arrays, camera intrinsics, video, masks, hashes, and a review board.

## Evidence level

The fork pilot is `relative_3d_geometry_pilot_not_reconstructed_scene_depth`.
It uses the frozen sparse anchor and a normalized pinhole camera because the
previously audited VGGT checkpoint is no longer present locally or on gp39--40.
It must not be described as recovered metric scene geometry. The ramen case
remains the only preserved `VGGT -> reconstructed 3-D -> projection` example.

## Next gate

- [x] Freeze one same-seed VACE rendering configuration and fail-closed GPU
  preflight before seeing stochastic output.
- [x] Upload and hash-verify the frozen inputs/launchers on gp39; remote
  preflight passed 35/36 checks and failed only the occupied-GPU gate.
- [x] Run the frozen VACE rendering comparison on a safely idle gp39 A6000.
- [x] Verify one pipeline load, 21 decoded frames, all hashes, and exact pixels
  outside the motion support.
- [x] Compare action/contact and photo realism separately against the Day 6 2-D
  dynamic-multikey control without changing seed or frame selection.
- [x] Close the single-anchor render gate without seed expansion: rigid fork
  topology improved, but the wrapped/lifted spaghetti payload is absent and
  both strict action and photo verdicts remain false.
- [ ] Add recovered depth for soup, rice, and fork only if an existing audited
  depth model becomes available; do not download a replacement silently.

## Next eligible renderer gate

- [ ] Build non-identity supervision targets that explicitly contain
  utensil-food binding and thin lifted strands.
- [ ] Train/evaluate one claim-limited VACE adapter against the frozen Day 8
  control before adding geometry detail, seeds, or held-out cases.
