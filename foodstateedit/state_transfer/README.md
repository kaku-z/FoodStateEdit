# Source-consistent state transfer — implementation v0.1

Status: **experimental CPU-tested components, not a trained editing system**.
No existing checkpoint, frozen runtime, experiment, or GPU process is modified.
This package does not yet generate photographs or claim cross-material efficacy.

Design provenance: academic-research-suite, implementation of the approved
state-transfer design, 2026-09-19. Verification scope: software contracts only;
food-generation effectiveness and generalization are UNVERIFIED.

## Implemented contracts

| File | Responsibility |
| --- | --- |
| `geometry.py` | source-labelled observed parcels; joint updates; explicit ownership/contact; point depth-buffer projection |
| `bridge.py` | THWC controls to t/y/x DiT tokens with explicit temporal selection and inactive reference prefix |
| `routing.py` | trainable zero-initialized state residual + sparse source-memory attention; optional PyTorch dependency |
| `../../scripts/demo_state_transfer.py` (repository `scripts/`) | CPU synthetic 21-frame data-flow demo with hash manifest; no image generation |
| repository `tests/test_state_transfer.py` | invariants, failures, visibility, sparse lookup, gradients and toy optimization |

## Coordinates and shapes

The camera is fixed. x points right, y down, z forward. Positions/depth share
camera-relative units; displacements are incremental camera-unit vectors, not
pixels. A pinhole `Camera` maps points to image pixel centres.

`FoodState` contains N unique source IDs, N source pixel coordinates, original
and current `[N,3]` positions, ownership (`remain/carried/detached`), confidence,
contact and `[E,2]` observed 4-neighbour graph edges. The edges are not a learned
material model. Removing an edge is caller-supplied, not simulated fracture.

`Transition` updates ALL N parcels together. It cannot append a duplicate or
delete an occluded parcel. This is a representation invariant, not image-level
mass conservation. No physical mass is estimated from pixel counts.

Projection outputs `[H,W,10]`:

0..3: keep / move / rearrange / expose, exclusive in the prototype;
4..5: original source UV in [-1,1], align_corners=True; (2,2) is invalid;
6: positive relative depth, 0 for unknown revealed geometry;
7: contact probability;
8: source/geometry confidence;
9: utensil occupancy.

Projection uses the nearest pixel per parcel and a depth buffer. It does NOT
rasterize surfaces, infer interiors, or enforce non-penetration. Uncovered
original food pixels are UNKNOWN exposure with confidence zero: never assumed
to be a plate, empty volume, or permanent liquid cavity. A supplied neighbouring
parcel may occupy the location; a supplied foreground utensil may occlude it.
Hidden and out-of-view parcels remain in the state. Equal-depth ties prefer
static geometry, then the utensil; food ties use source ID order.

## Minimal use

```python
state0 = from_mask(food_mask, positive_reference_depth, camera)
state1 = apply_transition(state0, Transition(delta_xyz, owner, contact, confidence), camera)
frame = project(state1, camera, positive_reference_depth)
condition_video = np.stack([frame0, frame1, ...])  # [T,H,W,10]
controls, active = pack_controls(condition_video, explicit_frame_indices,
                                 token_grid_hw, prefix_tokens=actual_prefix_count)

# In a NEW, opt-in model integration, after a selected DiT block:
adapter = StateTransferAdapter(token_dim, reference_channels).to(device=device, dtype=dtype)
hidden, diagnostics = adapter(hidden, clean_reference_features,
                              torch.as_tensor(controls, device=device),
                              torch.as_tensor(active, device=device))
```

`clean_reference_features` is `[B,C,Hs,Ws]` aligned to the original image.
It must come from a consistent frozen image/VAE feature producer; a pipeline
integration still has to select and verify that producer. The adapter learns
query/key/value projections; it does not assume raw VAE and DiT features match.

Exact correspondence reads only four neighbouring memory cells with bilinear
position priors. Source readout is disabled for exposure, invalid correspondence,
zero confidence, and utensils. The separate state residual may still act on
exposure; no source texture is read for that region. Both output projections
are initialized to zero, so an untrained branch is exactly identity. This is a
safe starting point, not a useful untrained editor.

## Wan integration boundary — NOT yet connected

The observed repository entry is
`vendor_overrides/GeoEdit/diffsynth/pipelines/wan_video.py::model_fn_wan_video`.
Future integration must explicitly:

1. Load trained adapter weights with schema/version and source-model hashes.
2. Cache clean reference features and compute per-frame state conditions.
3. Align to actual DiT patch sizes and temporal receptive fields. `pack_controls`
   only does explicit nearest-cell sampling; it is a prototype, not exact VAE
   temporal alignment. Never average source UVs across visibility boundaries.
4. Pass controls through CFG duplication, reference-prefix padding, temporal
   tiling and both high/low-noise model branches. Length mismatch must fail.
5. Attach a residual call after explicitly selected blocks. Keep it disabled by
   default. Frozen backbone weights still need an autograd graph when training
   adapters inside the backbone. The public no_grad inference call is NOT the
   training entry point. Memory/offload feasibility has not been measured.
6. Audit original reference injection paths. This extra branch cannot prevent
   the pretrained model from reconstructing the old surface through another
   reference path. Do not claim source-exclusion guarantees for output images.

We intentionally do not patch the frozen runner or claim a working GPU hook.
No checkpoint with demonstrated food-editing effectiveness is supplied.

## Missing components and staged acceptance

**Stage 1 (this implementation):** supplied/oracle states -> projection ->
aligned token controls -> trainable residual. Test identity, unique IDs,
occlusions, unknown exposure, partial source refill, invalid inputs, gradients.

**Stage 2:** real image-aligned reference encoder + opt-in Wan residual hook +
training entry with a same-condition VACE baseline. Test known-correct control
states before training a planner. Report generated image effects separately
from this package's deterministic invariants.

**Stage 3:** learned material-conditioned joint state predictor, contact and
connection dynamics, surface/volume visibility, uncertainty calibration.
Requires training data; no material-specific image-colour/ROI hacks are used
as a substitute. Soup refill, noodle topology, cutting and granular settling
are NOT implemented by the current oracle transition interface.

**Stage 4:** real held-out cross-material testing: complete method, no source
routing, and shuffled correspondence with geometry/prompt/seed held fixed.
Inspect sources independently of the planner. Report image-level duplication,
source correctness (unknown when unidentifiable), remainder naturalness,
photographic quality and preservation. Apply identical compositing to baselines
and preserve raw outputs; exact background copying is not learned improvement.

## Local validation

From repository root, using Python with numpy and torch:

```text
python -m unittest discover -s tests -p test_state_transfer.py -v
python scripts/demo_state_transfer.py --output-root <NEW_DIRECTORY>
```

The demo fails if the target exists. It saves controls, positions, visible IDs
and SHA-256 evidence. It uses synthetic prescribed movements, not real food
simulation, inferred physics, trained generation, or a benchmark result.

Validated locally on 2026-09-19 with Python 3.13: **19 new tests pass; full
unittest discovery passes 287 tests**. The toy optimizer test exercises only
backpropagation on synthetic tensors; it does not produce a trained checkpoint.
The generated demo is `artifacts/scfst_contract_demo_v1_20260919/controls.npz`
with a sibling `manifest.json` (21 frames, shape `[21,48,64,10]`).
