# E5: shared bite/remain consistency pilot

## Question

Can one shared manually parameterized removal volume make the lifted cake bite
and the remaining notch correspond, while a local VACE pass turns the hidden
cut walls and plate floor into a natural photograph?

This is a synthetic development diagnostic, not held-out evidence. The shared
volume is a screen-space projection of the existing manual cuboid, not measured
3D truth, and the experiment cannot establish physical mass conservation.

## Implementation

- One cuboid defines both the moving payload projection and the three newly
  revealed cavity surfaces (two cake walls and the plate floor).
- Real cake pixels from the original visible crumb face seed the two cut-wall
  conditions. Neutral plate pixels fit a smooth affine plate field for the
  floor. A small local shadow is added only at wall/floor contact.
- The repair support grows with the recorded lift amount and is clipped to the
  prior local compositing footprint. The moving payload is excluded.
- All pixels outside the inward-feathered alpha and all pixels in the planned
  payload mask remain exactly equal to the E3 baseline.
- Three unit tests cover reveal timing, payload exclusion, semantic cavity
  surfaces and exact protected-pixel compositing.

## Runs

Both gp40 GPU 0 runs passed a fresh RTX A6000 preflight (48,539 MiB free,
0% utilization, zero compute processes and more than 251,000 MiB available host
memory), loaded the pipeline once and completed 21 frames at 20 steps, seed 1,
VACE scale 1, LoRA off and TTM off. No retry or parallel GPU use occurred.

### E5: original reference

- Controls manifest: `de6e7e68b6e87cd617be27e0353e31ddb35c21db77dbfe37ea0c890474682061`
- Runner: `55ff392807748957b7cd0868299a82cc80f8c1fafdefd5a04dea855cb71ffa29`
- Wall time: 238.94 s
- Result: the original reference still contains the bite at its source
  location. The model regenerates a block despite the empty-cavity local
  condition. The desired state is not achieved.

### E5b: cavity-state reference

- Controls manifest: `14f007d57c42289c428600991b460add04a6c73552c3a774821a061a3ea024b2`
- Runner: `97e7c8ce3c45939c5dce1e3d8cf766c1fd8fd931971aaf52775db2f5a0f0f5b8`
- Wall time: 239.88 s
- Controlled change: the reference and its prompt description use the
  shared-volume empty-cavity anchor. Geometry, masks, seed and inference budget
  are otherwise unchanged.
- Result: the source location is visibly empty and the plate floor is exposed.
  This supports reference-state conflict as one cause of E5 failure. However,
  the walls are excessively planar/triangular, their texture is not yet a
  natural cake fracture, and the lifted payload partly occludes the notch in
  image space. It is not a clear photographic success.

For both runs, 48 manifest records were rehashed locally with zero mismatch;
the result-manifest, run-manifest and preflight hashes also match the remote
files. The maximum protected-pixel difference is zero.

## Decision

Status: `partial_mechanism_support_visual_quality_failed`.

The experiment supports one algorithmic requirement: the reference image must
represent the same material state as the desired control. It does **not** yet
support the claim that shared 3D control produces natural remaining food.

The next implementation should add a visibility constraint to the action
planner so that the payload leaves the source projection by a margin before the
hold frame, and replace flat planar wall warps with a material boundary model
(layer-aware cake cross section plus small crumb-edge displacement shared by
the payload and notch). This requires regenerating the first-pass trajectory;
another source-only completion pass cannot correct a payload that still
occludes the cavity.

## Evidence

- Four-way crop: `artifacts/e5_bite_remain_review_v1_20260917/four_way_source_crop.png`
- Review validation: `artifacts/e5_bite_remain_review_v1_20260917/validation.json`
- E5 local result: `artifacts/e5_bite_remain_gp40_20260917T062853Z`
- E5b local result: `artifacts/e5b_bite_remain_gp40_20260917T063729Z`
- E5 remote result: `/host/space0/guo-z/tf-ufi/outputs/e5_bite_remain_consistency_v1_20260917T062853Z`
- E5b remote result: `/host/space0/guo-z/tf-ufi/outputs/e5b_bite_remain_anchor_reference_v1_20260917T063729Z`
