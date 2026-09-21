# E6 regression diagnosis and E7 correction

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-19
- Verification Status: UNVERIFIED (GPU runs pending at protocol creation)
- Version Label: e7_cavity_fix_v2

## Observed defect and source audit

E6 preserved the desired empty region but also reproduced white triangular
patches inside the intended cake walls and an angular floor shadow. Reviewing
the actual cavity reference and renderer exposes the same defects before
diffusion. The old donor rectangle is `reference[250:350,115:285]`: 6,383 of
17,000 pixels have HSV saturation below 42 (37.55%). Inspection of the input
shows the rectangle crosses the sloping bottom boundary of the cake into the
plate. The low-saturation fraction is an appearance diagnostic, not a semantic
ground-truth segmentation score.

The old renderer perspective-warps the entire mixed-material rectangle onto
both hidden cut walls. It also multiplies a binary dilated wall/floor region by
0.90, explicitly seeding a hard shadow boundary. E6 then gives those reference
latents weight 1 on steps 0--7, propagating the rendered defect into generation.
The strength of this mechanism is subject to the controlled runs below.

An independent inherited problem is visible outside the cavity: the E3 output
already includes a grey swept-area polygon. E5/E6 preserve that *generated*
baseline, not the original photograph. Zero error relative to that baseline
does not certify correct background appearance. This local paired experiment
holds the compositor fixed; it does not claim to repair that separate defect.

## Frozen paired correction

- Controls: `artifacts/e7_cavity_fix_v2_20260919/controls`.
- Controls manifest SHA-256:
  `2b914c9d8691709a4c7bbcb09353be1025fe295781cb96a8cb9d9bdb57369662`.
- Source E5b manifest:
  `14f007d57c42289c428600991b460add04a6c73552c3a774821a061a3ea024b2`.
- Same prompt, negative prompt, seed 1, 21 frames, 20 steps, VACE scale 1,
  LoRA off, reference count, action geometry and final alpha as E5b/E6.
- No motion-planner change; the E6 hold-offset helper was not connected to
  inference and must not be cited as an evaluated improvement.

The corrected renderer selects the largest square entirely inside a declared
crumb mask (old ROI, HSV S>55 and V>65), rejecting insufficient donors. Here it
selects a 32x32 patch at x=251:283,y=295:327. Reflection tiling approximately
preserves crumb scale before perspective mapping. Floor contact shading uses
continuous distance falloff on visible floor. The material rule is specific to
this development case and is not a general food segmenter.

1. **E7 hard:** corrected donor/rendering; original E6 hard projection schedule.
   Versus E6 this tests the rendering correction package, not isolated shadow
   or donor contributions.
2. **E7 soft:** identical corrected input; projection weight
   `0.35*(1-k/8)^2` on steps k=0..7, then zero. A 3x3 spatially averaged mask is
   multiplied by the original support, keeping weight zero outside it. This
   changes projection policy only relative to E7 hard. No temporal smoothing
   or latent frequency filtering is used.

The runtime is copied into a new experiment directory; its default hard branch
is retained. Soft projection is enabled only by an explicit pipe attribute.
Each soft step records applied strength and support size. The original frozen
runtime and all past result directories remain unchanged.

## Checks and interpretation

Require resource preflight, one pipeline load per arm, 21 lossless frames,
byte-for-byte manifest verification, exact protected-pixel preservation,
and eight nonempty soft-projection trace entries for the soft arm. Four new
regression tests and seven existing geometry/schedule tests pass locally.

Compare all methods at the same final frame and frames 9/12/15/20 with fixed
crops. Inspect wall material, source refill, contact, new boundary artifacts
and remaining texture quality. Low-saturation wall fraction diagnoses the
specific white contamination but is not a general photo-quality metric.
One synthetic cake/one seed cannot establish dataset-level superiority,
generalization, statistical significance, or physical volume conservation.

## Exploratory compositor correction (added after first E7 hard output)

This is kept separate from the GPU pair. The original photograph is the base,
and per-frame support is formed from the frozen RGB proxy's actual changes,
the existing cavity-repair alpha and the planned payload. A 12-pixel margin
and an 8-pixel inward feather retain the operation core exactly while restoring
the original photograph elsewhere. At frame zero, unchanged proxy and empty
repair support yield the exact original image. The compositor is applied with
the same parameters to both E7 outputs. It is deterministic postprocessing,
not evidence that diffusion learned better background preservation. Unplanned
motion beyond the margin may be clipped and requires visual inspection.

After adding its regression, five new tests and seven earlier tests pass.
