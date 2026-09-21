# E4: Natural remaining cake after bite removal

## Scope and evidence

The user requested a more natural remaining portion after taking a bite. This pilot adds one local VACE completion pass to the recorded E3 cake sequence. The scientific status is a single synthetic development case, seed 1, internally reviewed without blinding. It is not a new effectiveness or generalization result.

Code inspection found that the existing source renderer fills the vacated silhouette with an average plate color and warps the same front-face cake texture onto two hypothesized cut faces. This is an approximate control image, not observed ground truth. It is a plausible contributor to artificial-looking surfaces, but this pilot does not isolate that cause from the prompt or second inference pass.

## Change

- Keep frames 0–8 unchanged in the delivered sequence.
- Starting with lift at frame 9, compute the visible source repair mask as the dilated original bite silhouette minus the dilated moving payload silhouette.
- Pass a hybrid conditioning video: actual E3 photo pixels outside the repair mask and geometric cut-face boundaries inside the mask. Thus the inactive VACE branch contains photographs, not black Canny background.
- Draw only the two cut-face boundaries; do not paste donor cake texture or draw an artificial foreground contour around the plate floor.
- Complete the source region with the frozen VACE model and original reference, then composite using an inward-feathered per-frame alpha.
- Preserve all recorded baseline pixels outside this alpha, including the planned moving payload. This is a guarantee with respect to the manual planned mask, not independently verified object segmentation.

## Execution and integrity

gp40 physical GPU 0 passed a fresh RTX A6000 gate: 48,539 MiB free, 0% utilization, no compute process, 251,842 MiB available host memory. Execution lasted 240.17 seconds, with one pipeline load, 21 frames, 20 steps, seed 1, scale 1, LoRA off and TTM off. No retries were performed.

- Controls manifest SHA-256: `37f55d066436671e97559834bb5a1720866f71dfd5e45fa4f716f7ecdd5daa49`
- Runner SHA-256: `d3f76d0a421f0c821197774138b40ca2af598d732c6320f88caf7a5602cd9332`
- Remote/local SHA-256: 50 files, no missing files, no mismatches, no extras.
- All 21 raw and 21 projected PNG frames are present.
- Maximum difference outside repair alpha: 0.
- Maximum difference in planned payload mask: 0.
- Three unit tests passed for reveal timing, payload exclusion, hybrid conditioning and protected-pixel compositing.

## Visual finding

The source cut faces appear more continuous and smoother, and the exposed plate floor has a stronger contact shadow. However, the shadow is too dark, the cavity remains a regular cuboid notch, and some fine crumb detail is lost. There is no clear overall naturalness win from this single internal comparison. The existing large E3 support-shaped seam remains, because it lies mostly outside this intentionally source-only repair.

Correction to the preliminary verbal diagnosis: close inspection shows that part of the apparent residual block is the rear wall of the notch. It must not automatically be scored as duplicated food. A visible source correspondence or real before/after reference is needed to distinguish the two.

The current image explicitly starts from a pre-cut rectangular bite, so straight boundaries are not intrinsically wrong. To evaluate a genuinely scooped or torn bite, the input task and shared bite geometry must change coherently; simply adding random irregularity to the cavity would break correspondence with the lifted portion.

## Next algorithm change

Model the remaining geometry and lifted portion from the same removal volume. Use material-specific boundaries (crumbly cut cake, redistributed rice grains, connected noodle strands) and provide separate guidance for cavity walls versus exposed plate. Estimate local illumination from the surrounding plate and cake instead of relying on an unconstrained shadow prompt. Evaluate source naturalness, cavity/payload compatibility and whole-image naturalness separately against E3 and a plain extra-pass baseline with the same compute budget.

## Files

- Local evidence: `artifacts/e4_source_cavity_gp40_20260917T051026Z`
- Before/after panel: `artifacts/e4_source_cavity_review_v1_20260917/before_after.png`
- Validation: `artifacts/e4_source_cavity_review_v1_20260917/validation.json`
- Controls: `artifacts/e4_source_cavity_controls_v1_20260917`
- Remote output: `/host/space0/guo-z/tf-ufi/outputs/e4_source_cavity_v1_20260917T051026Z`
