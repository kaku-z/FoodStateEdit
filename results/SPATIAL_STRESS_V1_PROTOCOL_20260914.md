# Spatial stress pilot: depth approach and yaw rotation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: user-authorized experiment implementation and execution
- Date: 2026-09-14
- Verification Status: all six generations completed, SHA-256 verified and internally visually reviewed; see `SPATIAL_STRESS_V1_RESULT_20260914.md`
- Version: spatial_stress_v1

## Question and scope

Does explicit relative-depth projection preserve the requested spatial change through VACE better than translation-only or similarity-transform 2D controls?
Two development cases, one seed each: potage approaching the camera and a synthetic cake bite turning 45 degrees in relative depth. This is a mechanism/stress diagnostic, not a held-out benchmark. It does not estimate population success rates or demonstrate statistical superiority.

## Frozen comparisons

- Three arms per case: `planar_translation`, `planar_similarity`, `relative3d`.
- The similarity arm has a generous oracle-assisted fit to projected payload landmarks. It receives 2D rotation and scale, so it is stronger than translation-only, but is not an independently depth-free deployed method.
- Identical reference, text, negative text, shared union mask, seed 2, 21 frames, 20 steps, VACE scale 1, LoRA off, TTM off within each case.
- Same pretrained VACE and original runtime, no model download or additional training.
- All arms share the contact-anchor trajectory and smoothstep. Smooth motion by itself therefore cannot establish a 3D-specific contribution.
- Current renderer has fixed face ordering and procedural source repair. No new occlusion-correctness claim is supported. Dynamic visibility is a separate unfinished experiment.

## Review before any PPT claim

Inspect every raw lossless frame and the projected sequence for all six cells, without dropping negative examples.

1. From contact onward, does the utensil remain in contact with the payload?
2. During lifting, does the same payload follow the utensil rather than detach or duplicate?
3. Does the requested size/perspective change survive in the generated frames?
4. Are there abrupt shape changes, collapsed utensil geometry, or damaged food?
5. Is photo quality acceptable? Report separately from motion and geometry.
6. Compare raw and composited frames. Mask-exterior preservation is imposed by common compositing and is not credited to the 3D representation or the generator.

Any agent visual assessment is an internal, non-blind observation, not human-study data. Ambiguous contact is recorded as uncertain. Do not translate differences in control-space reprojection residual into output quality or physical correctness.

## Execution

Host: gp40, explicit physical GPU 0 for soup and GPU 1 for cake. Three arms sequential per worker, at most two workers.

Fresh runtime: `/host/space0/guo-z/tf-ufi/runtime/spatial_stress_v1_20260914T1358Z`

Fresh output: `/host/space0/guo-z/tf-ufi/outputs/spatial_stress_v1_20260914T1358Z`

Each worker rechecks the existing A6000 resource gate immediately before execution. It exits itself if another user's compute process appears on its GPU. A 3600-second wall deadline bounds each new worker and preserves any partial output. No automatic retry or reuse of failed paths.

Local frozen controls: `artifacts/spatial_stress_controls_v1_20260914T1352Z`.

Controls manifest SHA-256: `090c2a4cdab600ae5b16ed58a5d9925e464b29adf46f42fc2719c4cd9dcef674`.

Upload bundle SHA-256: `10b33b3a7fbebc12644358be14ed34f6bef3697fb319572cb1b56850a76a0e16`.

## Outcome (after frozen execution)

- Six of six generations complete, each with 21 lossless raw frames. 171 non-cache evidence files passed remote/local SHA-256 verification.
- Full raw and lossless-composite frame-grid inspection completed as internal AI visual assessment, not a human study.
- Control representation effects are visible, but improved end-to-end photo quality and broad 3D superiority are not established. Main PPT/PDF figures remain unchanged.
- Implementation wording correction: `MaxFilter(33)` is a 33x33 window (approximately 16-pixel expansion on each side), not a 33-pixel radius. Frozen config and controls are preserved.
