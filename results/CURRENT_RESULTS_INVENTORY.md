# Evidence inventory at sprint start

Snapshot date: 2026-08-26

Latest consolidated status: [2026-09-05 local progress report](LOCAL_PROGRESS_20260905.md).
The dated sections below retain historical observations; later sections supersede
earlier pending-work statements.

## 2026-09-06 Day 13 relative-3D topology-weighted diagnostic

- three matched 32-step arms completed with identical trainable initialization
  aggregate and recorded row/timestep/noise sequence;
- all six checkpoints passed 160-tensor structure and 80-layer official-load
  validation;
- one resident pipeline completed all five 21-frame conditions with exact
  outside-support preservation;
- weighted versus matched relative3D-uniform improved topology and pinch MAE by
  only 2.22% and 2.83%, below the precommitted 5%/5% gate;
- technical visual inspection found no clear semantic separation; two-person
  anonymous review package is ready but cannot reverse the failed numeric gate;
- conclusion: clean negative primary result with a small directional trend;
  no blind fork or balanced expansion is unlocked.

Result: [DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.md](DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.md).

## Validated evidence

### Noodle geometry benchmark

- frozen-SAM strand selection and deterministic geometry: 3/3 hard pass;
- natural appearance completion with the earlier VACE backend: 1/3;
- conclusion: the geometry front end generalizes across three audited images,
  but the appearance backend does not yet generalize.

Historical report: `../BENCHMARK_V2_REPORT.md` in the experiment root.

### Controlled liquid fill

- one controlled clear-broth fill case is a strict end-to-end pass;
- liquid/garnish decomposition and no-warm inference removed the major seam;
- real soup-image generalization is not validated.

### Spoon-scooping staged pilot

- one controlled source, one formal seed, four internal schedules;
- best schedule: rigid 18, material 8, 21 frames, 20 steps, no warm start;
- action-region proxy MAE: 19.293 to 11.411 (40.9% reduction);
- rigid tolerant-edge F1: 0.767 to 0.858 (11.9% relative increase);
- hard semantic/structural gate: pass;
- strict photorealism: partial because reflection/contact shadow remain soft.

### Universal state prototype

- liquid volume solver: state-only pass;
- granular mass/component solver: state-only pass;
- ramen adapter: action pass, photo incomplete;
- conclusion: shared state schema exists, but shared image rendering does not.

## Missing evidence

- no fork/pasta case;
- no rendered fried-rice case;
- no 60-case frozen test set;
- no three-seed formal protocol;
- no same-control comparison against three or more strong baselines;
- no blind multi-reviewer physical-realism study;
- no independent test proving that proxy similarity equals physical success;
- no clean release commit before this paper workspace.

## Reporting rule

Pilot metrics may motivate hypotheses and schedule choices. They may not be
combined with the future frozen test set as if they were held-out evidence.

## Sprint additions through Day 4

This section updates the dated sprint-start snapshot without rewriting it.

- The frozen 60-case benchmark, canonical input hashes, four pilot anchors, and
  Day 3 no-edit/vanilla/unified smoke runs are now tracked in this repository.
- Day 3 stochastic controls completed `8/8`; provisional action success was
  `1/4` for vanilla GeoEdit and `0/4` for the unified same-proxy control.
- Day 4 staged v1 completed `4/4` with exact outside-mask preservation but had
  `0/4` provisional action and photo success, so it is closed as a negative
  pilot and is not eligible for seed/test expansion.
- The Day 4 overlap audit identifies shorter-window shadowing as the next
  implementation target. These remain pilot diagnostics, not formal held-out
  claims.
- The controlled exclusive-mask follow-up removed that shadowing but retained
  `0/4` action success. This isolates the next bottleneck to the proxy/backend
  representation interface rather than mask overlap alone.

## Sprint additions through Day 6

- Geometry lock and material render preserve the declared action topology on
  `4/4` anchors, but both remain `0/4` for strict photo realism.
- Native VACE with a repeated static proxy is `0/4` for action and photo
  success, so disabling GeoEdit TTM does not solve thin-structure loss.
- Deterministic dynamic-multikey VACE is also `0/4` after the frozen final-mask
  projection, but it retains a solid utensil trajectory on all four anchors.
- The raw frozen ramen frame contains two separate chopsticks and one lifted,
  bowl-connected noodle. Exact final-mask projection removes most of the sticks
  because the generated geometry drifts outside that narrower support. This is
  a positive diagnostic signal, not a reported action success.
- The next eligible work is one support-aware diagnostic. No current negative
  pilot is eligible for additional seeds or the frozen 60-case test set.
- The post-hoc motion-union projection diagnostic confirms the support mismatch
  for ramen: two chopsticks and a lifted noodle are retained, but the broader
  support and residual noodle ghosting prevent a formal success claim. The
  larger support does not repair soup, rice, or pasta semantics.

## Sprint additions through Day 7

- The two-sample high-noise VACE-LoRA infrastructure smoke completed on gp39:
  two optimizer steps, two hash-locked checkpoints, and a normal exit.
- The final rank-8 adapter has 160 finite BF16 tensors in 80 complete A/B
  pairs; the offline official VACE loader updated all 80 targeted tensor pairs.
- This closes only the adapter plumbing gate. Targets equal controls, so the
  run supplies no evidence for action correctness, generalization, or photo
  realism and is not a new visual baseline.

## Sprint additions through Day 8

- The method core is now explicit: construct the utensil and food motion in
  3-D, resolve contact and visibility there, project it into a temporal 2-D
  control, and use the learned backend only for appearance rendering.
- The first non-noodle 3-D pilot covers the frozen pasta/fork anchor. It
  projects one rigid fork plus four plate-connected spaghetti strands over 21
  frames with depth-tested front/back crossings around the fork.
- The selected frame has fork depth `0.82`, helix depth range
  `[0.80128, 0.83872]`, reprojection error below `1.3e-13` pixels, and zero
  pixel change outside the declared motion support.
- This is a geometry-control pass, not an image-editing success. The fork case
  uses relative 3-D and a normalized pinhole camera; only the preserved ramen
  experiment currently contains reconstructed VGGT scene geometry.
- The frozen same-seed VACE comparison completed on gp39: 36/36 passing
  preflight checks, one pipeline load, 21 decoded frames, and zero pixel change
  outside the Day 8 motion support.
- Relative-3-D conditioning improves the Day 6 malformed spoon-like head into
  one recognizable fork with a coherent trajectory. It still produces no
  wrapped, lifted, plate-connected spaghetti, so strict action and photo
  success remain `0/1`.
- The raw comparison shows the same conclusion before exact projection. The
  positive result is therefore limited to rigid utensil topology; the current
  bottleneck is deformable food rendering and utensil-food binding.
- This negative single-anchor render is closed without seed or test-set
  expansion. The next eligible gate is non-identity action supervision for the
  VACE adapter, not additional hand-designed geometry alone.

## Sprint additions through the Day 9 result

- A stricter target audit found zero real-photo action targets suitable for
  formal adapter training. Exactly two synthetic ImageGen pseudo-targets are
  eligible for a claim-limited mechanism pilot: chopsticks/udon and spoon/broth.
- The older case-20 natural VACE frame is no longer eligible as action
  supervision under the current visible-lift gate; its pronounced oracle lift
  collapses to a short bowl-edge contact. This does not invalidate the separate
  deterministic geometry checks.
- The first complete raw-oracle dataset changed roughly 96%--99.9% of pixels
  and was rejected. The frozen v3 data changes only bounded action supports
  (5.086% and 6.961%) and preserves every pixel outside them exactly for both
  target and control.
- Pasta/fork occurs zero times in training. It remains the blind same-seed
  LoRA-off/on mechanism test after checkpoint validation.
- The adapter pilot is frozen at two pseudo-targets, rank 8, 16 optimizer steps,
  offline existing models only. Training and official checkpoint validation
  completed on gp39; the final checkpoint has 80 finite, nonzero LoRA pairs.
- The first full-pipeline LoRA run exposed a frozen-runtime incompatibility:
  the stock low-VRAM hot-loader patched zero VACE layers, producing files
  byte-identical to Day 8. The failure is preserved and excluded from adapter
  quality conclusions.
- Corrected inner-block injection reached all 80 high-noise VACE targets. The
  LoRA-on result then differed numerically from LoRA-off while preserving every
  pixel outside the motion support, proving that the adapter reached inference.
- The blind action still failed. Across six key frames, the fork presses into
  sauce but never wraps, lifts, or retains a plate-connected spaghetti strand.
  Strict action/contact and photo success remain `0/1`; no generalization claim
  is allowed.
- The next gate is a seen-family LoRA-off/on evaluation on the udon and broth
  training anchors, separating underfitting from cross-family transfer failure.

## Sprint additions through the Day 10 seen-family diagnostic

- Both frozen seen-family comparisons completed with one pipeline load per
  worker, all 80 high-noise VACE LoRA branches active, 21 frames per condition,
  and exact preservation outside the edit support.
- The udon/chopsticks result is slightly farther from its target with LoRA and
  still contains neither two chopsticks nor a lifted noodle.
- The broth/spoon result is 0.27% closer to its target with LoRA, but the same
  coherent spoon-with-broth action is already present with LoRA disabled. It is
  a useful base-renderer single-sample success, not an adapter gain.
- Across both seen families there are `0/2` visible semantic improvements from
  LoRA-on. The current adapter is therefore underfit, rather than failing only
  through cross-family transfer to fork/pasta.
- All reviewed video frames are effectively static. Repeating a final target
  for 21 training frames did not supervise approach, contact, acquisition, and
  lift as temporal events.
- The next eligible experiment is a bounded phase-varying overfit sanity test.
  Another blind fork run or broad benchmark expansion is not yet justified.

## Sprint additions through Day 11 training

- The two seen synthetic targets now contain explicit source, approach,
  contact, lift, and final-hold phases instead of one repeated final image.
- The frozen 64-step high-noise VACE-LoRA run completed on gp40 using only
  existing offline models. Checkpoints were saved at steps 16, 32, 48, and 64.
- The final checkpoint contains 160 finite BF16 tensors in 80 complete rank-8
  pairs, and the official VACE loader updated all 80 intended targets.
- This closes the training/plumbing portion of the overfit sanity gate only.
  Semantic learning and photo realism have not yet been evaluated.
- The next required experiment is the same-seed LoRA-off and four-checkpoint
  sweep on both seen phase-varying samples. A blind fork rerun and balanced
  multi-family expansion remain blocked until a clear seen semantic gain.
- That five-condition sweep is now hash-frozen. Its first launch produced no
  evaluable condition: the spoon preflight correctly rejected a newly occupied
  GPU, while the udon worker preserved an OOM after a separate process occupied
  its GPU between preflight and model startup. These are resource-race records,
  not semantic or photo-quality evidence; conflict-free retries remain pending.

## Sprint additions through the Day 11 checkpoint sweep

- A conflict-free serial retry completed both seen synthetic samples on gp39
  using the frozen seed, 21 frames, 20 inference steps, VACE scale 1, and TTM
  disabled. Each sample loaded the pipeline once and preserved every pixel
  outside the frozen support exactly.
- All 36 manifest-recorded outputs were pulled back and rehashed with zero
  mismatches. The retry used new output and preflight paths and did not
  overwrite either the preserved gp40 failures or the mismatched older gp39
  LoRA directory.
- Udon target-support MAE improves monotonically from 13.6511 at LoRA-off to
  11.9998 at step 64, but the phase/contact review shows no clearer pinch,
  payload acquisition, connected lift, or final hold.
- Broth is numerically best at step 16 (20.9574 versus 21.2355 LoRA-off), but
  the visible spoon action already occurs with LoRA disabled. Steps 48 and 64
  add a pale/translucent spoon-bowl artifact and are less photographic.
- The conservative result is `0/2` clear semantic gains and `0/2` clear photo
  gains. The seen synthetic overfit-capacity gate is negative; balanced family
  expansion and another blind fork evaluation remain blocked.
- This result cannot support claims about generalization, real data, physical
  correctness, or paper-level photo realism.

## Sprint additions through the Day 12 isolation result

- Dedicated udon-only and broth-only adapters were compared at step 32 against
  the shared Day 11 step 64 checkpoint at the same 32 selected-sample
  exposures. Dedicated step 64 remains an additional capacity probe only.
- Both five-condition sweeps completed serially on a conflict-free gp38 A6000,
  with one pipeline load per sample, 21 frames per condition, and exact
  outside-support preservation.
- All 40 pulled run-package files, 36 manifest-recorded outputs, and 14
  supporting evidence files were rehashed with zero mismatch. A client-side
  SSH reset during the spoon observation did not stop the remote worker; the
  fresh remote run completed all conditions normally.
- Udon dedicated step 32 is numerically worse than shared step 64 and has no
  clearer pinch, acquisition, connected lift, or hold.
- Broth dedicated step 32 is numerically and visually cleaner than the degraded
  shared step 64 result, but the same visible spoon action already occurs with
  LoRA disabled. Later dedicated checkpoints reintroduce a pale spoon-rim
  artifact.
- The matched interference diagnosis is not supported: clear semantic gain is
  `0/2`, and clear photo-realism gain over LoRA-off is `0/2`. Balanced expansion
  and blind fork evaluation remain blocked.
- This is only a seen synthetic single-sample isolation result; it cannot
  support claims about generalization, real data, physical correctness, or
  paper-level photo realism.

## Day 13 implementation and 2026-09-05 outage snapshot

- The relative-3D udon dataset, three topology masks, matched training entry
  point, preflight, and evaluation implementation are hash-frozen in
  `configs/flexible_completion_execution_v1.json`.
- Prior task observations recorded complete 32-step planar-uniform and
  relative3d-uniform training. These Day 13 remote checkpoints have not been
  pulled and verified in this local consolidation.
- Topology-weighted v3 was launched remotely; its completion is unknown.
  User-confirmed server power interruption prevents reading current evidence.
- The five-condition evaluation has not started. No positive contribution or
  Day 13 visual-quality result is asserted.
- Local Day 11/12 evidence and Day 13 frozen inputs are organized by
  `scripts/package_local_progress.ps1`; generated package:
  `artifacts/local_progress_20260905_v1/`. The package contains selected figures
  and reports, not a complete backup of remote training outputs.
