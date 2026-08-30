# Evidence inventory at sprint start

Snapshot date: 2026-08-26

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
- Photo realism remains unevaluated until the frozen 3-D control is passed
  through VACE and compared at the same seed against the Day 6 2-D control.
