# FoodStateEdit paper release workspace

This repository is the frozen, lightweight workspace for the 20-day
FoodStateEdit paper sprint started on 2026-08-26.

The historical experiment root, model weights, videos, and large output folders
remain outside this repository. This repository tracks source snapshots,
schemas, manifests, evaluation definitions, paper figures/tables, and the exact
paths and hashes needed to reproduce reported results.

## Frozen research task

Given a food image without the target manipulation, a structured action
specification, a utensil type, and sparse/local spatial controls, generate an
edited image that depicts the utensil manipulating the food while satisfying
material, contact, conservation, source-removal, and scene-preservation
constraints.

The task is structured-control image editing. It is not currently claimed to be
text-only, fully automatic, universally general, or universally photorealistic.

## Current method core

FoodStateEdit is organized as a `3-D action -> 2-D control -> learned render`
pipeline:

1. lift a sparse food/utensil action specification into reconstructed or
   explicitly labelled relative 3-D;
2. solve rigid utensil motion, deformable food motion, contact, and conservation
   in that 3-D action space;
3. project with one camera and depth-aware visibility into a temporal 2-D
   control;
4. render appearance with VACE, optionally conditioned by a task adapter; and
5. enforce exact scene preservation outside the declared motion support.

The historical ramen case contains reconstructed VGGT geometry. The current
fork pilot uses a normalized relative-3-D camera because its audited depth
checkpoint is unavailable; the two evidence levels are not interchangeable.

## Planned material/action families

- liquid: spoon scooping soup;
- granular: spoon or spatula scooping fried rice;
- strand: chopsticks lifting ramen or udon;
- strand-contact: fork twirling and lifting pasta.

## Start here

- `results/LOCAL_PROGRESS_20260905.md`: Chinese progress report, presentation
  narrative, figure index, and Day 13 recovery status during the server outage;
- `SCOPE_FREEZE.md`: claims, non-claims, gates, and change policy;
- `PROVENANCE.md`: upstream code, environment, model, and result hashes;
- `schemas/`: frozen case, run, and metric contracts;
- `benchmark/`: anchor and future pilot/test manifests;
- `results/CURRENT_RESULTS_INVENTORY.md`: evidence available at sprint start;
- `DAY1_CHECKLIST.md`: completion evidence for the first sprint day.
- `DAY2_CHECKLIST.md`: auditable entry point for the 60-case benchmark freeze.
- `DAY3_CHECKLIST.md`: closed baseline availability and four-anchor execution gate;
- `DAY4_CHECKLIST.md`: four-layer staged integration and resident-worker gate;
- `DAY4_EXCLUSIVE_V2_CHECKLIST.md`: closed same-seed non-shadowing-mask
  follow-up gate;
- `DAY5_CHECKLIST.md`: geometry-locked harmonization contract and gate;
- `DAY5_VACE_DIRECT_CHECKLIST.md`: native VACE static-proxy baseline gate;
- `DAY5_MATERIAL_RENDER_CHECKLIST.md`: topology-preserving 2.5D material-render gate;
- `DAY6_DYNAMIC_MULTIKEY_CHECKLIST.md`: deterministic action-phase VACE gate;
- `DAY7_ADAPTER_V0_CHECKLIST.md`: claim-limited VACE-LoRA infrastructure-smoke gate;
- `DAY8_3D_PROJECTION_CHECKLIST.md`: relative-3-D fork motion/projection gate;
- `DAY9_ACTION_ADAPTER_CHECKLIST.md`: completed action-supervised LoRA training,
  corrected wrapped-VACE injection, and blind fork result gate;
- `DAY10_SEEN_ACTION_LORA_CHECKLIST.md`: completed seen-family underfit
  diagnostic and the requirement for phase-varying supervision;
- `DAY11_PHASE_ACTION_OVERFIT_CHECKLIST.md`: completed phase-varying 64-step
  overfit training and negative seen-checkpoint capacity gate;
- `DAY12_PHASE_ACTION_ISOLATION_CHECKLIST.md`: completed single-sample adapter
  isolation test with a negative matched-exposure semantic gate;
- `DAY13_3D_GUIDED_FLEXIBLE_COMPLETION_CHECKLIST.md`: frozen three-arm design
  for testing 3D-derived topology-weighted diffusion completion;
- `results/DAY5_METHOD_GRID_CAPTION.md`: hash-traceable qualitative comparison caption;
- `results/DAY6_DYNAMIC_MULTIKEY_PREFLIGHT.md`: pre-output schedule, hashes, and
  deterministic-control invariants;
- `results/DAY6_VACE_DYNAMIC_MULTIKEY_REVIEW.md`: projected outcome and the raw
  ramen support-mismatch diagnosis;
- `results/DAY6_MOTION_UNION_PROJECTION_DIAGNOSTIC.md`: post-hoc support test,
  exact-protection audit, and claim limits;
- `results/DAY7_ADAPTER_V0_PREFLIGHT.md`: frozen trainer, dataset hashes,
  resource gate, and the boundary between smoke evidence and generalization;
- `results/DAY8_3D_PROJECTION_PILOT.md`: fork/spaghetti 3-D geometry result,
  depth-crossing evidence, and the strict reconstructed-versus-relative claim
  boundary;
- `results/DAY8_VACE_3D_COMPARE_PREFLIGHT.md`: frozen same-seed 2-D-versus-3-D
  control comparison and fail-closed GPU gate;
- `results/DAY8_VACE_FORK_3D_REVIEW.md`: completed same-seed render, separate
  action/photo verdicts, and the renderer-bottleneck decision;
- `results/DAY9_ACTION_ADAPTER_RESULT.md`: checkpoint evidence, the preserved
  zero-patch runtime failure, corrected 80-branch evaluation, and blind negative
  action/photo verdict;
- `results/DAY11_PHASE_ACTION_OVERFIT_TRAINING.md`: completed 64-step training,
  four checkpoint hashes, official-loader validation, and the no-claim boundary;
- `results/DAY11_PHASE_ACTION_CHECKPOINT_SWEEP_RESULT.md`: completed two-sample,
  five-condition sweep, separate action/contact and photo reviews, and the
  no-expansion decision;
- `results/DAY12_PHASE_ACTION_ISOLATION_PREFLIGHT.md`: dedicated udon/spoon
  training arms, immutable hashes, resource gate, and interference decision rule;
- `results/DAY12_PHASE_ACTION_ISOLATION_RESULT.md`: completed dedicated
  checkpoint sweeps, matched-exposure review, and the no-blind-fork decision;
- `results/DAY13_3D_GUIDED_FLEXIBLE_COMPLETION_DESIGN.md`: causal experiment
  design, loss definition, positive-contribution gate, and execution lock;
- `results/day7_adapter_v0_preflight_initial_v2.json`: machine-readable remote
  preflight evidence; 51/52 checks passed and only the occupied-GPU gate failed;
- `results/day7_adapter_v0_gp39_failure_missing_librosa_v1/`: preserved
  pre-model-load failure caused by the upstream trainer's unused audio operator;
- `configs/baselines_v1.json`: formal methods, seeds, schedules, and exclusions;
- `configs/staged_schedule_v1.json`: pre-inference frozen four-layer pilot schedule;
- `configs/staged_schedule_v2_exclusive.json`: pre-inference frozen follow-up
  that isolates non-shadowing projection-mask ownership;
- `results/DAY3_BASELINE_AUDIT.md`: evidence-backed reproducibility audit;
- `results/DAY3_SMOKE_REVIEW.md`: separate technical and provisional visual
  outcomes for the eight GeoEdit baseline runs;
- `results/DAY4_STAGED_REVIEW.md`: four-layer staged-pilot outcome, failure
  analysis, and no-expansion decision;
- `results/day4_staged_v1/`: hash-verified resident-worker manifests, internal
  diagnostic labels, and the semantic-mask overlap audit;
- `results/DAY4_EXCLUSIVE_REVIEW.md`: same-seed exclusive-mask follow-up and
  representation/backend gate decision;
- `results/day4_exclusive_masks_v2/` and
  `results/day4_staged_exclusive_v1/`: frozen mask packages, run manifests,
  pixel comparison, and internal diagnostic labels;
- `results/day4_common_proxy_v1/`: machine-readable four-anchor common-proxy
  manifests and the internal review decision (source images are not tracked);
- `benchmark/data_manifest_v1.csv`: frozen 60-case 5/10 split;
- `benchmark/canonical_input_manifest_v1.csv`: hash-locked common x4-aligned
  editing inputs derived from the frozen sources;
- `benchmark/anchor_manifest_v1.csv`: the four primary pilot anchors;
- `benchmark/ANCHOR_ANNOTATION_PROTOCOL.md`: five-layer action annotation
  semantics and hard constraints;
- `benchmark/DATASET_PROVENANCE.md`: dataset terms, paths, and citation.

## Local contract validation

From PowerShell in this repository, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_release.ps1
```

This check does not load or download model weights. It validates the copied
framework tests, JSON contracts, Python syntax, and the four frozen GeoEdit
override hashes.

## Large-file policy

No model weights or generated videos are committed. Every formal run must refer
to immutable paths plus file hashes in its run manifest. Existing weights must
be reused with download disabled.
