# FoodStateEdit paper release workspace

The [completed Day 25 ablation](results/DAY25_SAME_CONDITION_ABLATION_RESULT_20260913.md)
now covers the full 4-case x 3-seed x 5-condition matrix: 47 new and 13
hash-locked cells, with all 60 final images verified.  Explicit motion controls
are necessary relative to native input on these selected cases, but internal
review finds no clear relative3D-over-planar or material-adaptive-over-fixed
gain in any of the four cases.  Held-out effectiveness therefore remains
locked; the 20-image Day 32 pilot is the next falsification gate.

The [Day 32 material-policy pilot](results/DAY32_MATERIAL_POLICY_PILOT_PLAN_20260913.md)
is frozen as 210 unique generations over all 20 development images.  gp39 was
blocked by newly occupied GPUs; after Day 25 completed, shard_b passed gp40
GPU1 preflight and began pipeline loading.  Shard_a remains safely unstarted
because its preserved shared preflight path requires a new non-overwriting
retry version.  The 40-image held-out split remains locked.

The [Day 31 family-template control pilot](results/DAY31_FAMILY_TEMPLATE_CONTROLS_PILOT_20260913.md)
completed deterministic planar and relative-3D 21-frame controls for all 20
pilot images across liquid, granular, strand and strand-contact families.  All
180 collected files rehashed exactly.  These visibly schematic proxies are
accepted only as VACE intervention inputs; they are not final images or
effectiveness evidence.  The held-out split remains locked until the complete
Day 25 review freezes the material-level VACE-scale rule.

The [Day 30 held-out readiness audit](results/DAY30_HELDOUT_INPUT_READINESS_20260913.md)
verified all 40 frozen real test inputs on gp40 with zero SHA-256 mismatch and a
balanced 10-image-per-family split, without opening images or generating test
outputs.  Day 25 is now complete, but its negative relative3D/material-scale
gate still prevents test generation.  All 40 annotations remain pending and
the final material-level Ours rule requires the broader Day 32 pilot.

The [Day 29 available-baseline statistics](results/DAY29_AVAILABLE_BASELINE_STATISTICS_20260913.md)
now aggregate the executed ChordEdit and Qwen development outputs with a
case-clustered descriptive bootstrap. Qwen is `3/12` for provisional action and
`12/12` for photo appearance but `0/12` for exact preservation; ChordEdit is
`0/12`, `1/12`, and `12/12`, respectively, and both remain `0/12` strict
end-to-end. Three independently shuffled 24-item blind-review packages are
ready, with the method key stored separately. These remain selected development
data: Day 25, frozen Ours, held-out generation and actual independent ballots
are still required before confirmatory claims.

The [Day 28 Qwen utensil-refinement diagnostic](results/DAY28_QWEN_UTENSIL_REFINEMENT_RESULT_20260913.md)
ran one crop-local candidate for each selected ramen, soup, fried-rice and
synthetic-cake result. One offline Qwen pipeline produced all four candidates;
all 55 transferred evidence hashes matched and every protected pixel stayed
exact. Qwen visibly improves some local material cues, especially the soup
spoon and cake fork, but all candidates failed at least one frozen automatic
gate, so all formal finals conservatively roll back. This is useful negative
diagnostic evidence, not a promoted final-stage improvement claim.

The [Day 27 Qwen-Image-Edit direct-input baseline](results/DAY27_QWEN_IMAGE_EDIT_DIRECT_BASELINE_RESULT_20260912.md)
completed 12/12 matched development outputs with one offline pipeline load on a
safe RTX A6000. All 62 remote output files and four supporting evidence files
matched local SHA-256 values. The actual Qwen outputs are a strong photographic
baseline (`12/12` provisional Photo Success), but only `3/12` passed the full
action rubric and none preserved the frozen outside-support pixels exactly.
The post-hoc hard composites are retained only as a mask-misregistration
diagnostic, not as the Qwen endpoint or preservation evidence. FLUX Kontext,
ChronoEdit, independent blinded scoring and held-out testing remain open.

The [Day 26 ChordEdit direct-input baseline](results/DAY26_CHORDEDIT_DIRECT_BASELINE_RESULT_20260912.md)
completed 12/12 same-input outputs across ramen, soup, fried rice and synthetic
cake on four safe RTX A6000 GPUs. All 72 remote evidence files matched local
SHA-256 values and exact outside-support preservation passed, but the internal
non-blind review found `0/12` Action Success and `0/12` Strict End-to-End
Success. This adds one reproducible external-editor negative baseline; it does
not replace the still-unavailable FLUX Kontext or ChronoEdit comparisons and
does not unlock held-out testing by itself.

The [Day 23 geometry-locked ChordEdit pilot](results/DAY23_GEOMETRY_LOCKED_CHORDEDIT_REFINEMENT_RESULT_20260912.md)
tested one local diffusion-refinement candidate after relative-3D VACE. Exact
outside-support preservation succeeded, but the editor generated a second
spoon-like object; the semantic gate rejected it and the final image rolled
back byte-for-byte. The companion [native/planar/relative-3D soup ablation](results/DAY24_NATIVE_PLANAR_RELATIVE3D_SOUP_COMPARISON_20260912.md)
shows that an explicit motion proxy is necessary for this development case,
while providing no clear evidence that relative 3-D is better than planar
control.

The [Day 21 realism-scale result](results/DAY21_REALISM_SCALE_SWEEP_RESULT_20260912.md)
tests one controlled cause of the visibly graphic outputs: over-strong VACE
following of a non-photographic proxy. Lower scale clearly improves the final
appearance for rice and cake, but only cake at scale `0.6` passes the strict
final-image gate and none of the three cases passes the full temporal-action
gate. This is a material-specific development result, not a universal fix.

The [cross-method comparison and evaluation status](results/DAY21_CROSS_METHOD_EVALUATION_STATUS_20260912.md)
consolidates the existing four-anchor, single-seed GeoEdit/VACE/FoodStateEdit
pilot and provides a hash-verified comparison grid. It also records the missing
evidence explicitly: Qwen-Image-Edit, FLUX Kontext and ChronoEdit have not been
executed on the matched inputs, and no independent blinded or held-out
benchmark exists yet.

The [2026-09-12 post-outage gp40 Day 11 reproduction](results/day11_phase_action_checkpoint_sweep_gp40_retry_v2_20260911T144335Z/REVIEW.md)
completed both frozen seen-synthetic checkpoint sweeps and verified all 45
downloaded files against the remote hashes. It reproduces the negative gate:
`0/2` clear semantic gains and `0/2` clear photo-realism gains. Balanced
expansion and blind-fork evaluation remain blocked.

The [Day 20 non-noodle results](results/DAY20_NON_NOODLE_RESULTS_20260909.md)
complete six paired soup/rice/cake conditions, with 51 output files plus one
preflight verified. The original Day 19 batch remains a preserved technical
failure; only its missing cake relative-3D arm was recovered. These are
multi-material prototypes, not clear 3D superiority or photorealism evidence.
The [two-day presentation plan](results/FORMAL_PRESENTATION_PLAN_20260909.md)
separates demonstrated engineering work from proposed creator/viewer benefits.

The [Day 19 four-material pilot](results/DAY19_MULTIMATERIAL_LAUNCH_20260908.md)
was launched on 2026-09-08: soup, fried rice, pre-cut cake and noodles, with
two same-seed control arms each. This dated launch snapshot is not a completed
result; consult the remote run manifest before reporting any outcome.

The [Day 18 high-lift support repair](results/DAY18_HIGH_LIFT_SUPPORT_REPAIR_20260908.md)
identified a concrete compositing defect: the old alpha hid raised sticks
already present in native Day 17 VACE output. Swept-support repair and one
same-seed rerun retained the utensil and removed ragged handle clipping, but
unambiguous two-stick pinch/bowl attachment and photo realism still did not
pass. This is a one-seen-synthetic engineering repair, not learned efficacy
or a new algorithmic contribution.

[Read the completed Day 13 result](results/DAY13_FLEXIBLE_COMPLETION_RESULT_20260906.md)
or the [consolidated progress and figures (中文)](results/LOCAL_PROGRESS_20260905.md).
The Day 13 topology-weighted pilot executed successfully but failed its
precommitted positive gate: the directional metric improvements were too small
and no clear visible semantic gain was found. Effectiveness is not established.

The [Day 14 contact-guidance feasibility experiment](results/DAY14_CONTACT_GUIDANCE_OBSERVER_RESULT_20260907.md)
validated limited synthetic-mask behavior but rejected the RGB observer: it
preferred a solid color block to the target. No new VACE inference ran.

The [Day 15 frozen SAM3 observer pilot](results/DAY15_SAM3_OBSERVER_RESULT_20260907.md)
also failed its prerequisite gate: 3/9 checks passed. Direct text prompting
segmented most of the noodle bowl instead of one lifted strand, did not detect
chopsticks in the synthetic target, and did not establish two-stick instances.
No new VACE inference ran and SAM3 guidance remains prohibited.

The [Day 16 geometry-prompted SAM3 pilot](results/DAY16_GEOMETRY_PROMPTED_SAM3_RESULT_20260907.md)
passed 7/9 checks and isolated a preliminary strand-specific representation
signal: planar outputs had geometry-aligned lifted-strand IoU 0, whereas the
matched relative-3D outputs reached 0.695--0.728. The two chopstick masks still
merged, and topology weighting did not beat LoRA-off, so this remains a
seen-synthetic diagnostic rather than VACE effectiveness evidence.

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
4. render appearance with VACE, optionally conditioned by a task adapter;
5. optionally propose a geometry-locked local diffusion appearance repair,
   subject to instance/contact/payload gates and automatic rollback; and
6. enforce exact scene preservation outside the declared motion support.

The historical ramen case contains reconstructed VGGT geometry. The current
fork pilot uses a normalized relative-3-D camera because its audited depth
checkpoint is unavailable; the two evidence levels are not interchangeable.

## Planned material/action families

The [multi-material scope v2](MULTI_MATERIAL_SCOPE_V2.md) makes noodles, soup,
fried rice and cake the four required exploratory demonstration classes.
Cake is a new cohesive-soft-solid class, not an already implemented or
validated adapter. The original pasta family and frozen benchmark stay intact.
No multi-family efficacy or automatic GPU execution is unlocked by this plan.

- liquid: spoon scooping soup;
- granular: spoon or spatula scooping fried rice;
- strand: chopsticks lifting ramen or udon;
- strand-contact: fork twirling and lifting pasta.
- cohesive soft solid (new pilot scope): fork lifting a pre-cut cake bite.

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
- `results/DAY27_QWEN_IMAGE_EDIT_DIRECT_BASELINE_RESULT_20260912.md`: completed
  Qwen direct-input baseline, endpoint correction, preservation diagnostic and
  internal action/photo review;
- `results/DAY28_QWEN_UTENSIL_REFINEMENT_RESULT_20260913.md`: completed
  four-case Qwen utensil-only refinement diagnostic with exact protected-region
  preservation and fail-closed rollback;
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
