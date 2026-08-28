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

## Planned material/action families

- liquid: spoon scooping soup;
- granular: spoon or spatula scooping fried rice;
- strand: chopsticks lifting ramen or udon;
- strand-contact: fork twirling and lifting pasta.

## Start here

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
