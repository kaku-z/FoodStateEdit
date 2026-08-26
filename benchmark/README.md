# Benchmark workspace

Current canonical Day 2 files:

- `candidate_inventory_v1.csv`: deterministic 128-image candidate pool;
- `inventory_summary_v1.json`: sampling, category, hash, and duplicate summary;
- `visual_review_v3.json`: complete first-pass eligibility decisions;
- `candidate_inventory_reviewed_v3.csv`: all candidates with decisions;
- `data_manifest_provisional_v3.csv`: current 60-case 5-pilot/10-test split;
- `data_manifest_v1.csv`: human-confirmed frozen 60-case split;
- `freeze_record_v1.json`: hashes and confirmation evidence for the freeze;
- `canonical_input_manifest_v1.csv`: exact common editing inputs, dimensions,
  hashes, and source-to-input structure audit;
- `canonical_input_summary_v1.json`: aggregate x4 alignment and dHash audit;
- `split_summary_v3.json`: deterministic selection rule and selected case IDs;
- `anchor_manifest_v1.csv`: four source-case-to-action-anchor assignments;
- `anchor_specs_v1.json`: normalized action geometry and hard constraints;
- `anchors_v1/`: dense five-layer masks and valid case manifests;
- `anchor_annotation_summary_v1.json`: coverage, contact, and conservation
  checks for the four anchors;
- `ANCHOR_ANNOTATION_PROTOCOL.md`: layer semantics and change control;
- `DATASET_PROVENANCE.md`: official use terms, paths, and required citation.

The split was confirmed after review of the four boards in
`artifacts/day2_frozen_review_v3/`. Source images are not copied into this
repository and must not be redistributed.

`audit/day2/` retains the rejected inventory/review iterations. In particular,
v0 contained unsuitable omelet-rice candidates, v1 missed five edge utensils,
and v2 retained three composition/artifact failures. These rows were not
silently replaced.

The ignored review overlays are in `artifacts/day2_anchor_overlays_v1/`. They
embed the restricted source images and therefore must not be committed or
redistributed.
