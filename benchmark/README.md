# Benchmark workspace

Current canonical Day 2 files:

- `candidate_inventory_v1.csv`: deterministic 128-image candidate pool;
- `inventory_summary_v1.json`: sampling, category, hash, and duplicate summary;
- `visual_review_v3.json`: complete first-pass eligibility decisions;
- `candidate_inventory_reviewed_v3.csv`: all candidates with decisions;
- `data_manifest_provisional_v3.csv`: current 60-case 5-pilot/10-test split;
- `split_summary_v3.json`: deterministic selection rule and selected case IDs;
- `DATASET_PROVENANCE.md`: official use terms, paths, and required citation.

The current split is provisional. It must not be renamed to `frozen` until a
human confirms the four boards in `artifacts/day2_frozen_review_v3/`. Source
images are not copied into this repository and must not be redistributed.

`audit/day2/` retains the rejected inventory/review iterations. In particular,
v0 contained unsuitable omelet-rice candidates, v1 missed five edge utensils,
and v2 retained three composition/artifact failures. These rows were not
silently replaced.
