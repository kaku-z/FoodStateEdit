# Day 29 available-baseline statistics and blind-package start

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-13
- Verification Status: ANALYZED
- Version Label: day29_available_baseline_statistics_v1

## Scope

This analysis covers the already executed ChordEdit and Qwen development baselines only:
four selected cases, three fixed seeds, and one internal non-blind scoring pass. It is not
the held-out benchmark and does not include Ours. The percentile intervals resample only
four image clusters and describe uncertainty inside this selected set; they are not
population-level generalization intervals.

## Descriptive endpoint summary

| Method | Endpoint | Pass / total (rate) | Image-cluster bootstrap 95% interval |
| --- | --- | ---: | ---: |
| chordedit | action_success | 0/12 (0.000) | [0.000, 0.000] |
| chordedit | photo_success | 1/12 (0.083) | [0.000, 0.250] |
| chordedit | preservation_success | 12/12 (1.000) | [1.000, 1.000] |
| chordedit | strict_end_to_end_success | 0/12 (0.000) | [0.000, 0.000] |
| qwen | action_success | 3/12 (0.250) | [0.000, 0.500] |
| qwen | photo_success | 12/12 (1.000) | [1.000, 1.000] |
| qwen | preservation_success | 0/12 (0.000) | [0.000, 0.000] |
| qwen | strict_end_to_end_success | 0/12 (0.000) | [0.000, 0.000] |

## Paired descriptive contrast

The contrast is Qwen minus ChordEdit, paired by case and clustered across the three seeds.

| Endpoint | Difference | Image-cluster bootstrap 95% interval |
| --- | ---: | ---: |
| action_success | +0.250 | [+0.000, +0.500] |
| photo_success | +0.917 | [+0.750, +1.000] |
| preservation_success | -1.000 | [-1.000, -1.000] |
| strict_end_to_end_success | +0.000 | [+0.000, +0.000] |

No p-values are reported: four selected development clusters and one non-independent
internal review are insufficient for a confirmatory test.

## Blind-review readiness

- Three separately shuffled reviewer packages were generated.
- Each package contains 24 anonymized source/output comparisons and a blank ballot.
- The method key is stored separately and must remain sealed until all ballots are locked.
- The package is a workflow rehearsal only. A final blind study must add frozen Ours and
  the held-out test outputs before recruitment/scoring.

## Fallacy scan

- Coverage: 11/11 checked.
- Simpson's paradox: not assessable with one case per family; aggregate and per-case rates are retained.
- Ecological fallacy: no individual-level inference is made.
- Berkson/selection bias: CAUTION — cases are selected development anchors.
- Collider bias: no covariate adjustment is performed.
- Base-rate neglect: not applicable to these binary editing endpoints.
- Regression to the mean: no extreme-score enrollment or pre/post claim.
- Survivorship bias: no failed generation was removed from the two source result files.
- Look-elsewhere effect: CAUTION — many earlier development diagnostics exist.
- Garden of forking paths: CAUTION — this is exploratory; frozen Day 25 and held-out rules remain mandatory.
- Correlation/causation: no causal effectiveness claim is made from these counts.
- Reverse causality: not applicable to the paired generation design.

## Current gate

Day 25's 47-generation same-condition ablation must complete before choosing the frozen
material-adaptive Ours rule. The 40-image held-out test must remain unopened until that
rule is frozen. FLUX Kontext and ChronoEdit are documented as unavailable rather than
assigned fabricated results.
