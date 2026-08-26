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
