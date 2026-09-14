# Day 22 publication-gap closure plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan-to-run
- Origin Date: 2026-09-12
- Verification Status: PARTIALLY VERIFIED
- Version Label: publication_gap_closure_v1

## Paper-level research question

For single-image food manipulation editing, does a structured relative-3-D
interaction control with material-adaptive diffusion strength improve action,
contact and source-consistency over native-input, planar-control and GeoEdit
baselines while retaining photographic appearance and exact protected-region
preservation?

This is the only primary question. Cake remains a disclosed synthetic
supplementary demonstration; it does not silently replace the frozen real
strand-contact family in the 60-image benchmark.

## Evidence already available

- Reproducible 4-anchor, seed-1 common-backbone comparison.
- Three one-image soup/rice/cake planar-versus-relative-3-D pilots.
- One strict final-image pass for cake at VACE scale 0.6, but zero of three full
  temporal-action passes.
- One seen-synthetic strand-region signal for relative-3-D controls, without
  independent ground-truth annotation.
- Exact protected-region compositing and hash-traceable run evidence.

These are development diagnostics, not formal effectiveness evidence.

## Phased experiment matrix

### Phase A - baseline completion

| Arm | Scope | Seeds | Status | Gate |
| --- | --- | --- | --- | --- |
| Input/no edit | soup pilot first | deterministic | available | identity control |
| Native-input VACE | soup pilot first | 1 | v2 frozen, pending execution | determines prompt-only capability |
| Planar proxy VACE | same soup | 1 | available | matched 2-D control |
| Relative-3-D proxy VACE | same soup | 1 | available | matched 3-D control |

The native-input arm repeats the source image for all 21 control frames and
uses the same prompt, edit support, seed, inference budget and final projection.
It supplies no utensil or payload motion proxy.

### Phase B - unique-method pilot freeze

Candidate method: relative-3-D control plus material-adaptive VACE strength and
predeclared semantic rollback.

Pilot-only candidate strengths are 1.0, 0.8 and 0.6. The final rule must be
frozen before any test image is generated. Per-image manual selection, seed
replacement and test-set threshold tuning are prohibited. A validated semantic
observer or blinded pilot labels must establish utensil identity, contact,
payload, source change and duplication before an automatic rollback rule is
accepted.

Pilot scope: the existing five real pilot images in each of liquid, granular,
strand and strand-contact, with seeds 1, 2 and 3. The pilot may select one
material-level rule; it may not create case-specific parameters.

### Phase C - locked held-out test

- 40 frozen real test images: 10 per original family.
- Seeds 1, 2 and 3 for every stochastic method.
- Fixed prompts, dimensions, controls, frame indices and scoring rules.
- No case removal, output-dependent rerun, best-seed reporting or post-test
  schedule adjustment.
- Cake is reported separately as synthetic supplemental evidence.

Minimum full-table methods:

1. native-input VACE;
2. Vanilla GeoEdit;
3. planar proxy VACE;
4. fixed-strength relative-3-D VACE;
5. frozen adaptive/rollback FoodStateEdit;
6. at least one reproducible strong external editor on identical inputs.

Qwen-Image-Edit, FLUX Kontext and ChronoEdit remain missing until complete
local weights, versioned runtime and same-input outputs are verified. Models
must not be downloaded merely to fill a table without an explicit decision.

## Primary endpoints

The unit of analysis is the frozen input image. Seeds are repeated stochastic
observations nested within image, not independent images.

- `ActionSuccess`: correct utensil count/type, valid payload, visible support or
  grip, consistent source reduction and no forbidden duplicate.
- `PhotoSuccess`: plausible material, local lighting/reflection/shadow, source
  repair and absence of obvious diffusion artifacts.
- `PreservationSuccess`: zero maximum pixel difference outside the declared
  alpha before lossy video encoding, plus accepted feather transition.
- `StrictE2ESuccess = ActionSuccess AND PhotoSuccess AND PreservationSuccess`.

Primary comparison: paired StrictE2ESuccess of frozen FoodStateEdit versus the
strongest reproducible baseline on the 40 held-out images. Report the paired
absolute difference, clustered bootstrap 95% confidence interval and the raw
success counts. Do not promote a p-value without effect size and interval.

## Secondary endpoints

- Utensil identity/count accuracy.
- Contact/support accuracy.
- Source-reduction consistency.
- Duplicate-payload failure rate.
- Five-point blinded photo-realism rating.
- Temporal phase-order success and contact persistence.
- Runtime per output, peak VRAM and required human correction time.
- Family-stratified results; no universal material claim from a pooled score
  alone.

## Evaluation and statistics

- At least two independent blinded reviewers; three are preferred.
- Randomized method/seed ordering and concealed method names.
- An explicit `uncertain` response is retained and never converted to pass.
- Report per-item decisions and reviewer agreement. Use Krippendorff's alpha or
  an appropriate categorical agreement statistic with confidence intervals.
- Use image-clustered bootstrap intervals so multiple seeds from one image do
  not inflate the sample size.
- Freeze all primary/secondary endpoints and multiplicity handling before Phase
  C.

The companion protocol is
`benchmark/FOODSTATEEDIT_BLIND_EVALUATION_PROTOCOL_V1.md`.

## Required paper tables and figures

1. Main held-out comparison table with all primary endpoint counts/rates.
2. Ablation table: native, planar, fixed relative-3-D, adaptive/rollback.
3. Efficiency table: runtime, peak VRAM and human correction time.
4. Fixed qualitative grid: Input, each baseline, Ours; no per-method best frame.
5. Contact-to-lift-to-final sequence for representative held-out cases.
6. Failure grid for liquid spill, utensil drift, duplicate payload and broken
   strand contact.

## Execution gates

1. Native-input soup v2 must technically complete and be reviewed.
2. The native/planar/relative-3-D comparison determines whether an explicit
   motion proxy is necessary for the soup pilot.
3. Phase B runs only after the observer/ballot contract is fixed.
4. Phase C remains locked until the unique Ours rule and all baselines are
   frozen and reproducible.
5. A negative pilot is preserved and may motivate redesign; it cannot be
   converted into a positive result by changing seeds or excluding cases.

## Claim boundary

Until Phase C and blind review finish, the repository supports a reproducible
development system and failure analysis. It does not support superiority,
generalization, physical correctness, fully automatic operation or universal
photo realism.
