# E6 negative-optimization diagnosis and E7 repair

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-19
- Verification Status: measured outputs and integrity checks verified; visual judgments unblinded
- Version Label: e7_cavity_fix_v2_result

## Outcome

The conspicuous white triangle inside the cake wall was an input-rendering bug
amplified by E6 hard latent replacement. It is eliminated in both corrected
E7 runs. The soft version has a less rigid wall/frosting transition than the
corrected hard version on this example. Full photographic quality is still
not achieved: the cavity floor has a flat colored-patch appearance, shadows
remain imperfect, and the payload retains inherited contact artifacts.

The earlier statement that the failure necessarily requires soft or low-frequency
guidance was too strong. Correcting the renderer while keeping hard projection
already fixes the white-triangle defect. Soft guidance is a separate tested
refinement. No low-frequency latent filtering was implemented or tested.

## Causes traced to code and pixels

1. **Mixed-material donor.** `build_e5_bite_remain_consistency_v1.py` cropped
   rows 250:350 and columns 115:285 as crumb texture. The crop crosses the cake
   edge onto the plate. Of 17,000 donor pixels, 6,383 (37.55%) have HSV S<42;
   this rule is diagnostic and is not semantic ground truth. The entire patch
   was perspective-warped to each wall, including its plate pixels.
2. **Reference errors enforced with weight one.** E6 replaced all supported
   latent values with noisy reference values on steps 0--7. Its reference
   already contained the white triangle. In a fixed 5,558-pixel visible wall
   region, low-saturation fraction is 44.03% in the reference, 40.30% in E6,
   and 0% in E5b without latent replacement. The controlled repair preserves
   hard replacement and removes the defect after changing donor/rendering.
3. **Artificial shadow edge.** The old renderer dimmed a binary dilated wall
   neighborhood by 10%, seeding a straight shadow band. E7 uses continuous
   distance falloff on visible floor instead.
4. **Wrong preservation reference for inherited artifacts.** Local repair
   preserves the E3 generated baseline outside its alpha. That baseline already
   has a grey polygon caused by retaining generated background throughout a
   static swept-action support. A zero difference to that baseline is not
   equivalent to preservation of the original photograph.

Annotated source evidence:
[donor and control images](../artifacts/e7_cavity_review_v1_20260919/donor_root_cause.png).

## Implemented changes

- `scripts/cavity_state_projection.py`: largest all-material donor-square
  selection with fail-closed behavior; approximately scale-preserving reflected
  texture extension; smooth local floor shading; optional bounded, feathered
  projection with weight `0.35*(1-k/8)^2`, k=0..7, then no replacement.
- `scripts/prepare_e7_cavity_fix.py`: hash-verified derivation of new control
  assets from frozen E5b; original reference, prompt, seed, masks and geometry
  retained except for corrected cavity RGB appearance.
- `scripts/prepare_e7_runtime.py`: private copy of frozen runtime with an exact,
  optional projection-site patch. Default hard replacement is unchanged.
- `scripts/run_e7_cavity_repair_v1.py`: resource gate, paired hard/soft execution,
  per-step soft-projection trace, timeout and foreign-process protection.
- `scripts/original_reference_compositor.py`: separate postprocessing restores
  the original photo outside current-frame motion/repair support, with a
  12-pixel margin and 8-pixel inward feather. The operation core is unchanged.
- `scripts/review_e7_cavity_fix.py`: hash verification, protected-pixel checks,
  fixed-region diagnostic and uniform crops for all four historical/current arms.

The pure donor is 32x32 at x=251:283, y=295:327. Its mask is a case-specific
color/ROI rule, not a general food segmentation model. No model training,
prompt editing, motion replanning or model downloading was performed.

## Controlled results

All runs use seed 1, 21 frames, 20 steps, VACE scale 1 and no LoRA. E7 hard
and soft configuration files differ only in `variant`. E6 versus E7 hard
changes the material/rendering package; it does not isolate donor and shadow
changes from one another.

| Output | Visible-wall low-saturation fraction | Visual inspection |
|---|---:|---|
| E5b, old reference, projection off | 0.00% | Empty notch, angular diagonal/shadow remains |
| E6, old reference, hard | 40.30% | White triangular wall contamination, rigid floor |
| E7, corrected reference, hard | 0.00% | Contamination removed; regular texture/edges remain |
| E7, corrected reference, soft | 0.00% | Contamination removed; softer wall transition; floor still imperfect |

This fraction measures the identified contamination only. It is not Action,
Photo Success, physical correctness, or a statistical efficacy score.

[Four-way final crop](../artifacts/e7_cavity_review_v1_20260919/final_cavity_comparison.png)
and [fixed-frame timeline](../artifacts/e7_cavity_review_v1_20260919/cavity_timeline.png)
show the unmodified paired outputs. They intentionally retain the same old
compositor so the neural-generation comparison is not confounded by cleanup.

The separate original-reference compositor removes most of the large swept
background polygon in both E7 arms. In all 21 frames it preserves the operation
core exactly and restores original pixels exactly outside dynamic support.
Narrow halos near the moving utensil/local repair may remain. This is an explicit
compositing guarantee, not a learned background-preservation improvement.

[E7 soft with background restoration](../artifacts/e7_soft_original_background_v1_20260919/final.png)
is the current development candidate, not a paper-quality success case.

## Verification

- Five new regression tests plus seven existing geometry/schedule tests pass.
- Compilation succeeds for all six new scripts.
- Both GPU runs complete on gp40 physical RTX A6000 GPU 0 after separate gates.
- One pipeline load and 21 raw + 21 projected PNG frames per run.
- Hard: 260.40 s; soft: 240.31 s.
- 49/49 file records per new run verified locally (98/98 total).
- Remote/local file-manifest hashes match:
  - hard: `182136ef8c80d454f51785e2fa1981faf909740a13c37d875e92f1a97ea02449`
  - soft: `183f495050afd238ea86f66bb4af9f8fd0e2d27f97e82a2b3b67d9af56776454`
- Soft trace contains steps 0..7, each with 549 nonzero latent mask positions;
  maximum weight declines from 0.35 to 0.00546875, then projection stops.
- Local-repair outside-alpha and planned-payload pixel differences are zero.
- Both separate background-restoration outputs pass original-background and
  operation-core invariants in all 21 frames.

Machine-readable checks:
[verification.json](../artifacts/e7_cavity_review_v1_20260919/verification.json).
Protocol: [E7 protocol](E7_CAVITY_FIX_PROTOCOL_20260919.md).

## Reproduction and locations

Build fresh local controls:

```text
python scripts/prepare_e7_cavity_fix.py --output-root <NEW_LOCAL_BUNDLE>
```

Transfer the scripts, original base config and new controls to a fresh remote
bundle, verify hashes, then run `prepare_e7_runtime.py --bundle <NEW_BUNDLE>`.
After resource auditing, run the hard and soft configs sequentially:

```text
python run_e7_cavity_repair_v1.py --config e7_<hard-or-soft>_remote.json --controls controls --base-config e7_runtime_config.json --gpu <SAFE_PHYSICAL_GPU> --output-root <NEW_OUTPUT> --preflight-report <NEW_REPORT>
```

Use `review_e7_cavity_fix.py` with `--hard`, `--soft`, `--controls`,
`--output-root`, then `original_reference_compositor.py` with `--run`,
`--controls`, `--proxy-controls` and a new `--output-root`.

Remote bundle:
`gp40:/host/space0/guo-z/tf-ufi/experiments/e7_cavity_fix_v2_20260919T030209Z`.
Remote outputs:
`gp40:/host/space0/guo-z/tf-ufi/outputs/e7_{hard,soft}_gp40_20260919T030209Z`.
Background restoration was executed locally and is not present in those remote
raw-run directories.

The original E5/E6 evidence and frozen runtime were preserved. This one synthetic
development case establishes a reproducible defect correction; it does not
establish generalization, 3D superiority, photo-realistic source conservation,
or formal human-evaluation results. Floor material/shadow consistency is still
an open failure mode and should be tested as a separate surface-specific
constraint rather than increasing the global projection strength again.
