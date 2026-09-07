# Day 16 geometry-prompted SAM3 observer pilot

Date: 2026-09-07. Status: partial strand signal; overall observer gate failed.

## Bottom line

Frozen relative-3D boxes and positive/negative points fixed the most important
Day 15 strand failure. The SAM3 mask localized the scaffold's lifted noodle
segment rather than the full bowl, and seven of nine frozen gates passed.
Nevertheless, SAM3 still merged the two chopsticks, so the observer must not be
connected to VACE guidance.

The matched existing Day 13 images contain a useful preliminary signal:
both planar outputs had lifted-strand IoU 0, while all three relative-3D outputs
were between 0.695 and 0.728. This supports only a scaffold-conditioned,
seen-sample diagnostic that relative-3D conditioning preserved the intended
flexible strand region better than the planar condition. It does not establish
VACE effectiveness or a topology-loss gain.

## Frozen design

- one aligned seen udon source and the frozen relative-3D final-hold control;
- the five existing matched Day 13 final-hold outputs; no new VACE inference;
- one strand box and two per-stick boxes, each with frozen positive and negative
  points derived from the relative-3D scaffold;
- source, strand-erasure, stick-erasure and solid-strand-block controls;
- scaffold-derived masks, explicitly not independent annotation ground truth;
- nine all-or-nothing gates frozen before SAM3 inference.

Config SHA-256:
`1c5c73a158ae82ff5e84c1c13cf2aba43c85515e0d919fe67649579fb608634a`.
Completed runner SHA-256:
`18c2f2c075caa60c5ba56ed3432331f6cf56f4379752aeba48cf645612b888b9`.

## Gate result

| Frozen gate | Pass |
|---|:---:|
| Geometry control strand aligns | yes |
| Geometry control near stick aligns | yes |
| Geometry control far stick aligns | yes |
| Geometry control stick masks are distinct | **no** |
| Geometry control strand beats source | yes |
| Strand erasure lowers alignment | yes |
| Stick erasure lowers alignment | yes |
| Solid block is not preferred | yes |
| Weighted output has all three distinct structures | **no** |

The geometry-control predicted stick masks had IoU 0.5295, above the frozen
maximum 0.50, and centroid distance 7.30 px, below the frozen minimum 8 px.
For the topology-weighted output the masks overlapped even more (IoU 0.6831),
so two separate chopsticks were not established.

## Strand comparison

| Existing condition | Strand IoU |
|---|---:|
| Planar, LoRA off | 0.0000 |
| Planar, uniform step 32 | 0.0000 |
| Relative 3D, LoRA off | 0.7284 |
| Relative 3D, uniform step 32 | 0.6954 |
| Relative 3D, topology weighted step 32 | 0.7246 |

The control contrasts were also strong: geometry control 0.7792, unedited
source 0.3188, strand-erased control 0.2988 and solid block 0.2667. The visual
overlays agree: the relative-3D cases select the short lifted segment at the
specified location; planar cases either return no eligible strand mask or a
mask elsewhere in the bowl.

The topology-weighted arm is only +0.0292 over relative-3D uniform and -0.0038
against relative-3D LoRA-off. Therefore the current data support a preliminary
representation signal, not a learned topology-loss contribution.

## Execution and evidence

The authoritative run used gp38 physical GPU 0 after a fresh A6000 preflight:
48,539 MiB free, 0% utilization, zero compute processes and about 241 GiB host
memory available. GPU memory returned to 48,539 MiB after the run. No process
was terminated, no model was downloaded and no VACE inference ran.

The first launch failed before model load because the requested preflight parent
directory did not exist; it created no output. The retry used a new output and
preflight path. The remote result is
`/host/space0/guo-z/tf-ufi/outputs/day16_geometry_prompted_sam3_gp38_20260907T1124Z_v3`.
All 68 manifest-covered files matched locally in size and SHA-256. Raw result
SHA-256 is
`e8726a1975c44b5d44eced86f91f3768fd54df49914382407e2acb35e2efa9aa`.

Local masks, overlays, controls and raw records are in
`artifacts/day16_geometry_prompted_sam3_20260907`.

## Decision

Do not use SAM3 guidance yet. For the report, the defensible statement is:
relative-3D conditioning shows a strong preliminary, geometry-aligned signal
for the lifted flexible strand on this seen synthetic sample. Before claiming
effectiveness, freeze independent strand-centerline and per-stick instance
labels, validate on held-out synthetic controls, and require the two-stick gate
to pass. The learned topology-weighting branch remains unsupported.
