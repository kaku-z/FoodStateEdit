# Day 19: four-material matched pilot launched

This is a **launch snapshot, not an effectiveness result**. On 2026-09-08 at
12:46:30 UTC, the frozen eight-condition job passed resource admission on gp40
physical GPU 5. At the verified snapshot the pipeline was loaded once and
`soup__planar` was active; zero conditions had completed. Inspect the remote
run manifest for newer status rather than assuming this snapshot stays current.

## Frozen sequence

1. Soup: planar, relative3d.
2. Fried rice: planar, relative3d.
3. Pre-cut cake bite: planar, relative3d.
4. Noodle lift: planar, relative3d.

Every condition uses seed 1, 21 frames, 20 steps, VACE scale 1, LoRA off and
TTM off. The existing Wan2.2-VACE-Fun-A14B runtime is used offline. One pipeline
serves all conditions serially; this is one admitted workload, not eight
parallel GPU processes. Foreign GPU processes are checked before and after
each condition; their appearance aborts our batch without terminating them.
The initial gate is a snapshot, not an exclusive reservation.

Each image's arms share the source, prompt, negative prompt, resolution,
seed, model, step count, full swept edit mask and final compositing policy.
For soup/rice/cake the intervention is fixed planar translation versus
manually parameterized relative-3D perspective/depth motion. For noodle it is
fixed paint order versus depth-varying visibility along the same strand.
Consequently this is not one uniform, isolated 3D treatment across classes.
No reconstructed geometry, learned improvement or physical mass conservation
is established by these controls.

## Preparation evidence

- All 37 uploaded dataset files and three runtime/config files matched local
  SHA-256, with zero mismatches.
- All four cases have zero uncovered changed control pixels. Shared support
  fractions: soup 17.50%, rice 23.21%, cake 18.95%, noodle 12.58%.
- Full local suite: **177 tests passed, zero failures/errors/skips**.
- The first two local packages are retained. v1 had an unused precontact
  divide warning, a radial rice inpaint smear and incorrect cake-floor sampling;
  v2 still sampled a dark cake/shadow region for the exposed plate. v3 fixes
  these before VACE execution. No unsuccessful output was selected away.
- The source reduction and cake cut faces are coarse procedural hypotheses;
  they are control rasters, not target photographs or physical truth.

Preflight: RTX A6000, 48,539 MiB free, 0% utilization, zero compute processes
on GPU 5, 248,269 MiB available host memory. gp38/gp39/gp40/gp41/gp42 were
audited; A40/Blackwell were excluded. Other users' processes were not changed.

## Provenance and claim limits

Soup and rice use previously studied, upscaled real UECFOOD pilot inputs;
their source images and derived comparison pictures must not be publicly
redistributed under the repository's source policy. Noodle uses the existing
synthetic input. Cake is a **new synthetic INPUT**, not a FoodStateEdit output.
Built-in imagegen produced a cream-topped sponge cake with a pre-cut bite,
still in place and no utensil. No generated target or before/after solution
was supplied to VACE.

Cake saved input:
`artifacts/day19_multimaterial_sources_v1/cake_imagegen_input.png`.
Exact generation prompt:
`artifacts/day19_multimaterial_sources_v1/cake_input_prompt.txt`, also embedded
in the tracked [machine-readable launch record](day19_multimaterial_launch_20260908.json).
The original generated image remains in the Codex generated-images directory.

All four classes must be reported separately, including failures. One source
and one seed per class are exploratory examples, not generalization or novel
diffusion-algorithm evidence. Ambiguous contact or photographic appearance
cannot be automatically counted as a pass. Automatic heartbeat remains paused.

## Resume / recovery paths

- Config: `configs/day19_multimaterial_pilot_v1.json`.
- Config SHA-256: `ea8c0429a585d7f3c92b89d4eedd21ae5c706a1786c699c578465721ee643ac6`.
- Dataset manifest SHA-256: `7a71eb2c63bf52cbe22705d1b1aceca028c045fd9fdf6328a8ed12b3a5e56078`.
- Preflight SHA-256: `ee5c22de4372b528fc82238b0aab3bb0210217abe34e661713821bcd54ef6088`.
- Runtime: `/host/space0/guo-z/tf-ufi/runtime/day19_multimaterial_v1`.
- Data: `/host/space0/guo-z/tf-ufi/outputs/day19_multimaterial_dataset_v3`.
- Output: `/host/space0/guo-z/tf-ufi/outputs/day19_multimaterial_gp40_v1`.
- Inspect `run_manifest.json`, condition manifests, `COMPLETE`/`FAILED` and
  logs; do not infer completion from PID or file existence alone.

After completion, recover all files to a new local directory, compare hashes,
decode all eight 21-frame outputs, and review native/projected contact sheets
per material. Native and encoded-video outside-support differences must be
reported separately from exact compositing before encoding. A retry requires
fresh output and preflight paths, keeping partial results and failures intact.
