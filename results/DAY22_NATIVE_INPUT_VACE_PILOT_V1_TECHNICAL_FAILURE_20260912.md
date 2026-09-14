# Day 22 native-input VACE pilot v1: technical failure

## Outcome

No model inference ran. The runner stopped before resource preflight and before
creating an output directory because the frozen configuration contained a
mistyped dataset-manifest SHA-256.

## Frozen attempted condition

- Case: `soup`
- Native input: `reference.png` repeated for all 21 VACE control frames
- Edit support: the existing `edit_alpha.png` repeated for all 21 frames
- Excluded: planar and relative-3D utensil/food motion proxies
- Seed: 1
- Frames: 21
- Inference steps: 20
- VACE scale: 1.0
- LoRA: off
- TTM: off
- Requested physical GPU: gp40 GPU 2 (NVIDIA RTX A6000)

## Failure

The v1 configuration recorded:

`7a71eb2c63bf52cbe2275b1b1aceca028c045fd9fdf6328a8ed12b3a5e56078`

The verified local and gp40 dataset manifest is:

`7a71eb2c63bf52cbe22705d1b1aceca028c045fd9fdf6328a8ed12b3a5e56078`

The missing `0` after `...be227` caused the fail-closed validation error.
The dataset itself did not change.

## Preservation and safety checks

- gp40 dataset manifest size: 12,156 bytes
- gp40 and local manifest SHA-256 matched after diagnosis
- gp40 and local soup reference SHA-256:
  `dfa5d3870d0ae49177f3b7cd9cd2aa23c9ef4b6607f0ce5b7546e960d0df3eff`
- gp40 and local soup edit-mask SHA-256:
  `ce15d4a7af43dfc3aab8ff90ad957da272eb4b14295f9c9f6a15a86e25ffe46f`
- Intended v1 output root remained absent
- Intended v1 preflight report remained absent
- No model was loaded and no GPU inference was performed
- Uploaded v1 config and runner bundle remains under
  `/tmp/foodstateedit_day22_native_input_vace_pilot_v1`

## Next action

Create a corrected v2 freeze with new upload, output, and preflight paths. Run
only after a fresh all-host resource audit and explicit retry authorization.
