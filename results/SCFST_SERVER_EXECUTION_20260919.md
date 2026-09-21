# SCFST server execution — core contract only

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-19
- Verification Status: VERIFIED for uploaded-file identity, software tests and synthetic control output only
- Version Label: scfst_server_contract_v1

## Experiment Result

- **ID**: `scfst_contract_v1_20260919T112315Z`
- **Type**: generic CPU software-contract validation
- **Status**: completed
- **Host**: `gp40.cs.uec.ac.jp`
- **Python**: `/host/space0/guo-z/envs/geoedit/bin/python` (Python 3.12.12)
- **Environment**: NumPy 1.26.4, PyTorch 2.4.0+cu121; `CUDA_VISIBLE_DEVICES=` during execution
- **Remote working directory**: `/host/space0/guo-z/tf-ufi/experiments/scfst_contract_v1_20260919T112315Z`
- **Timeout**: 180 seconds per command
- **Exit Code**: 0

Commands executed after seven source files matched local SHA-256 values:

```text
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_state_transfer.py -v
CUDA_VISIBLE_DEVICES= python scripts/demo_state_transfer.py --output-root outputs/scfst_contract_demo_v1
```

## Results

- 19/19 state-transfer tests passed in 2.021 s; wrapped process time 5.40 s.
- Test peak RSS: 494,636 KiB.
- Synthetic demo completed in 1.07 s; peak RSS 46,456 KiB.
- Produced 21 control frames with shape `[21, 48, 64, 10]`.
- All 336 source parcels retained unique IDs.
- Four role channels sum to one at every pixel.
- The `controls.npz` hash agrees with the hash embedded in its remote manifest.
- Post-run gp40 audit: all eight RTX A6000 GPUs reported 48,539 MiB free, 0% utilization, and no compute applications. Host `MemAvailable` was 256,657,612 kB.

## Retrieved Evidence

Local copy: `artifacts/scfst_server_gp40_20260919T112315Z/`

| File | SHA-256 |
| --- | --- |
| `test.log` | `aca207ae4e7eb23fa090140a045b56365f1adf2591e8ea70f79b86f429332dc7` |
| `demo.log` | `9fa8eee5515aa3aa8f2dff48ee9f51205adb560c1860fbf3f29199469cb300ab` |
| `scfst_contract_demo_v1/manifest.json` | `c917d52defffb27b3bae939759a34a5279284fa6896bd431f87fa8732fad6f47` |
| `scfst_contract_demo_v1/controls.npz` | `5f20f8d6a224688a2ecf2eac99b3560d04eb6d96504f06e8f3224d43d6d6d3e1` |

## Scope and Open Boundary

This run validates executable data-flow contracts on the server: unique source
identity, joint state updates, visibility handling, token-layout conversion,
zero-initialized routing behavior, gradients, and a synthetic 21-frame control
sequence. The transition trajectory is prescribed. No VACE/Wan hook was enabled,
no adapter checkpoint was trained, no GPU inference was performed, and no edited
food image was generated. This evidence therefore does **not** establish image
quality, material dynamics, cross-food generalization, improvement over VACE,
or source-consistent behavior in generated pixels.
