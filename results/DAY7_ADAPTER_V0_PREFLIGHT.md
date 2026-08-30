# Day 7 VACE-LoRA adapter v0 preflight

## Decision

Freeze a two-sample, high-noise VACE-LoRA run as an infrastructure smoke test.
It may establish that the existing offline model can load the FoodStateEdit
paired-control format, backpropagate through the VACE branch, and save a LoRA
checkpoint. It cannot establish visual improvement, action generalization, or
photo realism because its targets are identical to deterministic proxy controls.

## Trainer selection

- Framework: official DiffSynth-Studio Wan2.2 VACE-LoRA trainer.
- Frozen commit: `899d2cd5740f50f7b6a1ec2cf7360a20897b1191`.
- Source archive SHA-256:
  `68344f19db71b4e8b0616d0a3aec436fddcc545d9ce03396ad797f29a1b32228`.
- Training entry-point SHA-256:
  `4a8c9c27c23b217f7438b37c71213df58414c39cb2b6998d3d75a4dd7992bc41`.
- Selection reason: the newest inspected upstream snapshot failed under the
  existing PyTorch 2.4 environment because it imported
  `torch.nn.attention.flex_attention`. The frozen earlier commit retains the
  official Wan2.2 VACE-LoRA path and passed an offline `--help` import smoke.
- Existing GeoEdit worktree was not modified.

## Frozen dataset

The dataset contains two 21-frame plumbing-only samples:

| Sample | Family/action | Utensil | Target policy |
| --- | --- | --- | --- |
| `ramen_chopsticks_001_proxy_identity_smoke` | strand/lift | chopsticks | proxy target equals VACE control |
| `pasta_fork_001_proxy_identity_smoke` | strand-contact/twirl-and-lift | fork | proxy target equals VACE control |

- Dataset manifest SHA-256:
  `5483f03bd79295f62cc961c3411198fe4c221a0979dbf833f76ba28c6f97b9ae`.
- Metadata SHA-256:
  `9e4c4fb663a01707cf91aa5bdb24c0b2b392e802f47dea37d73958df257c7652`.
- Every media file records a byte size and SHA-256 in the manifest.
- Builder provenance uses SHA-256 after CRLF-to-LF normalization so the same
  committed source is verifiable on Windows and Linux.
- Preflight requires metadata paths to agree with the manifest, validates every
  media size and hash, and confirms that each identity-smoke target/control pair
  is byte-identical.
- A failed initial materialization caused by selecting the wrong manifest root
  is preserved in the ignored local artifact directory
  `artifacts/day7_adapter_v0_smoke_dataset_failed_missing_manifest_20260829`.
- The first transferred freeze (`7fe820c`) is also retained locally and remotely:
  archive extraction converted source line endings, exposing that its raw
  builder hash was not cross-platform. It was not extracted as a runtime and no
  training was started. The corrected dataset uses a new `_v2` directory.

## Frozen run and safety gate

- Base: existing audited `Wan2.2-VACE-Fun-A14B` high-noise model; downloads are
  disabled.
- Adapted branch: `vace`; LoRA modules `q,k,v,o,ffn.0,ffn.2`; rank 8.
- Schedule: one epoch, two samples, one repeat, one GPU, BF16, two optimizer
  steps expected.
- Output: new-only
  `/tmp/foodstateedit_day7_adapter_v0_high_noise_lora_smoke_v1`.
- GPU gate: at least 48,000 MiB free, at most 5% utilization, and no compute
  process on the selected A6000.
- Host-memory gate: at least 80,000 MiB available.
- Any preflight or training failure is preserved; no existing directory or
  other user's process may be changed.

## Initial resource observation

On 2026-08-29, all eight A6000 GPUs were occupied by another user's eight-GPU
job at roughly 44.5 GiB per card and near 100% utilization. No adapter training
was launched. A fresh remote preflight is required after the exact dataset and
code snapshot are transferred.

## Remote preflight outcome

The corrected snapshot was transferred and checked on `gp40`:

- FoodStateEdit commit: `29bf716`.
- Code archive SHA-256:
  `e347e589c8f89f144fbd3303762c38add4938cee895e348dfe788448211ddfbf`.
- Config SHA-256:
  `1ce8d8965a571fd7d30733c0dfd58e0ee0d3c04f71c4b357272cf89f42d89023`.
- Preflight report SHA-256:
  `1eb8ef0751ef668910e9cc33218a3121328afbff727b4a40b2db91d15194bf74`.
- Result: 51 of 52 checks passed. The sole failure was `gpu_gate`.
- Available host memory was 209,950 MiB. All eight A6000s had only 3,904--4,014
  MiB free and 99--100% utilization. Each card was owned by one process from
  user `chen-q` under `HybridHOT`.
- The training output path remained absent. No process was interrupted and no
  checkpoint was produced.

## Generalization gate after smoke

A successful two-step run only opens the data pipeline. The first publishable
adapter candidate requires a disjoint paired-photo dataset covering all four
families (liquid, granular, strand, strand-contact), multiple dishes and camera
conditions per utensil/action, held-out food instances, and separate reporting
of action correctness, photo realism, source removal, and protected-region
preservation. Deterministic geometry and final 2D protection remain outside the
LoRA.

## Multi-host execution and preserved v1 failure

After expanding the resource search to `gp38`, `gp39`, `gp40`, `gp41`, and
`gp42`, `gp39` was the only machine satisfying the frozen A6000 gate: all eight
cards reported 48,539 MiB free, 0% utilization, and no compute process. The
other A6000/Blackwell hosts were occupied; `gp41` provides A40 cards whose total
memory cannot satisfy the 48,000 MiB free threshold.

The `gp39` preflight passed all 52 checks and selected physical GPU 0. The
upstream training script then exited before loading any model because it
unconditionally constructed `LoadAudio`, whose initializer imports the absent
optional package `librosa`. This is unrelated to the FoodStateEdit media, which
contains only `video`, `vace_video`, and `vace_reference_image` fields.

- Failed run manifest SHA-256:
  `39b273c1f00787c408407357183124161bc2622db149281a8af5daa834a0e234`.
- Passing `gp39` preflight SHA-256:
  `feb1c5d3e36e81a7df566ed78747a3a1d4e3e05a905aabc38d9f9c88c92d8bf2`.
- Failure log SHA-256:
  `c5cd99dee38a7a06ad55d6a167a4f951c5b6ff0cb8205b83fb17ca77197a5cbf`.
- Checkpoints: zero.

The failure directory remains untouched. The v2 fix does not install or
download `librosa`; a frozen wrapper rejects any dataset containing
`input_audio`, provides a sentinel only for the upstream unused import, and
then runs the unchanged, hash-verified trainer. Preflight verifies both the
wrapper hash and its offline `--help` path.

The first wrapper preflight was also preserved. It failed only
`no_audio_wrapper_help` because Transformers calls `find_spec("librosa")` and a
plain sentinel module has no import specification. Report SHA-256:
`f16496557337cc491620e305b1049a407031bd8fbccce8c28ec653cb5ab65efb`.
No output directory was created. The corrected wrapper assigns a standard
`ModuleSpec` to the sentinel and is frozen under a new source commit.

The next complete v2 launch passed every preflight check, then exposed one more
constructor-only access: `LoadAudio.__init__` stores `librosa.load` even when no
audio field is present. It failed before model loading and produced no
checkpoint.

- V2 failed run manifest SHA-256:
  `ecd105fc088666eafd58e93b8bd04af8e1e139f3e9f066fcb98e9ee124760a3a`.
- V2 passing preflight SHA-256:
  `09d9fb748fdaffb3d40638af918c9f10ba3a172db22488f28287a9fb1e126636`.
- V2 failure log SHA-256:
  `23cccd67ddd536f540510fb56c1637b01d7f9dc0da7051379a38c11fd901a28c`.

V3 provides a `librosa.load` callable that immediately raises if invoked. The
wrapper still rejects `input_audio` in `--data_file_keys`, so this callable can
only expose a contract violation; it cannot silently process audio. V3 uses the
new-only output `/tmp/foodstateedit_day7_adapter_v0_high_noise_lora_smoke_v3`.

V3 passed all preflight checks and loaded the frozen DiT, VACE, T5, and VAE,
then failed on the first dataloader item. ImageIO 2.37.2 selected PyAV; its
container-level `get_meta_data()` attempted to multiply a missing duration by a
stream time base and raised `TypeError`. Direct first-frame decoding and the
stream's rate/count are valid, so this is a decoder metadata compatibility
failure rather than corrupt media or a model/LoRA failure.

- V3 failed run manifest SHA-256:
  `dee053bacd170bfed98831cb505c47d11d48a73b156c47df2aa040dbcf679cba`.
- V3 passing preflight SHA-256:
  `5c8176a9dbb78d9f20a83b768944f23a4b9d0eeb540477538402451cfb3caa45`.
- V3 failure log SHA-256:
  `a698a927b52f826e012eb09a814c83a0d80ceb234b4599a8f1534c45b11df65c`.
- Checkpoints: zero.

The first V4 decoder-only smoke caught another API mismatch before model load:
ImageIO's PyAV `LegacyReader` has no `count_frames()` method, although its
stream reports the frozen count of 21. It created no training output. V5 keeps
PyAV's actual `get_data` frame decoder untouched and supplies the missing
`count_frames()` method plus video-level `fps`, `duration`, and `nframes` from
that same stream. Preflight runs this exact wrapper path over the first and last
pixels of every target/control MP4 and requires the frozen 21-frame count before
any model is loaded. V5 uses the new-only output
`/tmp/foodstateedit_day7_adapter_v0_high_noise_lora_smoke_v5`.

## Completed v5 infrastructure smoke

The frozen v5 run passed every preflight check on gp39 and selected physical
GPU 0. It completed one epoch over the two identity-target samples in 99.095
seconds and exited with code 0. Because `save_steps=1`, the upstream trainer
created `step-1.safetensors` and `step-2.safetensors`; the earlier expected
`epoch-0.safetensors` name was not part of the actual trainer contract.

- Runtime source commit: `18d4d25782a7ae580f3e4bb658a2d8bf9b6e3a38`.
- Runtime archive SHA-256:
  `a72ce0c64b80d084b37dae3897a39ee330c413d94800ca7b60af7f647f323b30`.
- Config SHA-256:
  `6b5f32bc7261d14679a119d3335b541f1cea99d82e96be35c9f3f12b673a2682`.
- Passing preflight SHA-256:
  `4bc6fec39892f47438d26cd8e742963ae097ed59ac9a012a68245ea51019cc59`.
- Complete run manifest SHA-256:
  `5ba4f706d555701e37e897bb827e5904cb8a9a0d3a81f191bc66397c62170257`.
- Final `step-2.safetensors` SHA-256:
  `b01efb7dd7734a17cd8c4e983f616e3e01d483e44de19f38be26ecebe57a4465`.

The offline checkpoint validation completed against the final step: 160 finite
BF16 tensors, 80 complete LoRA A/B pairs, rank 8, and 80 tensors updated by the
official `pipe.load_lora(pipe.vace, ...)` loader. Validation-report SHA-256:
`f44e5caea2dff6cd4ad3b6d947b6a040b826bf1b1e0d932dc6e5f3d949ddcbfa`.
This is infrastructure evidence only. It does not establish action success,
food-family generalization, or photo realism.
