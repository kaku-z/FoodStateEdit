# Day 8 same-seed VACE 3-D-control comparison preflight

## Frozen question

Does replacing only the Day 6 planar pasta/fork control with the Day 8
depth-aware 3-D projection improve visible fork-twirl contact under the same
VACE model, prompt, negative prompt, seed `1`, 21-frame schedule, 20 inference
steps, VACE scale `1.0`, disabled TTM, and fixed frame `18`?

The result is one pilot comparison. It cannot establish generalization or
metric scene-depth recovery.

## Input and comparison lock

- Frozen config SHA-256:
  `e3776437767663a82e48fa980cfc7e2bc187162d7dcf827207529d381e64a219`.
- Fail-closed preflight SHA-256:
  `e20523c526d6e7b503ad787e39c33da4af47af8d56274c1d5aa02f90b4412b52`.
- Inference runner SHA-256:
  `c247301ba4744c398de0e821b26ac14f7f29ab6d14a88d2c4607d4b5f2908202`.
- 3-D control manifest SHA-256:
  `7a22f15e93f595193dd295f43812c4fe841eb442a6698e843e41d9206ffaab15`.
- 3-D control video SHA-256:
  `3ea7b30f0cd3de411fff9c47c41fd4348768f5e3d8652a35860ab16ae1b55b87`.
- 3-D mask video SHA-256:
  `ee4fa6bd5a7a10dce7561dd66e97753cc213985bf7f9b9096b27a02f929c8b5a`.
- Exact-projection alpha SHA-256:
  `3776195165bd150d96c08287cc788278faabef93a484b865e2ee138ab7099d43`.
- Day 6 planar-control run-manifest SHA-256:
  `97b5c2fbd48b8edf318c524ab77c60192f2805d3d64c082fac6277cab1f86dc7`.
- Day 6 edited-frame SHA-256 recorded by that manifest:
  `8d9fd19d034e51e6afba8a66f7a5bc7d2a73752c7810111504228c1cb1950f63`.

## Resource observation on 2026-08-31

No inference was started. `gp38` and `gp40` had all eight A6000s occupied at
about 45 GiB and 100% utilization. `gp41` had all eight A40s occupied. `gp42`
had all eight Blackwell cards occupied and is not the frozen A6000 runtime.
`gp39` had ample nominal free VRAM, but all eight A6000s were at 100%
utilization with active `VLLM::Worker` processes owned by another user. This
fails the no-process and utilization gates even though the processes reserve
little VRAM.

The launcher therefore remains blocked. It requires an A6000 with at least
48,000 MiB free, at most 5% utilization, no compute process, and at least
80,000 MiB available host memory. It never terminates or modifies another
user's process, creates the output only after a passing preflight, refuses all
existing output/report paths, and forces offline model reuse.

## Review contract after completion

The two outputs must be assessed separately for (1) one fork with four tines,
continuous wrap/contact, plate-connected lifted strands, source reduction, and
absence of duplicate utensils; and (2) metal/food texture, lighting, seam,
shadow, and background realism. Exact protection outside the Day 8 motion
support is a technical invariant, not a substitute for either visual verdict.
