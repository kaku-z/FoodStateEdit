# Day 13 3D-guided flexible completion checklist

## Scientific contract

- [x] Select one seen synthetic udon sample; do not touch the blind fork case.
- [x] Separate the contextual 3D-representation question from the primary flexible-completion question.
- [x] Make `relative3d_topology_weighted` versus `relative3d_uniform` the primary causal comparison.
- [x] Keep target, reference, prompt, support, training budget, inference seed, and model bytes fixed.
- [x] Require a visible semantic improvement in addition to numerical improvement.
- [x] Keep held-out, real-data, physical-correctness, and paper-level-photo claims prohibited.

## Relative-3D control package

- [x] Build a 21-frame normalized-pinhole udon scaffold with two rigid chopsticks and one continuous flexible strand.
- [x] Keep one strand endpoint fixed in the bowl and bind the other endpoint to the pinch point from contact through final hold.
- [x] Encode pinch depth crossing and use depth-aware projection rather than fixed paint order.
- [x] Derive strand, pinch-contact, and source-connection mask videos from the same 3D scaffold.
- [x] Preserve frame zero and every pixel outside the declared support exactly before video encoding.
- [x] Save camera, 3D arrays, masks, videos, review boards, sizes, and SHA-256 records under a new output root.

## Matched training implementation

- [x] Add an independent seeded uniform training entry point without modifying the frozen upstream runtime.
- [x] Add the topology-weighted FlowMatch entry point with the frozen normalized weight formula.
- [ ] Prove identical initial LoRA hashes, dataloader order, timesteps, diffusion noise, precision, and optimizer state across arms.
- [x] Freeze all builder, dataset, trainer, preflight, validator, and evaluator hashes in a new execution config.
- [x] Keep each arm at exactly 32 optimizer steps and save steps 16 and 32.
- [x] Confirm every proposed local and remote path is absent before creation.

## Execution and review

- [x] Audit gp38--gp42 read only and run serially only on a qualifying idle RTX A6000.
- [ ] Run planar-uniform, relative3d-uniform, and relative3d-topology-weighted training without downloading models.
- [ ] Validate all checkpoints and run the frozen five-condition same-seed evaluation.
- [ ] Rehash every pulled artifact and require exact preservation outside support.
- [ ] Perform condition-blinded review by at least two independent reviewers.
- [ ] Evaluate pinch, strand continuity, bowl connection, lift, final hold, and photo realism separately.
- [ ] Apply the frozen positive-contribution gate without selecting another seed or checkpoint after seeing output.

## Stop rule

- [ ] If the primary gate fails, preserve the negative result and create a new frozen design before changing masks, weights, seeds, or samples.
- [ ] Do not run the blind fork or expand families from this design alone.

Implementation freeze evidence:

- dataset manifest SHA-256: `7dfe13533ff1e64a22793bd9184b923da833d193009a09e747430a6c7e47a902`
- execution config SHA-256: `061fb3797680cca7c17a3418246ec958433a99b604c9935be638e19adbdb0d4d`
- local contracts: 120 tests passed, 3 skipped
- gp40 v1 planar preflight report: `/tmp/foodstateedit_day13_planar_uniform_preflight_v1.json`; ready=false because imageio frame counting returned null on the uploaded videos, and no training output directory was created.
- gp40 v1 planar training output: `/tmp/foodstateedit_day13_planar_uniform_udon_lora_v1`; status=technical_failure_preserved because the first forward pass sent a 0-D timestep tensor into the frozen DiffSynth runtime, randomness_trace_count=0, and no checkpoint was created.
- gp40 weighted v2 training output: `/tmp/foodstateedit_day13_relative3d_topology_weighted_udon_lora_v2`; client session interruption left a preserved partial run with step-16 only, no run manifest, and 23 randomness trace rows.
- retry roots: planar and relative3d-uniform training output roots use `_v2`; weighted training output root uses `_v3`; evaluation output root uses `_v2`.

GPU execution is unlocked only through the immutable execution config and its
fresh per-arm resource preflight. The automatic heartbeat remains paused.

2026-09-05 local organization / outage note:

- User confirmed server power interruption. No remote status was recoverable
  during the latest read-only check.
- Earlier observations recorded complete planar v2 and relative3d-uniform v2
  runs (32 steps each), but those checkpoints are not locally recovered here.
- Weighted v3 completion is unknown; do not infer continued execution from its
  historical nohup launch or PID. Inspect temporary paths after restart.
- Five-condition evaluation and independent blinded review remain pending.
- Consolidated Chinese report: `results/LOCAL_PROGRESS_20260905.md`.
