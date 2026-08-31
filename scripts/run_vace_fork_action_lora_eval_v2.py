#!/usr/bin/env python3
"""Run the fork LoRA evaluation with VACE-block inner-linear injection.

DiffSynth's frozen low-VRAM runtime wraps each VACE attention block as one
AutoWrappedModule.  Its ordinary hot-loader therefore sees zero
AutoWrappedLinear children.  This runner keeps that runtime untouched and
injects the same low-rank branch into the wrapped block's inner Linear module.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import run_vace_fork_action_lora_eval as delegated_runner


EXECUTION_REVISION = "v2_vace_block_inner_linear_injection"
EXPECTED_INJECTIONS = 80
INJECTION_RECORD: dict[str, object] = {}


def config_from_argv() -> tuple[Path, dict[str, object]]:
    try:
        config_path = Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("--config is required before runtime initialization") from error
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("execution_revision") != EXECUTION_REVISION:
        raise ValueError(f"Unexpected execution revision: {config.get('execution_revision')}")
    return config_path, config


def unwrap_vram_module(module):
    while module.__class__.__name__ in {"AutoWrappedModule", "AutoWrappedNonRecurseModule"}:
        module = module.module
    return module


def resolve_parent(root, dotted_name: str):
    parts = dotted_name.split(".")
    current = root
    for part in parts[:-1]:
        current = unwrap_vram_module(current)
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return unwrap_vram_module(current), parts[-1]


def install_runtime_injection(pipe, checkpoint: Path, alpha: float) -> dict[str, object]:
    import torch
    from safetensors import safe_open

    class LoRAInjectedLinear(torch.nn.Module):
        def __init__(self, base, lora_a, lora_b, scale):
            super().__init__()
            if not isinstance(base, torch.nn.Linear):
                raise TypeError(f"LoRA target is not Linear: {type(base).__name__}")
            self.base = base
            self.in_features = base.in_features
            self.out_features = base.out_features
            self.register_buffer("lora_A", lora_a.to(device=base.weight.device, dtype=base.weight.dtype))
            self.register_buffer("lora_B", lora_b.to(device=base.weight.device, dtype=base.weight.dtype))
            self.scale = float(scale)

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            base_output = self.base(x)
            low_rank = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B)
            return base_output + low_rank * self.scale

    pairs: dict[str, dict[str, object]] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_A.default.weight" in key:
                target = key.replace(".lora_A.default.weight", "")
                pairs.setdefault(target, {})["A"] = handle.get_tensor(key)
            elif ".lora_B.default.weight" in key:
                target = key.replace(".lora_B.default.weight", "")
                pairs.setdefault(target, {})["B"] = handle.get_tensor(key)
            else:
                raise ValueError(f"Unexpected checkpoint key: {key}")

    incomplete = sorted(name for name, pair in pairs.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise ValueError(f"Incomplete LoRA pairs: {incomplete[:5]}")
    if len(pairs) != EXPECTED_INJECTIONS:
        raise ValueError(f"Expected {EXPECTED_INJECTIONS} LoRA pairs, found {len(pairs)}")

    injected: list[str] = []
    tensor_abs_max = 0.0
    for target, pair in sorted(pairs.items()):
        parent, child_name = resolve_parent(pipe.vace, target)
        base = getattr(parent, child_name)
        if base.__class__.__name__ != "Linear":
            raise TypeError(f"Target {target} resolved to {type(base).__name__}, expected unfused Linear")
        lora_a = pair["A"]
        lora_b = pair["B"]
        if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]:
            raise ValueError(f"Invalid LoRA shapes for {target}: {tuple(lora_a.shape)}, {tuple(lora_b.shape)}")
        if lora_a.shape[1] != base.in_features or lora_b.shape[0] != base.out_features:
            raise ValueError(f"LoRA/base shape mismatch for {target}")
        tensor_abs_max = max(tensor_abs_max, float(lora_a.float().abs().max()), float(lora_b.float().abs().max()))
        setattr(parent, child_name, LoRAInjectedLinear(base, lora_a, lora_b, alpha))
        injected.append(target)

    record = {
        "mode": EXECUTION_REVISION,
        "injected_linear_count": len(injected),
        "expected_injected_linear_count": EXPECTED_INJECTIONS,
        "checkpoint_tensor_count": len(pairs) * 2,
        "checkpoint_pair_count": len(pairs),
        "checkpoint_tensor_abs_max": tensor_abs_max,
        "targets": injected,
    }
    print(f"{len(injected)} VACE inner Linear layers are runtime-injected by LoRA.")
    return record


def install_pipeline_patch(config: dict[str, object]) -> None:
    runtime = config["runtime"]
    for key, value in runtime["offline_environment"].items():
        os.environ[key] = value
    sys.path.insert(0, runtime["geoedit_root"])
    from geoedit import inference

    original_load_pipeline = inference.load_pipeline
    expected_checkpoint = Path(config["adapter"]["checkpoint"]["path"]).resolve()
    expected_alpha = config["adapter"]["alpha"]

    def patched_load_pipeline(vram_limit):
        pipe = original_load_pipeline(vram_limit)

        def patched_load_lora(module, lora_config=None, alpha=1, **kwargs):
            checkpoint = Path(lora_config).resolve()
            if module is not pipe.vace:
                raise ValueError("The action LoRA may only target the high-noise pipe.vace module")
            if checkpoint != expected_checkpoint or alpha != expected_alpha:
                raise ValueError("Checkpoint or alpha differs from the frozen injection contract")
            global INJECTION_RECORD
            INJECTION_RECORD = install_runtime_injection(pipe, checkpoint, alpha)

        pipe.load_lora = patched_load_lora
        return pipe

    inference.load_pipeline = patched_load_pipeline


def finalize_provenance(config: dict[str, object]) -> None:
    output_root = Path(config["output_root"])
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_revision"] = EXECUTION_REVISION
    manifest["worker_sha256"] = delegated_runner.sha256_file(Path(__file__).resolve())
    manifest["delegated_runner_sha256"] = delegated_runner.sha256_file(Path(delegated_runner.__file__).resolve())
    manifest["adapter"]["runtime_injection"] = INJECTION_RECORD
    if manifest.get("status", "").startswith("complete") and INJECTION_RECORD.get("injected_linear_count") != EXPECTED_INJECTIONS:
        raise RuntimeError("Completed inference did not record all expected LoRA injections")

    command_path = output_root / "command.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command["execution_revision"] = EXECUTION_REVISION
    command["expected_runtime_injected_linear_count"] = EXPECTED_INJECTIONS
    delegated_runner.write_json(command_path, command)
    for output in manifest.get("outputs", []):
        if Path(output["path"]).name == "command.json":
            output["size_bytes"] = command_path.stat().st_size
            output["sha256"] = delegated_runner.sha256_file(command_path)
    delegated_runner.write_json(manifest_path, manifest)


def main() -> int:
    _, config = config_from_argv()
    install_pipeline_patch(config)
    return_code = delegated_runner.main()
    finalize_provenance(config)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
