#!/usr/bin/env python3
"""Validate a FoodStateEdit VACE-LoRA checkpoint structurally and by official load."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


METHOD = "foodstateedit_vace_lora_high_noise_smoke"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def lora_pair_id(key: str) -> tuple[str, str] | None:
    match = re.match(r"^(.*)\.lora_([AB])(?:\.default)?\.weight$", key)
    if match is None:
        return None
    return match.group(1), match.group(2)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    report: dict[str, object] = {
        "schema_version": "foodstateedit.adapter_checkpoint_validation.v0",
        "method": METHOD,
        "status": "technical_failure",
        "started_at": started_at.isoformat(),
        "checkpoint": str(checkpoint),
    }
    return_code = 3
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["method"] != METHOD:
            raise ValueError(f"Unexpected method: {config['method']}")
        output_root = Path(config["training"]["output_root"]).resolve()
        if checkpoint != output_root and output_root not in checkpoint.parents:
            raise ValueError("Checkpoint is outside the frozen training output root")
        if checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing safetensors checkpoint: {checkpoint}")

        for key, value in config["offline_environment"].items():
            os.environ[key] = value
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        available_mib = available_memory_mib()
        minimum_mib = config["resource_gate"]["min_available_system_memory_mib"]
        if available_mib < minimum_mib:
            raise RuntimeError(f"Only {available_mib} MiB host memory available; require {minimum_mib}")

        import torch
        from safetensors import safe_open

        tensor_records: list[dict[str, object]] = []
        pair_members: dict[str, set[str]] = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            if not keys:
                raise ValueError("Checkpoint contains no tensors")
            for key in keys:
                tensor = handle.get_tensor(key)
                pair = lora_pair_id(key)
                if pair is None:
                    raise ValueError(f"Unexpected non-LoRA tensor key: {key}")
                pair_members.setdefault(pair[0], set()).add(pair[1])
                if tensor.ndim != 2 or config["training"]["lora_rank"] not in tensor.shape:
                    raise ValueError(f"Unexpected rank/shape for {key}: {tuple(tensor.shape)}")
                if not torch.isfinite(tensor).all().item():
                    raise ValueError(f"Non-finite tensor values in {key}")
                tensor_records.append(
                    {
                        "key": key,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                )
                del tensor
        incomplete_pairs = sorted(name for name, members in pair_members.items() if members != {"A", "B"})
        if incomplete_pairs:
            raise ValueError(f"Incomplete LoRA A/B pairs: {incomplete_pairs[:10]}")

        trainer_root = Path(config["trainer"]["remote_root"])
        sys.path.insert(0, str(trainer_root))
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        model_root = Path(config["model"]["root"])
        high_noise_path = model_root / config["training"]["model_paths"][0]
        load_output = io.StringIO()
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[ModelConfig(path=str(high_noise_path))],
            tokenizer_config=None,
            redirect_common_files=False,
        )
        if pipe.vace is None:
            raise RuntimeError("Frozen high-noise model did not produce pipe.vace")
        with contextlib.redirect_stdout(load_output):
            pipe.load_lora(pipe.vace, str(checkpoint), alpha=0.0)
        loader_text = load_output.getvalue()
        match = re.search(r"(\d+) tensors are (?:updated|fused) by LoRA", loader_text)
        if match is None or int(match.group(1)) <= 0:
            raise RuntimeError(f"Official LoRA loader did not update any VACE tensors: {loader_text!r}")
        updated_tensors = int(match.group(1))

        report.update(
            {
                "status": "complete",
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": sha256_file(config_path),
                "trainer_commit": config["trainer"]["commit"],
                "trainer_script_sha256": config["trainer"]["train_script_sha256"],
                "base_model_file": str(high_noise_path),
                "host_memory_available_mib_before_load": available_mib,
                "tensor_count": len(tensor_records),
                "pair_count": len(pair_members),
                "rank": config["training"]["lora_rank"],
                "official_loader_updated_tensor_count": updated_tensors,
                "tensor_records": tensor_records,
                "claim_limit": "Checkpoint structure and official VACE loadability only; no visual-quality claim.",
            }
        )
        return_code = 0
        del pipe
        gc.collect()
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        finished_at = datetime.now(timezone.utc)
        report["finished_at"] = finished_at.isoformat()
        report["duration_seconds"] = (finished_at - started_at).total_seconds()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
