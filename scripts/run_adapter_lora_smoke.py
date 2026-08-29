#!/usr/bin/env python3
"""Launch the frozen two-step VACE-LoRA smoke only after a passing preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


METHOD = "foodstateedit_vace_lora_high_noise_smoke"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--preflight-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["method"] != METHOD:
        raise ValueError(f"Unexpected method: {config['method']}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    if str(output_root) != config["training"]["output_root"]:
        raise ValueError("Output root does not match the frozen config")
    if str(dataset_root) != config["dataset"]["remote_root"]:
        raise ValueError("Dataset root does not match the frozen config")

    preflight_script = Path(__file__).resolve().with_name("preflight_adapter_training.py")
    preflight_command = [
        sys.executable,
        str(preflight_script),
        "--config",
        str(config_path),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--report",
        str(args.preflight_report.resolve()),
    ]
    if args.gpu_index is not None:
        preflight_command.extend(["--gpu-index", str(args.gpu_index)])
    preflight = subprocess.run(preflight_command, check=False)
    if preflight.returncode != 0:
        print("Preflight blocked the run; no training output directory was created.", file=sys.stderr)
        return preflight.returncode
    preflight_report = json.loads(args.preflight_report.resolve().read_text(encoding="utf-8"))
    selected_gpu = preflight_report["selected_gpu"]
    if selected_gpu is None:
        raise RuntimeError("Passing preflight did not select a GPU")

    trainer = config["trainer"]
    trainer_root = Path(trainer["remote_root"])
    train_script = trainer_root / trainer["train_script"]
    no_audio_wrapper = config_path.parents[1] / trainer["no_audio_wrapper"]
    model_root = Path(config["model"]["root"])
    model_paths = [str(model_root / path) for path in config["training"]["model_paths"]]
    training = config["training"]
    output_root.mkdir(parents=True)

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        str(no_audio_wrapper),
        "--upstream-script",
        str(train_script),
        "--",
        "--dataset_base_path",
        str(dataset_root),
        "--dataset_metadata_path",
        str(dataset_root / config["dataset"]["metadata"]),
        "--dataset_repeat",
        str(training["dataset_repeat"]),
        "--dataset_num_workers",
        "0",
        "--data_file_keys",
        "video,vace_video,vace_reference_image",
        "--max_pixels",
        str(training["max_pixels"]),
        "--num_frames",
        str(training["num_frames"]),
        "--model_paths",
        json.dumps(model_paths),
        "--tokenizer_path",
        str(model_root / config["model"]["tokenizer_root"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--weight_decay",
        str(training["weight_decay"]),
        "--num_epochs",
        str(training["epochs"]),
        "--save_steps",
        str(training["save_steps"]),
        "--remove_prefix_in_ckpt",
        "pipe.vace.",
        "--output_path",
        str(output_root),
        "--lora_base_model",
        training["lora_base_model"],
        "--lora_target_modules",
        training["lora_target_modules"],
        "--lora_rank",
        str(training["lora_rank"]),
        "--extra_inputs",
        "vace_video,vace_reference_image",
        "--task",
        "sft",
        "--max_timestep_boundary",
        str(training["max_timestep_boundary"]),
        "--min_timestep_boundary",
        str(training["min_timestep_boundary"]),
    ]
    if training["gradient_checkpointing_offload"]:
        command.append("--use_gradient_checkpointing_offload")
    if training["model_cpu_offload"]:
        command.append("--enable_model_cpu_offload")
    if training["optimizer_cpu_offload"]:
        command.append("--enable_optimizer_cpu_offload")

    environment = os.environ.copy()
    environment.update(config["offline_environment"])
    environment["PYTHONPATH"] = str(trainer_root)
    environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command_record = {
        "method": METHOD,
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "DIFFSYNTH_SKIP_DOWNLOAD",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HOME",
                "PYTHONPATH",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    (output_root / "command.json").write_text(json.dumps(command_record, indent=2) + "\n", encoding="utf-8")

    started_at = datetime.now(timezone.utc)
    log_path = output_root / "train.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    finished_at = datetime.now(timezone.utc)

    checkpoints = []
    for path in sorted(output_root.rglob("*.safetensors")):
        checkpoints.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    status = "complete" if return_code == 0 and checkpoints else "technical_failure"
    run_manifest = {
        "schema_version": "foodstateedit.adapter_smoke_run.v0",
        "method": METHOD,
        "scientific_status": config["scientific_status"],
        "status": status,
        "return_code": return_code,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "selected_physical_gpu": selected_gpu,
        "config_sha256": sha256_file(config_path),
        "preflight_report_sha256": sha256_file(args.preflight_report.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_root / config["dataset"]["manifest"]),
        "trainer_commit": trainer["commit"],
        "trainer_script_sha256": sha256_file(train_script),
        "no_audio_wrapper_sha256": sha256_file(no_audio_wrapper),
        "model_hash_audit_sha256": sha256_file(Path(config["model"]["hash_audit"])),
        "command_sha256": sha256_file(output_root / "command.json"),
        "log_sha256": sha256_file(log_path),
        "checkpoints": checkpoints,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
