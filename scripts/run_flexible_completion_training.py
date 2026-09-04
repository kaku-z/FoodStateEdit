#!/usr/bin/env python3
"""Run one frozen Day 13 training arm after a fresh safety preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from preflight_vace_fork_3d_compare import sha256_file


SCHEMA_VERSION = "foodstateedit.flexible_completion_execution.v1"
METHOD = "relative3d_topology_weighted_flexible_completion_v1"
ARM_IDS = (
    "planar_uniform",
    "relative3d_uniform",
    "relative3d_topology_weighted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    return parser.parse_args()


def hash_jsonl_sequence(path: Path, keys: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = {key: record[key] for key in keys}
            digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION or config.get("method") != METHOD:
        raise ValueError("Unexpected Day 13 execution config")
    if config.get("execution_allowed") is not True:
        raise ValueError("Execution config has not been unlocked")
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.preflight_report.resolve()
    arm = config["arms"][args.arm]
    if str(dataset_root) != config["dataset"]["remote_root"]:
        raise ValueError("Dataset root differs from frozen remote root")
    if str(output_root) != arm["output_root"]:
        raise ValueError("Output root differs from frozen arm root")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")

    preflight_script = repo_root / config["implementation"]["training_preflight"]["path"]
    preflight_command = [
        sys.executable,
        str(preflight_script),
        "--config",
        str(config_path),
        "--dataset-root",
        str(dataset_root),
        "--arm",
        args.arm,
        "--output-root",
        str(output_root),
        "--gpu-index",
        str(args.gpu_index),
        "--report",
        str(report_path),
    ]
    preflight = subprocess.run(preflight_command, check=False)
    if preflight.returncode != 0:
        print("Preflight blocked training; no output directory was created.", file=sys.stderr)
        return preflight.returncode
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ready") or report.get("selected_gpu") != args.gpu_index:
        raise RuntimeError("Passing preflight did not preserve the requested GPU")

    output_root.mkdir(parents=True, exist_ok=False)
    training = config["training"]
    runtime = config["trainer_runtime"]
    model = config["model"]
    trainer_root = Path(runtime["root"])
    trainer_script = repo_root / config["implementation"]["matched_trainer"]["path"]
    wrapper = repo_root / config["implementation"]["no_audio_wrapper"]["path"]
    trace_path = output_root / "randomness_trace.jsonl"
    initialization_path = output_root / "initialization_manifest.json"
    model_paths = [str(Path(model["root"]) / record["path"]) for record in model["files"][:3]]
    data_file_keys = ["video", "vace_video", "vace_reference_image"]
    if arm["loss_mode"] == "topology_weighted":
        data_file_keys.extend(
            [
                "flexible_strand_mask_video",
                "pinch_contact_mask_video",
                "source_connection_mask_video",
            ]
        )
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        str(wrapper),
        "--upstream-script",
        str(trainer_script),
        "--",
        "--dataset_base_path",
        str(dataset_root),
        "--dataset_metadata_path",
        str(dataset_root / arm["metadata"]),
        "--dataset_repeat",
        str(training["dataset_repeat"]),
        "--dataset_num_workers",
        "0",
        "--data_file_keys",
        ",".join(data_file_keys),
        "--max_pixels",
        str(training["max_pixels"]),
        "--num_frames",
        str(training["num_frames"]),
        "--model_paths",
        json.dumps(model_paths),
        "--tokenizer_path",
        str(Path(model["root"]) / model["tokenizer_root"]),
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
        "--loss_mode",
        arm["loss_mode"],
        "--training_seed",
        str(training["training_seed"]),
        "--randomness_trace_path",
        str(trace_path),
        "--initialization_manifest_path",
        str(initialization_path),
        "--strand_weight",
        str(training["topology_weights"]["strand"]),
        "--contact_weight",
        str(training["topology_weights"]["contact"]),
        "--source_connection_weight",
        str(training["topology_weights"]["source_connection"]),
    ]
    if training["gradient_checkpointing_offload"]:
        command.append("--use_gradient_checkpointing_offload")
    if training["model_cpu_offload"]:
        command.append("--enable_model_cpu_offload")
    if training["optimizer_cpu_offload"]:
        command.append("--enable_optimizer_cpu_offload")

    environment = os.environ.copy()
    environment.update(config["offline_environment"])
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "PYTHONPATH": os.pathsep.join([str(trainer_root), str(repo_root)]),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": str(training["training_seed"]),
        }
    )
    command_record = {
        "schema_version": "foodstateedit.flexible_completion_training_command.v1",
        "arm": args.arm,
        "loss_mode": arm["loss_mode"],
        "control": arm["control"],
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
                "PYTHONHASHSEED",
            )
        },
    }
    command_path = output_root / "command.json"
    command_path.write_text(json.dumps(command_record, indent=2) + "\n", encoding="utf-8")
    started = datetime.now(timezone.utc)
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
    finished = datetime.now(timezone.utc)

    checkpoints = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_root.glob("step-*.safetensors"))
    ]
    trace_records = []
    if trace_path.is_file():
        trace_records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    initialization = json.loads(initialization_path.read_text(encoding="utf-8")) if initialization_path.is_file() else {}
    expected_names = {f"step-{step}.safetensors" for step in training["expected_checkpoint_steps"]}
    observed_names = {Path(record["path"]).name for record in checkpoints}
    expected_steps = training["expected_optimizer_steps"]
    complete = (
        return_code == 0
        and expected_names == observed_names
        and len(trace_records) == expected_steps
        and [record.get("optimizer_step") for record in trace_records] == list(range(1, expected_steps + 1))
        and initialization.get("trainable_tensor_count") == training["expected_trainable_tensor_count"]
    )
    trace_keys = (
        "optimizer_step",
        "training_row_id",
        "training_sample_id",
        "timestep_seed",
        "noise_seed",
        "timestep_id",
        "timestep",
        "noise_sha256",
        "input_latent_shape",
        "prediction_shape_after_first_frame_policy",
        "conditioned_first_frame_excluded",
    )
    evidence_paths = [command_path, log_path]
    if trace_path.is_file():
        evidence_paths.append(trace_path)
    if initialization_path.is_file():
        evidence_paths.append(initialization_path)
    manifest = {
        "schema_version": "foodstateedit.flexible_completion_training_run.v1",
        "status": "complete_requires_cross_arm_match_and_checkpoint_validation" if complete else "technical_failure_preserved",
        "arm": args.arm,
        "loss_mode": arm["loss_mode"],
        "control": arm["control"],
        "return_code": return_code,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "selected_physical_gpu": args.gpu_index,
        "config_sha256": sha256_file(config_path),
        "preflight_report_sha256": sha256_file(report_path),
        "dataset_manifest_sha256": sha256_file(dataset_root / config["dataset"]["manifest"]["path"]),
        "initial_trainable_state_aggregate_sha256": initialization.get("aggregate_sha256"),
        "randomness_trace_count": len(trace_records),
        "matched_randomness_sequence_sha256": hash_jsonl_sequence(trace_path, trace_keys) if trace_path.is_file() else None,
        "checkpoints": checkpoints,
        "evidence": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in evidence_paths
        ],
        "blind_fork_evaluation_allowed": False,
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
