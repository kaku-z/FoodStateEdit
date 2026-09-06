#!/usr/bin/env python3
"""Validate frozen Day 13 checkpoints and bridge legacy config schemas, offline.

This does not train or run inference. All reports go to a new evidence directory.
The frozen legacy validator's method/claim text is historical boilerplate; the
compatibility manifest records the actual one-sample Day 13 provenance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ARMS = ("planar_uniform", "relative3d_uniform", "relative3d_topology_weighted")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def compatibility_config(execution: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    return {
        "method": "foodstateedit_vace_lora_action_pseudo_v1",
        "compatibility_only": True,
        "actual_method": execution["method"],
        "actual_unique_sample_count": execution["dataset"]["unique_sample_count"],
        "claim_limit": execution["claim_limit"],
        "training": {
            "output_root": execution["arms"][arm]["output_root"],
            "lora_rank": execution["training"]["lora_rank"],
            "model_paths": [execution["model"]["files"][0]["path"]],
        },
        "trainer": {
            "remote_root": execution["trainer_runtime"]["root"],
            "commit": execution["trainer_runtime"]["commit"],
            "train_script_sha256": execution["implementation"]["matched_trainer"]["sha256"],
        },
        "model": copy.deepcopy(execution["model"]),
        "offline_environment": copy.deepcopy(execution["offline_environment"]),
        "resource_gate": copy.deepcopy(execution["resource_gate"]),
    }


def normalized_template(execution: dict, template: dict) -> dict:
    result = copy.deepcopy(template)
    audit = execution["model"]["hash_audit"]
    runtime = result["runtime"]
    old_audit = runtime["model_hash_audit"]
    old_path = old_audit["path"] if isinstance(old_audit, dict) else old_audit
    old_hash = old_audit["sha256"] if isinstance(old_audit, dict) else runtime["model_hash_audit_sha256"]
    if old_path != audit["path"] or old_hash != audit["sha256"]:
        raise ValueError("Runtime template model audit differs from frozen training model")
    for field in ("seed", "num_frames", "num_inference_steps", "vace_scale", "enable_ttm", "review_frame_indices"):
        if result["inference"][field] != execution["evaluation"][field]:
            raise ValueError(f"Inference protocol mismatch: {field}")
    runtime["model_hash_audit"] = copy.deepcopy(audit)
    return result


def check_validation(report: dict, checkpoint: Path, expected_hash: str) -> None:
    if (report.get("status") != "complete"
            or report.get("tensor_count") != 160
            or report.get("official_loader_updated_tensor_count") != 80
            or Path(report.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or report.get("checkpoint_sha256") != expected_hash
            or digest(checkpoint) != expected_hash):
        raise ValueError("Checkpoint validation not complete or not bound to this exact checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--runtime-template-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    execution_path = args.execution_config.resolve()
    execution = read(execution_path)
    if (execution.get("schema_version") != "foodstateedit.flexible_completion_execution.v1"
            or not execution.get("execution_allowed")
            or execution.get("blind_fork_evaluation_allowed") is not False):
        raise ValueError("Not an authorized seen-only execution config")
    output = args.output_root.resolve()
    repo = execution_path.parents[1]
    scripts = execution["implementation"]
    for key in ("cross_arm_match_verifier", "checkpoint_validator", "evaluation_config_builder"):
        record = scripts[key]
        if digest(repo / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen script hash mismatch: {key}")
    template = normalized_template(execution, read(args.runtime_template_config))
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "compatibility_manifest.json", {
        "actual_method": execution["method"],
        "source_execution_config_sha256": digest(execution_path),
        "preparation_script_sha256": digest(Path(__file__)),
        "actual_unique_sample_count": execution["dataset"]["unique_sample_count"],
        "claim_limit": execution["claim_limit"],
        "legacy_validator_note": "Legacy method and two-pseudo-target wording in raw validator reports are boilerplate, not Day 13 scientific provenance. Reports validate tensor structure and official loadability only.",
        "runtime_change": "Represent the same frozen model audit as path/size/hash record; no inference parameter change.",
    })
    python = execution["python"]
    cross = output / "cross_arm_match.json"
    subprocess.run([python, str(repo / scripts["cross_arm_match_verifier"]["path"]),
                    "--config", str(execution_path), "--report", str(cross)], check=True)
    validations = []
    for arm in ARMS:
        root = Path(execution["arms"][arm]["output_root"])
        manifest = read(root / "run_manifest.json")
        if manifest["config_sha256"] != digest(execution_path):
            raise ValueError(f"Training config mismatch: {arm}")
        compat = output / f"{arm}_validator_compat.json"
        write_new(compat, compatibility_config(execution, arm))
        for step in execution["training"]["expected_checkpoint_steps"]:
            checkpoint = root / f"step-{step}.safetensors"
            record = next(r for r in manifest["checkpoints"] if r["path"] == checkpoint.name)
            report_path = output / f"{arm}_step{step}_validation.json"
            command = [python, str(repo / scripts["checkpoint_validator"]["path"]),
                       "--config", str(compat), "--checkpoint", str(checkpoint),
                       "--expected-sha256", record["sha256"], "--report", str(report_path)]
            with (output / f"{arm}_step{step}_validation.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            check_validation(read(report_path), checkpoint, record["sha256"])
            print(f"Validated {arm} step {step}", flush=True)
            if step == execution["evaluation"]["primary_checkpoint_step"]:
                validations.extend(["--validation", f"{arm}={report_path}"])
    template_path = output / "runtime_template_normalized.json"
    write_new(template_path, template)
    subprocess.run([python, str(repo / scripts["evaluation_config_builder"]["path"]),
                    "--execution-config", str(execution_path), "--runtime-template-config", str(template_path),
                    "--cross-arm-match", str(cross), *validations,
                    "--output-config", str(output / "evaluation_config.json")], check=True)
    write_new(output / "COMPLETE.json", {"status": "checkpoints_validated_evaluation_frozen_not_run",
              "evaluation_config_sha256": digest(output / "evaluation_config.json")})


if __name__ == "__main__":
    main()
