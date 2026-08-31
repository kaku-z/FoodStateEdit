#!/usr/bin/env python3
"""Fail-closed preflight for the synthetic-pseudo-target action LoRA pilot."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from preflight_adapter_training import (
    add_check,
    available_memory_mib,
    read_gpu_state,
    resolve_dataset_file,
    sha256_file,
    sha256_text_lf,
)


METHOD = "foodstateedit_vace_lora_action_pseudo_v1"
SCIENTIFIC_STATUS = "mechanism_pilot_with_synthetic_pseudotargets"
TARGET_POLICY = "disclosed_imagegen_pseudotarget_nonidentity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    add_check(checks, "method", config.get("method") == METHOD, config.get("method"), METHOD)
    add_check(
        checks,
        "scientific_status",
        config.get("scientific_status") == SCIENTIFIC_STATUS,
        config.get("scientific_status"),
        SCIENTIFIC_STATUS,
    )
    add_check(
        checks,
        "claim_limit",
        "not real-data training" in config.get("claim_limit", "")
        and "cannot establish" in config.get("claim_limit", ""),
        config.get("claim_limit"),
        "Must explicitly exclude real-data and generalization/photo-realism claims.",
    )
    add_check(checks, "output_absent", not args.output_root.exists(), str(args.output_root), "Output root must not exist.")
    add_check(
        checks,
        "frozen_output_root",
        str(args.output_root.resolve()) == config["training"]["output_root"],
        str(args.output_root.resolve()),
        config["training"]["output_root"],
    )
    add_check(
        checks,
        "expected_optimizer_steps",
        config["training"]["expected_optimizer_steps"]
        == config["dataset"]["sample_count"]
        * config["training"]["dataset_repeat"]
        * config["training"]["epochs"],
        config["training"]["expected_optimizer_steps"],
        "sample_count * dataset_repeat * epochs",
    )

    runtime_root = config_path.parents[1]
    launcher = config["launcher"]
    preflight_path = runtime_root / launcher["preflight"]
    runner_path = runtime_root / launcher["runner"]
    add_check(
        checks,
        "preflight_sha256",
        preflight_path.is_file() and sha256_file(preflight_path) == launcher["preflight_sha256"],
        sha256_file(preflight_path) if preflight_path.is_file() else None,
        launcher["preflight_sha256"],
    )
    add_check(
        checks,
        "runner_sha256",
        runner_path.is_file() and sha256_file(runner_path) == launcher["runner_sha256"],
        sha256_file(runner_path) if runner_path.is_file() else None,
        launcher["runner_sha256"],
    )
    audit_path = runtime_root / config["target_audit"]["path"]
    audit_hash = sha256_file(audit_path) if audit_path.is_file() else None
    add_check(
        checks,
        "target_audit_hash",
        audit_hash == config["target_audit"]["sha256"],
        audit_hash,
        config["target_audit"]["sha256"],
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    audit_decision = audit.get("decision", {})
    add_check(
        checks,
        "target_audit_counts",
        audit_decision.get("eligible_count") == 2
        and audit_decision.get("real_photo_target_count") == 0
        and audit_decision.get("synthetic_pseudotarget_count") == 2,
        audit_decision,
        "Exactly two synthetic pseudo-targets and zero real-photo targets.",
    )
    add_check(
        checks,
        "held_out_fork_policy",
        config["held_out"].get("family") == "fork_twirl_and_lift"
        and config["held_out"].get("training_occurrences") == 0,
        config["held_out"],
        "Fork twirl/lift is unseen in training.",
    )

    trainer = config["trainer"]
    trainer_root = Path(trainer["remote_root"])
    train_script = trainer_root / trainer["train_script"]
    no_audio_wrapper = runtime_root / trainer["no_audio_wrapper"]
    dataset_builder = runtime_root / trainer["dataset_builder"]
    commit_marker = trainer_root / "SOURCE_COMMIT"
    archive_marker = trainer_root / "SOURCE_ARCHIVE_SHA256"
    add_check(checks, "trainer_script", train_script.is_file(), str(train_script), "Frozen trainer exists.")
    add_check(
        checks,
        "trainer_script_sha256",
        train_script.is_file() and sha256_file(train_script) == trainer["train_script_sha256"],
        sha256_file(train_script) if train_script.is_file() else None,
        trainer["train_script_sha256"],
    )
    add_check(
        checks,
        "no_audio_wrapper_sha256",
        no_audio_wrapper.is_file() and sha256_file(no_audio_wrapper) == trainer["no_audio_wrapper_sha256"],
        sha256_file(no_audio_wrapper) if no_audio_wrapper.is_file() else None,
        trainer["no_audio_wrapper_sha256"],
    )
    add_check(
        checks,
        "dataset_builder_sha256_lf",
        dataset_builder.is_file() and sha256_text_lf(dataset_builder) == trainer["dataset_builder_sha256_lf"],
        sha256_text_lf(dataset_builder) if dataset_builder.is_file() else None,
        trainer["dataset_builder_sha256_lf"],
    )
    add_check(
        checks,
        "trainer_commit_marker",
        commit_marker.is_file() and commit_marker.read_text(encoding="utf-8").strip() == trainer["commit"],
        commit_marker.read_text(encoding="utf-8").strip() if commit_marker.is_file() else None,
        trainer["commit"],
    )
    add_check(
        checks,
        "trainer_archive_marker",
        archive_marker.is_file()
        and archive_marker.read_text(encoding="utf-8").strip() == trainer["source_archive_sha256"],
        archive_marker.read_text(encoding="utf-8").strip() if archive_marker.is_file() else None,
        trainer["source_archive_sha256"],
    )

    model = config["model"]
    model_root = Path(model["root"])
    model_audit_path = Path(model["hash_audit"])
    model_audit_hash = sha256_file(model_audit_path) if model_audit_path.is_file() else None
    add_check(
        checks,
        "model_audit",
        model_audit_hash == model["hash_audit_sha256"],
        model_audit_hash,
        model["hash_audit_sha256"],
    )
    model_audit_text = model_audit_path.read_text(encoding="utf-8") if model_audit_path.is_file() else ""
    for record in model["files"]:
        path = model_root / record["path"]
        size = path.stat().st_size if path.is_file() else None
        add_check(checks, f"model_size:{record['path']}", size == record["size_bytes"], size, str(record["size_bytes"]))
        add_check(
            checks,
            f"model_recorded_hash:{record['path']}",
            record["sha256"] in model_audit_text,
            record["sha256"] in model_audit_text,
            "Expected full hash is in the frozen audit.",
        )
    for record in model["tokenizer_files"]:
        path = model_root / record["path"]
        size = path.stat().st_size if path.is_file() else None
        actual_hash = sha256_file(path) if path.is_file() else None
        add_check(checks, f"tokenizer_size:{record['path']}", size == record["size_bytes"], size, str(record["size_bytes"]))
        add_check(checks, f"tokenizer_hash:{record['path']}", actual_hash == record["sha256"], actual_hash, record["sha256"])

    dataset_root = args.dataset_root.resolve()
    add_check(
        checks,
        "frozen_dataset_root",
        str(dataset_root) == config["dataset"]["remote_root"],
        str(dataset_root),
        config["dataset"]["remote_root"],
    )
    manifest_path = dataset_root / config["dataset"]["manifest"]
    metadata_path = dataset_root / config["dataset"]["metadata"]
    manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    add_check(
        checks,
        "dataset_manifest_hash",
        manifest_hash == config["dataset"]["manifest_sha256"],
        manifest_hash,
        config["dataset"]["manifest_sha256"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    add_check(
        checks,
        "dataset_contract",
        manifest.get("scientific_status") == SCIENTIFIC_STATUS
        and manifest.get("target_policy") == TARGET_POLICY
        and manifest.get("sample_count") == 2
        and manifest.get("held_out", {}).get("family") == "fork_twirl_and_lift"
        and manifest.get("held_out", {}).get("training_occurrences") == 0,
        {
            "scientific_status": manifest.get("scientific_status"),
            "target_policy": manifest.get("target_policy"),
            "sample_count": manifest.get("sample_count"),
            "held_out": manifest.get("held_out"),
        },
        "Two non-fork synthetic pseudo-targets under the frozen policy.",
    )
    provenance = manifest.get("provenance", {})
    add_check(
        checks,
        "manifest_provenance",
        provenance.get("builder_sha256") == trainer["dataset_builder_sha256_lf"]
        and provenance.get("target_audit_sha256") == config["target_audit"]["sha256"],
        provenance,
        "Builder and target-audit hashes match the config.",
    )
    metadata_hash = sha256_file(metadata_path) if metadata_path.is_file() else None
    add_check(
        checks,
        "metadata_hash",
        metadata_hash == config["dataset"]["metadata_sha256"] == manifest.get("metadata", {}).get("sha256"),
        metadata_hash,
        config["dataset"]["metadata_sha256"],
    )

    rows: list[dict[str, str]] = []
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    add_check(checks, "metadata_rows", len(rows) == 2, len(rows), "2")
    samples = manifest.get("samples", [])
    add_check(checks, "manifest_samples", len(samples) == len(rows) == 2, len(samples), "2")
    audited_ids = set(audit_decision.get("eligible_primary_targets", []))
    training_families: set[str] = set()
    for row_index, row in enumerate(rows):
        sample = samples[row_index] if row_index < len(samples) else {}
        training_families.add(str(sample.get("family")))
        nonidentity = sample.get("nonidentity_checks", {})
        add_check(
            checks,
            f"pseudo_target_contract:{row_index}",
            sample.get("target_kind") == "synthetic_imagegen_pseudotarget"
            and sample.get("real_photo_supervision") is False
            and sample.get("audit_candidate_id") in audited_ids
            and nonidentity.get("target_video_hash_differs_from_control") is True
            and nonidentity.get("target_outside_edit_support_max_difference") == 0
            and nonidentity.get("control_outside_edit_support_max_difference") == 0
            and 0 < nonidentity.get("edit_support_fraction", 0) < 0.2,
            {
                "target_kind": sample.get("target_kind"),
                "real_photo_supervision": sample.get("real_photo_supervision"),
                "audit_candidate_id": sample.get("audit_candidate_id"),
                "nonidentity_checks": nonidentity,
            },
            "Disclosed non-identity pseudo-target with exact protected pixels and bounded support.",
        )
        records = sample.get("files", {})
        for key in ("video", "vace_video", "vace_reference_image", "target_keyframe", "control_keyframe", "edit_alpha"):
            record = records.get(key, {})
            relative_path = row.get(key, "") if key in row else record.get("path", "")
            if key in row:
                add_check(
                    checks,
                    f"dataset_path:{row_index}:{key}",
                    relative_path == record.get("path"),
                    relative_path,
                    str(record.get("path")),
                )
            try:
                path = resolve_dataset_file(dataset_root, str(relative_path))
            except ValueError as error:
                add_check(checks, f"dataset_file:{row_index}:{key}", False, str(error), "Path stays inside dataset root.")
                continue
            size = path.stat().st_size if path.is_file() else None
            actual_hash = sha256_file(path) if path.is_file() else None
            add_check(checks, f"dataset_size:{row_index}:{key}", size == record.get("size_bytes"), size, str(record.get("size_bytes")))
            add_check(checks, f"dataset_hash:{row_index}:{key}", actual_hash == record.get("sha256"), actual_hash, str(record.get("sha256")))
        add_check(
            checks,
            f"nonidentity_hash:{row_index}",
            records.get("video", {}).get("sha256") != records.get("vace_video", {}).get("sha256"),
            {"video": records.get("video", {}).get("sha256"), "vace_video": records.get("vace_video", {}).get("sha256")},
            "Target and control hashes must differ.",
        )
    add_check(
        checks,
        "no_fork_training_family",
        "strand_contact" not in training_families and "fork_twirl_and_lift" not in training_families,
        sorted(training_families),
        "No fork family in training rows.",
    )

    help_env = os.environ.copy()
    help_env.update(config["offline_environment"])
    help_env["PYTHONPATH"] = str(trainer_root)
    video_paths: set[Path] = set()
    for row in rows:
        for key in ("video", "vace_video"):
            if row.get(key):
                video_paths.add(resolve_dataset_file(dataset_root, row[key]))
    for video_path in sorted(video_paths):
        decode_result = subprocess.run(
            [trainer["python"], str(no_audio_wrapper), "--video-decode-smoke", str(video_path)],
            check=False,
            capture_output=True,
            text=True,
            env=help_env,
            timeout=60,
        )
        decoded = None
        if decode_result.returncode == 0:
            try:
                decoded = json.loads(decode_result.stdout)
            except json.JSONDecodeError:
                pass
        add_check(
            checks,
            f"video_decode:{video_path.relative_to(dataset_root).as_posix()}",
            decode_result.returncode == 0
            and decoded is not None
            and decoded.get("frame_count") == config["dataset"]["frames"]
            and decoded.get("fps", 0) > 0
            and decoded.get("first_shape") == decoded.get("last_shape"),
            decoded if decoded is not None else decode_result.stderr[-2000:],
            "Exactly 21 decodable frames with stable shape and positive fps.",
        )

    module_status = {name: importlib.util.find_spec(name) is not None for name in config["required_modules"]}
    add_check(checks, "python_modules", all(module_status.values()), module_status, "All required modules importable.")
    trainer_help = subprocess.run(
        [trainer["python"], str(train_script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=help_env,
        timeout=60,
    )
    add_check(checks, "trainer_help", trainer_help.returncode == 0, trainer_help.stderr[-2000:], "Frozen trainer --help exits 0.")
    wrapper_help = subprocess.run(
        [
            trainer["python"],
            str(no_audio_wrapper),
            "--upstream-script",
            str(train_script),
            "--",
            "--data_file_keys",
            "video,vace_video,vace_reference_image",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=help_env,
        timeout=60,
    )
    add_check(checks, "no_audio_wrapper_help", wrapper_help.returncode == 0, wrapper_help.stderr[-2000:], "No-audio wrapper --help exits 0.")

    available_mib = available_memory_mib()
    gate = config["resource_gate"]
    add_check(
        checks,
        "system_memory",
        available_mib >= gate["min_available_system_memory_mib"],
        available_mib,
        f">={gate['min_available_system_memory_mib']} MiB available",
    )
    gpus, processes = read_gpu_state()
    safe_gpus = [
        gpu
        for gpu in gpus
        if gpu["name"] == gate["required_gpu_name"]
        and gpu["memory_free_mib"] >= gate["min_free_memory_mib"]
        and gpu["utilization_percent"] <= gate["max_utilization_percent"]
        and (not gate["require_no_compute_process"] or not gpu["compute_processes"])
    ]
    selected = None
    if args.gpu_index is not None:
        selected = next((gpu for gpu in safe_gpus if gpu["index"] == args.gpu_index), None)
    elif safe_gpus:
        selected = safe_gpus[0]
    add_check(
        checks,
        "gpu_gate",
        selected is not None,
        {"requested": args.gpu_index, "safe_gpu_indices": [gpu["index"] for gpu in safe_gpus], "gpus": gpus, "processes": processes},
        "At least one idle NVIDIA RTX A6000 with >=48000 MiB free and no compute process.",
    )

    ready = all(check["passed"] for check in checks)
    report = {
        "schema_version": "foodstateedit.adapter_action_preflight.v1",
        "ready": ready,
        "selected_gpu": selected["index"] if selected is not None else None,
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": manifest_hash,
        "target_audit_sha256": audit_hash,
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2)
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"Refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
