"""Run one frozen E1 case with one pipeline load and three serial conditions."""

import argparse
import contextlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from run_high_lift_vace_pilot import (
    project_outputs,
    resource_preflight,
    run_text,
    sha256,
    validate_file,
    write_json,
)


EXPECTED_ARMS = [
    "appearance_laden_rgb",
    "structure_scribble",
    "structure_scribble_payload_reference",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    manifest_path = args.controls / "manifest.json"
    if sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("E1 manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "foodstateedit.e1_control_factorization_controls.v1":
        raise ValueError("Unexpected E1 controls manifest")
    experiment = manifest["config"]
    arms = [item["id"] for item in experiment["arms"]]
    if arms != EXPECTED_ARMS:
        raise ValueError("Frozen E1 arm set changed")
    case = next(item for item in manifest["cases"] if item["case_id"] == args.case)
    data_root = args.controls / args.case
    for record in case["files"].values():
        validate_file(args.controls / record["path"], record)
    if args.output_root.exists() or args.preflight_report.exists():
        raise FileExistsError("E1 output or preflight path already exists")

    gate = resource_preflight(base_config, args.output_root, args.preflight_report, args.gpu)
    args.output_root.mkdir(parents=True, exist_ok=False)
    inference_cfg = experiment["inference"]
    start = time.monotonic()
    done = threading.Event()
    state = {
        "schema_version": "foodstateedit.e1_control_factorization_run.v1",
        "status": "initializing",
        "case_id": args.case,
        "pid": os.getpid(),
        "physical_gpu": args.gpu,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_load_count": 0,
        "completed_conditions": [],
        "manifest_sha256": args.manifest_sha256,
        "base_config_sha256": sha256(args.base_config),
        "runner_sha256": sha256(Path(__file__)),
        "seed": inference_cfg["seed"],
        "num_frames": inference_cfg["num_frames"],
        "num_inference_steps": inference_cfg["num_inference_steps"],
        "vace_scale": inference_cfg["vace_scale"],
        "lora_enabled": inference_cfg["lora_enabled"],
        "enable_ttm": inference_cfg["enable_ttm"],
        "claim_limit": experiment["claim_limit"],
    }
    write_json(args.output_root / "run_manifest.json", state)

    def abort(signum, frame):
        raise RuntimeError("E1 worker stopped at its own safety/deadline boundary")

    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGALRM, abort)
    signal.alarm(int(experiment["hard_timeout_seconds_per_case"]))

    def assert_no_foreign_process() -> None:
        lines = run_text([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]).splitlines()
        for line in lines:
            fields = [value.strip() for value in line.split(",")]
            if len(fields) >= 4 and fields[0] == gate["gpu"]["uuid"] and int(fields[1]) != os.getpid():
                raise RuntimeError("Foreign compute process appeared: " + line)

    def resource_watch() -> None:
        while not done.wait(10):
            try:
                assert_no_foreign_process()
            except Exception as exc:
                (args.output_root / "resource_stop.txt").write_text(repr(exc), encoding="utf-8")
                os.kill(os.getpid(), signal.SIGTERM)
                return

    try:
        runtime = base_config["runtime"]
        validate_file(Path(runtime["model_hash_audit"]["path"]), runtime["model_hash_audit"])
        for relative, expected in runtime["geoedit_files"].items():
            if sha256(Path(runtime["geoedit_root"]) / relative) != expected:
                raise ValueError("GeoEdit runtime hash mismatch: " + relative)
        for relative, record in runtime["model_files"].items():
            validate_file(Path(runtime["model_root"]) / relative, record, hash_file=False)

        reference = Image.open(data_root / "reference.png").convert("RGB")
        payload_reference = Image.open(data_root / "payload_reference.png").convert("RGB")
        alpha = Image.open(data_root / "shared_alpha.png").convert("RGB")
        width, height = reference.size
        arrays = np.load(data_root / "controls.npz", allow_pickle=False)
        if sorted(arrays.files) != sorted(EXPECTED_ARMS):
            raise ValueError("Unexpected E1 control arrays")
        for arm in EXPECTED_ARMS:
            if arrays[arm].shape != (inference_cfg["num_frames"], height, width, 3) or arrays[arm].dtype != np.uint8:
                raise ValueError(f"Invalid E1 control tensor: {arm}")

        os.environ.update(runtime["offline_environment"])
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HOME": str(args.output_root / "cache"),
            "TRANSFORMERS_CACHE": str(args.output_root / "cache"),
            "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["model_root"]).parent.parent),
        })
        sys.path.insert(0, runtime["geoedit_root"])
        assert_no_foreign_process()
        threading.Thread(target=resource_watch, daemon=True).start()
        state["status"] = "loading_pipeline"
        write_json(args.output_root / "run_manifest.json", state)
        from geoedit import inference

        with (args.output_root / "pipeline.log").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
        state["pipeline_load_count"] = 1

        for arm in EXPECTED_ARMS:
            assert_no_foreign_process()
            condition_root = args.output_root / arm
            condition_root.mkdir(exist_ok=False)
            begin = time.monotonic()
            state.update(status="running", active_condition=arm)
            write_json(args.output_root / "run_manifest.json", state)
            references = [reference]
            if arm == "structure_scribble_payload_reference":
                references.append(payload_reference)
            with (condition_root / "inference.log").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                video = pipe(
                    prompt=case["prompt"],
                    negative_prompt=case["negative_prompt"],
                    height=height,
                    width=width,
                    num_frames=inference_cfg["num_frames"],
                    num_inference_steps=inference_cfg["num_inference_steps"],
                    vace_video=[Image.fromarray(frame) for frame in arrays[arm]],
                    vace_video_mask=[alpha] * inference_cfg["num_frames"],
                    vace_reference_image=references,
                    vace_scale=inference_cfg["vace_scale"],
                    enable_ttm=inference_cfg["enable_ttm"],
                    seed=inference_cfg["seed"],
                    tiled=True,
                )
                if len(video) != inference_cfg["num_frames"]:
                    raise ValueError("Unexpected generated frame count")
                lossless = condition_root / "raw_frames"
                lossless.mkdir()
                for index, frame in enumerate(video):
                    frame.save(lossless / f"{index:02d}.png")
                inference.save_video(video, str(condition_root / "raw.mp4"), fps=inference_cfg["fps"], quality=5)
                del video
            project_outputs(
                condition_root / "raw.mp4",
                data_root / "reference.png",
                data_root / "shared_alpha.png",
                condition_root,
                inference_cfg["num_frames"],
                inference_cfg["fps"],
                inference_cfg["review_frame_indices"],
            )
            assert_no_foreign_process()
            record = {
                "condition": arm,
                "reference_count": len(references),
                "wall_seconds": time.monotonic() - begin,
                "files": {
                    str(path.relative_to(condition_root)): {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in condition_root.rglob("*") if path.is_file()
                },
            }
            write_json(condition_root / "condition_manifest.json", record)
            state["completed_conditions"].append(record)
            write_json(args.output_root / "run_manifest.json", state)
        state["status"] = "complete_requires_blind_visual_review"
    except BaseException as exc:
        state.update(status="technical_failure_preserved", error=repr(exc))
        (args.output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        done.set()
        signal.alarm(0)
        state.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.monotonic() - start,
        )
        write_json(args.output_root / "run_manifest.json", state)
    return 0 if state["status"] == "complete_requires_blind_visual_review" else 3


if __name__ == "__main__":
    raise SystemExit(main())
