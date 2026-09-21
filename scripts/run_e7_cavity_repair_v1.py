"""Run E7 corrected-material hard/soft projection on frozen controls."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from bite_remain_consistency import composite
from run_high_lift_vace_pilot import (resource_preflight, run_text, sha256,
                                      validate_file, write_json)
from state_projection_control import validate_state_projection_schedule


def make_sheet(frames):
    width, height = frames[0].size
    sheet = Image.new("RGB", (width * 3, (height + 30) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, frame_id in enumerate([0, 8, 9, 12, 15, 20]):
        x = (slot % 3) * width
        y = (slot // 3) * (height + 30)
        sheet.paste(frames[frame_id], (x, y + 30))
        draw.text((x + 4, y + 5), f"E7 cavity repair / f{frame_id}", fill="black")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = args.controls / "manifest.json"
    if sha256(manifest_path) != cfg["controls_manifest_sha256"]:
        raise ValueError("controls manifest mismatch")
    controls_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if controls_manifest["schema_version"] != "foodstateedit.bite_remain_controls.v1":
        raise ValueError("unexpected controls schema")
    for relative, record in controls_manifest["files"].items():
        validate_file(args.controls / relative, record)
    frozen_cfg = controls_manifest["config"]
    for key in ("seed", "num_frames", "num_inference_steps", "vace_scale", "prompt", "negative_prompt"):
        if cfg[key] != frozen_cfg[key]:
            raise ValueError(f"E5b causal-control mismatch: {key}")
    if frozen_cfg.get("reference_mode") != "cavity_anchor":
        raise ValueError("E6 requires the frozen E5b cavity reference")
    if sha256(args.base_config) != cfg["base_config_sha256"]:
        raise ValueError("base config mismatch")
    if args.output_root.exists() or args.preflight_report.exists():
        raise FileExistsError("new output and preflight paths are required")

    algorithm = cfg["algorithm"]
    schedule = validate_state_projection_schedule(
        cfg["num_inference_steps"],
        algorithm["replace_start_index"],
        algorithm["rigid_tstrong_index"],
        algorithm["cavity_tstrong_index"],
    )
    if cfg.get("variant") not in ("hard", "soft"):
        raise ValueError("E7 variant must be hard or soft")
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    gate = resource_preflight(base_config, args.output_root, args.preflight_report, args.gpu)
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    done = threading.Event()
    state = {
        "schema_version": "foodstateedit.cavity_repair_run.v1",
        "variant": cfg["variant"],
        "render_helper_sha256": sha256(Path(__file__).with_name("cavity_state_projection.py")),
        "status": "initializing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "physical_gpu": args.gpu,
        "pipeline_load_count": 0,
        "config_sha256": sha256(args.config),
        "controls_manifest_sha256": cfg["controls_manifest_sha256"],
        "runner_sha256": sha256(Path(__file__)),
        "state_projection_helper_sha256": sha256(Path(__file__).with_name("state_projection_control.py")),
        "base_config_sha256": sha256(args.base_config),
        "seed": cfg["seed"],
        "num_frames": cfg["num_frames"],
        "num_inference_steps": cfg["num_inference_steps"],
        "vace_scale": cfg["vace_scale"],
        "lora_enabled": False,
        "enable_ttm": True,
        "projection_schedule": schedule,
        "causal_baseline": cfg["causal_baseline"],
        "claim_limit": cfg["claim_limit"],
    }
    write_json(args.output_root / "run_manifest.json", state)

    def abort(signum, frame):
        raise RuntimeError("own worker stopped at resource or deadline boundary")

    def check_resource():
        rows = run_text([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        for row in rows.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if fields and fields[0] == gate["gpu"]["uuid"] and int(fields[1]) != os.getpid():
                raise RuntimeError("foreign GPU process appeared: " + row)

    def watch():
        while not done.wait(10):
            try:
                check_resource()
            except Exception as exc:
                (args.output_root / "resource_stop.txt").write_text(repr(exc), encoding="utf-8")
                os.kill(os.getpid(), signal.SIGTERM)
                return

    signal.signal(signal.SIGALRM, abort)
    signal.signal(signal.SIGTERM, abort)
    signal.alarm(cfg["hard_timeout_seconds"])
    try:
        runtime = base_config["runtime"]
        validate_file(Path(runtime["model_hash_audit"]["path"]), runtime["model_hash_audit"])
        for relative, digest in runtime["geoedit_files"].items():
            if sha256(Path(runtime["geoedit_root"]) / relative) != digest:
                raise ValueError("frozen runtime changed: " + relative)
        for relative, record in runtime["model_files"].items():
            validate_file(Path(runtime["model_root"]) / relative, record, hash_file=False)

        with np.load(args.controls / "controls.npz", allow_pickle=False) as arrays:
            base = arrays["baseline"]
            condition = arrays["condition"]
            mask = arrays["generation_mask"]
            alpha = arrays["projection_alpha"]
            payload = arrays["planned_payload"]
        if base.shape != condition.shape or base.shape[0] != cfg["num_frames"]:
            raise ValueError("control shape mismatch")
        if mask.shape != base.shape[:3] or alpha.shape != mask.shape or payload.shape != mask.shape:
            raise ValueError("mask shape mismatch")
        if not np.isin(mask, [0, 255]).all() or np.any(alpha[mask == 0]):
            raise ValueError("invalid mask contract")
        if np.any(condition[mask == 0] != base[mask == 0]):
            raise ValueError("inactive condition changed")
        if np.any(alpha[payload > 0]):
            raise ValueError("planned payload is not protected")

        height, width = base.shape[1:3]
        reference = Image.open(args.controls / "cavity_anchor.png").convert("RGB")
        zero_mask = np.zeros((height, width), np.uint8)
        motion_video = [Image.fromarray(frame) for frame in condition]
        motion_mask = [Image.fromarray(zero_mask).convert("RGB") for _ in range(cfg["num_frames"])]
        cavity_mask = [Image.fromarray(frame).convert("RGB") for frame in mask]

        os.environ.update(runtime["offline_environment"])
        os.environ.update(
            CUDA_VISIBLE_DEVICES=str(args.gpu),
            HF_HOME=str(args.output_root / "cache"),
            TRANSFORMERS_CACHE=str(args.output_root / "cache"),
            DIFFSYNTH_MODEL_BASE_PATH=str(Path(runtime["model_root"]).parent.parent),
        )
        sys.path.insert(0, runtime["geoedit_root"])
        check_resource()
        threading.Thread(target=watch, daemon=True).start()
        from geoedit import inference

        with (args.output_root / "inference.log").open("x") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            state["status"] = "loading_pipeline"
            write_json(args.output_root / "run_manifest.json", state)
            pipe = inference.load_pipeline(inference.resolve_vram_limit(None))
            pipe.cavity_projection_policy = cfg["soft_projection"] if cfg["variant"] == "soft" else None
            pipe.cavity_projection_trace = []
            state.update(pipeline_load_count=1, status="running")
            write_json(args.output_root / "run_manifest.json", state)
            video = pipe(
                prompt=cfg["prompt"],
                negative_prompt=cfg["negative_prompt"],
                height=height,
                width=width,
                num_frames=cfg["num_frames"],
                num_inference_steps=cfg["num_inference_steps"],
                vace_video=motion_video,
                vace_video_mask=cavity_mask,
                vace_reference_image=[reference],
                vace_scale=cfg["vace_scale"],
                enable_ttm=True,
                motion_signal_video=motion_video,
                motion_signal_mask=motion_mask,
                ttm_hole_mask=cavity_mask,
                ttm_warm_start=algorithm["warm_start"],
                ttm_replace_mode=algorithm["replace_mode"],
                ttm_disable_vhi=algorithm["disable_vhi"],
                ttm_initial_clean=algorithm["initial_clean"],
                tweak_index=algorithm["replace_start_index"],
                tstrong_index=algorithm["rigid_tstrong_index"],
                hole_tstrong_index=algorithm["cavity_tstrong_index"],
                seed=cfg["seed"],
                tiled=True,
            )
            write_json(args.output_root / "projection_trace.json", pipe.cavity_projection_trace)
            if cfg["variant"] == "soft" and len(pipe.cavity_projection_trace) != algorithm["cavity_tstrong_index"]:
                raise AssertionError("soft projection did not execute at every scheduled step")
            if len(video) != cfg["num_frames"]:
                raise ValueError("output frame count mismatch")
            raw_dir = args.output_root / "raw_frames"
            final_dir = args.output_root / "projected_frames"
            raw_dir.mkdir(); final_dir.mkdir()
            projected = []
            outside_error = 0
            payload_error = 0
            for frame_id, image in enumerate(video):
                raw = np.asarray(image.convert("RGB"))
                image.convert("RGB").save(raw_dir / f"{frame_id:02d}.png")
                final = composite(base[frame_id], raw, alpha[frame_id])
                difference = np.abs(final.astype(np.int16) - base[frame_id].astype(np.int16))
                outside_error = max(outside_error, int(difference[alpha[frame_id] == 0].max(initial=0)))
                payload_error = max(payload_error, int(difference[payload[frame_id] > 0].max(initial=0)))
                rendered = Image.fromarray(final)
                rendered.save(final_dir / f"{frame_id:02d}.png")
                projected.append(rendered)
            if outside_error or payload_error:
                raise AssertionError("protected pixels changed")
            make_sheet(projected).save(args.output_root / "projected_review.png")
            projected[-1].save(args.output_root / "projected_final_hold.png")
            inference.save_video(projected, str(args.output_root / "projected.mp4"), fps=cfg["fps"], quality=5)
        check_resource()
        state.update(
            status="complete_requires_visual_review",
            projected_frames=len(projected),
            outside_repair_max_difference=outside_error,
            planned_payload_max_difference=payload_error,
        )
    except BaseException as exc:
        state.update(status="technical_failure_preserved", error=repr(exc))
        (args.output_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        done.set()
        signal.alarm(0)
        state.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.monotonic() - started,
        )
        write_json(args.output_root / "run_manifest.json", state)
        records = {
            file.relative_to(args.output_root).as_posix(): {
                "sha256": sha256(file), "size_bytes": file.stat().st_size,
            }
            for file in args.output_root.rglob("*") if file.is_file()
        }
        write_json(args.output_root / "files_manifest.json", records)
    return 0 if state["status"] == "complete_requires_visual_review" else 3


if __name__ == "__main__":
    raise SystemExit(main())
