"""Command-line inference for GeoEdit dual-branch denoising.

This file contains only the public inference path. Experiment-specific
attention hooks and local filesystem assumptions are intentionally omitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


DEFAULT_NEGATIVE_PROMPT = (
    "重复的物体，色调艳丽，过曝，动态，突兀，细节模糊不清，字幕，风格，作品，画作，画面，"
    "整体发灰，棱镜状玻璃几何拉伸，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，闪动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".gif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GeoEdit inference with Wan2.2-VACE-Fun-A14B"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing prompt.txt, first_frame.*, and depth, "
            "motion_signal, and mask inputs as videos or PNG/JPEG images."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--prompt", default=None, help="Overrides input-dir/prompt.txt.")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Overrides input-dir/negative_prompt.txt.",
    )
    parser.add_argument("--control-video", type=Path, default=None)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--motion-signal-video", type=Path, default=None)
    parser.add_argument("--motion-signal-mask", type=Path, default=None)
    parser.add_argument(
        "--contact-mask",
        type=Path,
        default=None,
        help="Optional contact/topology layer with its own projection endpoint.",
    )
    parser.add_argument(
        "--material-mask",
        type=Path,
        default=None,
        help=(
            "Optional secondary mask for deformable/material content. It can "
            "use a shorter proxy-injection window than the rigid mask."
        ),
    )
    parser.add_argument(
        "--hole-mask",
        type=Path,
        default=None,
        help="Optional source-repair layer with its own projection endpoint.",
    )
    parser.add_argument("--mask-old", type=Path, default=None)
    parser.add_argument("--tweak-index", type=int, default=3)
    parser.add_argument("--tstrong-index", type=int, default=15)
    parser.add_argument(
        "--contact-tstrong-index",
        type=int,
        default=None,
        help="Last denoising-step index for contact-mask proxy injection.",
    )
    parser.add_argument(
        "--material-tstrong-index",
        type=int,
        default=None,
        help=(
            "Last denoising-step index for material-mask proxy injection. "
            "Defaults to --tstrong-index."
        ),
    )
    parser.add_argument(
        "--hole-tstrong-index",
        type=int,
        default=None,
        help="Last denoising-step index for hole-mask/source-repair injection.",
    )
    parser.add_argument(
        "--replace-mode",
        choices=("mask_new", "non_hole"),
        default="non_hole",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Start from random noise instead of the reference branch at tweak-index.",
    )
    parser.add_argument(
        "--disable-vhi",
        action="store_true",
        help="Ablation: inject clean reference latents instead of timestep-matched noisy latents.",
    )
    parser.add_argument(
        "--initial-clean",
        action="store_true",
        help="Ablation: initialize the warm start with clean reference latents.",
    )
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument(
        "--vram-limit",
        type=float,
        default=None,
        help="Usable GPU memory in GiB. By default, total memory minus 2 GiB.",
    )
    return parser


def read_prompt(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def resolve_existing(explicit: Path | None, input_dir: Path, *names: str) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Input does not exist: {path}")
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Expected one of {', '.join(names)} under {input_dir}."
    )


def load_video_frames(video: VideoData, limit: int | None = None) -> list[Image.Image]:
    length = len(video) if limit is None else min(len(video), limit)
    return [video[index] for index in range(length)]


def load_video_or_image(
    path: Path, height: int, width: int, num_frames: int
) -> VideoData | list[Image.Image]:
    """Load a video, or repeat one still image for the requested duration."""
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return VideoData(str(path), height=height, width=width)
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            frame = image.convert("RGB").resize((width, height))
        return [frame.copy() for _ in range(num_frames)]
    raise ValueError(
        f"Unsupported input format for {path}. Expected a video, PNG, JPG, or JPEG."
    )


def load_mask_frames(
    path: Path, height: int, width: int, num_frames: int
) -> list[Image.Image]:
    if path.suffix.lower() in VIDEO_SUFFIXES:
        frames = load_video_frames(
            VideoData(str(path), height=height, width=width), num_frames
        )
    else:
        frames = [Image.open(path).convert("L").resize((width, height))]
    if len(frames) == 1:
        frames *= num_frames
    if len(frames) < num_frames:
        raise ValueError(
            f"Mask {path} has {len(frames)} frames; expected at least {num_frames}."
        )
    return [frame.resize((width, height)) for frame in frames[:num_frames]]


def compute_hole_masks(
    old_frames: list[Image.Image], new_frames: list[Image.Image], dilation: int = 5
) -> list[Image.Image]:
    if dilation % 2 == 0:
        dilation += 1
    holes: list[Image.Image] = []
    for old_frame, new_frame in zip(old_frames, new_frames):
        old_mask = old_frame.convert("L")
        if dilation > 1:
            old_mask = old_mask.filter(ImageFilter.MaxFilter(dilation))
        old_bool = np.asarray(old_mask) > 127
        new_bool = np.asarray(new_frame.convert("L")) > 127
        hole = np.logical_and(old_bool, np.logical_not(new_bool))
        holes.append(Image.fromarray(hole.astype(np.uint8) * 255, mode="L").convert("RGB"))
    return holes


def validate_args(args: argparse.Namespace) -> None:
    replace_mode = getattr(args, "replace_mode", "mask_new")
    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num-frames must be 4n+1 (for example 21, 41, 61, or 81).")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive.")
    if not 0 <= args.tweak_index < args.num_inference_steps:
        raise ValueError("--tweak-index must be in [0, num-inference-steps).")
    if not args.tweak_index <= args.tstrong_index <= args.num_inference_steps:
        raise ValueError(
            "--tstrong-index must be between tweak-index and num-inference-steps."
        )
    staged_layers = (
        ("contact", getattr(args, "contact_mask", None), getattr(args, "contact_tstrong_index", None)),
        ("material", getattr(args, "material_mask", None), getattr(args, "material_tstrong_index", None)),
        ("hole", getattr(args, "hole_mask", None), getattr(args, "hole_tstrong_index", None)),
    )
    for name, mask, endpoint in staged_layers:
        if endpoint is not None and mask is None:
            raise ValueError(f"--{name}-tstrong-index requires --{name}-mask.")
        if endpoint is not None and not (
            args.tweak_index <= endpoint <= args.num_inference_steps
        ):
            raise ValueError(
                f"--{name}-tstrong-index must be between tweak-index and "
                "num-inference-steps."
            )
        if mask is not None and replace_mode != "mask_new":
            raise ValueError(
                f"Staged {name}-mask injection requires --replace-mode mask_new."
            )


def load_optional_layer_mask(
    path: Path | None, height: int, width: int, num_frames: int
) -> VideoData | list[Image.Image] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input does not exist: {resolved}")
    return load_video_or_image(resolved, height, width, num_frames)


def resolve_vram_limit(explicit_limit: float | None) -> float:
    if explicit_limit is not None:
        return explicit_limit
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GeoEdit inference requires an NVIDIA GPU.")
    return torch.cuda.mem_get_info("cuda:0")[1] / (1024**3) - 2.0


def load_pipeline(vram_limit: float) -> WanVideoPipeline:
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda:0",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda:0",
    }
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        redirect_common_files=False,
        vram_limit=vram_limit,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    control_path = resolve_existing(
        args.control_video,
        input_dir,
        "depth.mp4",
        "depth.png",
        "depth.jpg",
        "depth.jpeg",
    )
    reference_path = resolve_existing(
        args.reference_image, input_dir, "first_frame.png", "first_frame.jpg", "first_frame.jpeg"
    )
    motion_path = resolve_existing(
        args.motion_signal_video,
        input_dir,
        "motion_signal.mp4",
        "motion_signal.png",
        "motion_signal.jpg",
        "motion_signal.jpeg",
    )
    new_mask_path = resolve_existing(
        args.motion_signal_mask,
        input_dir,
        "mask.mp4",
        "mask.png",
        "mask.jpg",
        "mask.jpeg",
    )

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = read_prompt(resolve_existing(None, input_dir, "prompt.txt"))
    negative_path = input_dir / "negative_prompt.txt"
    negative_prompt = (
        args.negative_prompt
        if args.negative_prompt is not None
        else read_prompt(negative_path) if negative_path.exists() else DEFAULT_NEGATIVE_PROMPT
    )

    with Image.open(reference_path) as image:
        reference_image = image.convert("RGB")
        width, height = reference_image.size
    # Wan VAE requires dimensions divisible by 16. Match DiffSynth's rounding
    # before constructing VideoData so every branch has the same shape.
    height = (height + 15) // 16 * 16
    width = (width + 15) // 16 * 16
    reference_image = reference_image.resize((width, height))

    control_video = load_video_or_image(
        control_path, height, width, args.num_frames
    )
    motion_video = load_video_or_image(
        motion_path, height, width, args.num_frames
    )
    motion_mask = load_video_or_image(
        new_mask_path, height, width, args.num_frames
    )
    contact_mask = load_optional_layer_mask(
        args.contact_mask, height, width, args.num_frames
    )
    material_mask = load_optional_layer_mask(
        args.material_mask, height, width, args.num_frames
    )
    staged_hole_mask = load_optional_layer_mask(
        args.hole_mask, height, width, args.num_frames
    )

    old_mask_path = args.mask_old.expanduser().resolve() if args.mask_old else None
    if old_mask_path is not None and not old_mask_path.exists():
        raise FileNotFoundError(f"Input does not exist: {old_mask_path}")
    if old_mask_path is None:
        for name in ("mask_old.png", "mask_old.jpg", "mask_old.jpeg", "mask_old.mp4"):
            candidate = input_dir / name
            if candidate.exists():
                old_mask_path = candidate
                break

    hole_masks = None
    if args.replace_mode == "non_hole":
        if old_mask_path is None:
            raise FileNotFoundError(
                "replace-mode=non_hole requires mask_old as a video, PNG, JPG, or JPEG."
            )
        old_frames = load_mask_frames(old_mask_path, height, width, args.num_frames)
        new_frames = load_mask_frames(new_mask_path, height, width, args.num_frames)
        hole_masks = compute_hole_masks(old_frames, new_frames)

    pipe = load_pipeline(resolve_vram_limit(args.vram_limit))
    height, width = pipe.check_resize_height_width(height, width)

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        vace_video=control_video,
        vace_reference_image=reference_image,
        enable_ttm=True,
        motion_signal_video=motion_video,
        motion_signal_mask=motion_mask,
        ttm_contact_mask=contact_mask,
        ttm_material_mask=material_mask,
        ttm_hole_mask=staged_hole_mask,
        ttm_mask_old=hole_masks,
        ttm_warm_start=not args.no_warm_start,
        ttm_replace_mode=args.replace_mode,
        ttm_disable_vhi=args.disable_vhi,
        ttm_initial_clean=args.initial_clean,
        tweak_index=args.tweak_index,
        tstrong_index=args.tstrong_index,
        contact_tstrong_index=args.contact_tstrong_index,
        material_tstrong_index=args.material_tstrong_index,
        hole_tstrong_index=args.hole_tstrong_index,
        seed=args.seed,
        tiled=True,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not video:
        raise RuntimeError("The pipeline returned no output frames.")
    preview_output = output.with_suffix(".png")
    video[0].save(preview_output)
    print(f"Saved first frame: {preview_output}")
    save_video(video, str(output), fps=args.fps, quality=args.quality)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
