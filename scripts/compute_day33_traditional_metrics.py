#!/usr/bin/env python3
"""Compute reference-preservation metrics for the frozen Day25--27 outputs.

These metrics compare each edited image with its input.  They measure image
fidelity/preservation, not whether the requested utensil action is correct.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "results" / "day25_ablation_matrix_verified_v1.json"
OUT_DIR = ROOT / "artifacts" / "day33_traditional_metrics_v1"
JSON_PATH = ROOT / "results" / "day33_traditional_metrics_v1.json"
MD_PATH = ROOT / "results" / "DAY33_TRADITIONAL_METRICS_20260913.md"

CASES = ("ramen", "soup", "rice", "cake")
SEEDS = (1, 2, 3)
POLICY_CONDITION = {
    "ramen": "relative3d_scale_0p8",
    "soup": "relative3d_scale_1p0",
    "rice": "relative3d_scale_1p0",
    "cake": "relative3d_scale_0p6",
}
METHODS = (
    "ChordEdit",
    "Qwen-Image-Edit",
    "Native VACE",
    "2D planar proxy",
    "fixed relative3D",
    "FoodStateEdit material-policy candidate",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decoded_pixel_sha256(path: Path, mode: str) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert(mode))
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> tuple[np.ndarray, bool]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        resized = size is not None and image.size != size
        if resized:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64), resized


def gaussian_blur(channel: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius = 5
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    padded_x = np.pad(channel, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = sum(kernel[i] * padded_x[:, i : i + channel.shape[1]] for i in range(2 * radius + 1))
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    return sum(kernel[i] * padded_y[i : i + channel.shape[0], :] for i in range(2 * radius + 1))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Wang-style local SSIM using an 11-pixel Gaussian neighbourhood (sigma/radius 1.5).
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    maps = []
    for channel in range(3):
        x = a[:, :, channel]
        y = b[:, :, channel]
        mu_x = gaussian_blur(x)
        mu_y = gaussian_blur(y)
        sigma_x = gaussian_blur(x * x) - mu_x * mu_x
        sigma_y = gaussian_blur(y * y) - mu_y * mu_y
        sigma_xy = gaussian_blur(x * y) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        maps.append(np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator != 0))
    return np.mean(np.stack(maps, axis=2), axis=2)


def psnr_from_mse(mse: float) -> float:
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((255.0**2) / mse)


def metrics(reference: np.ndarray, output: np.ndarray, edit_mask: np.ndarray) -> dict[str, float]:
    diff = output - reference
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    full_ssim = ssim_map(reference, output)

    outside = ~edit_mask
    outside_rgb = np.repeat(outside[:, :, None], 3, axis=2)
    outside_values = diff[outside_rgb]
    outside_abs = abs_diff[outside_rgb]
    outside_mse = float(np.mean(outside_values * outside_values))

    # Exclude an 11-pixel-wide neighbourhood around the edit support when
    # averaging local SSIM so edited pixels do not leak into outside-support windows.
    mask_image = Image.fromarray((edit_mask.astype(np.uint8) * 255), mode="L")
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(11))) > 0
    outside_ssim_valid = ~dilated
    inside_rgb = np.repeat(edit_mask[:, :, None], 3, axis=2)
    changed_gt10 = np.max(abs_diff, axis=2) > 10.0

    return {
        "full_psnr_db": psnr_from_mse(mse),
        "full_ssim": float(np.mean(full_ssim[5:-5, 5:-5])),
        "full_rgb_mae": float(np.mean(abs_diff)),
        "outside_psnr_db": psnr_from_mse(outside_mse),
        "outside_ssim": float(np.mean(full_ssim[outside_ssim_valid])),
        "outside_rgb_mae": float(np.mean(outside_abs)),
        "outside_fraction_pixels_max_channel_gt_10": float(np.mean(changed_gt10[outside])),
        "outside_exact_pixel_fraction": float(np.mean(np.max(abs_diff, axis=2)[outside] == 0.0)),
        "inside_rgb_mae_change_magnitude": float(np.mean(abs_diff[inside_rgb])),
    }


def bootstrap_case_ci(records: list[dict], key: str, iterations: int = 20000) -> list[float]:
    by_case = {
        case: float(np.mean([r[key] for r in records if r["case_id"] == case]))
        for case in CASES
    }
    values = np.asarray([by_case[case] for case in CASES], dtype=np.float64)
    rng = np.random.default_rng(2530030)
    samples = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def resolve_day25_cells() -> dict[tuple[str, int, str], Path]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    cells = {}
    for cell in matrix["cells"]:
        path = ROOT / Path(cell["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != cell["sha256"]:
            raise RuntimeError(f"Hash mismatch: {path}")
        cells[(cell["case_id"], int(cell["seed"]), cell["condition"])] = path
    return cells


def output_path(method: str, case: str, seed: int, cells: dict[tuple[str, int, str], Path]) -> Path:
    if method == "ChordEdit":
        return ROOT / "artifacts" / "day26_chordedit_direct_baseline_v1" / case / f"seed_{seed}" / "final.png"
    if method == "Qwen-Image-Edit":
        return ROOT / "artifacts" / "day27_qwen_image_edit_direct_baseline_v1" / case / f"seed_{seed}" / "raw_qwen.png"
    condition = {
        "Native VACE": "native_scale_1p0",
        "2D planar proxy": "planar_scale_1p0",
        "fixed relative3D": "relative3d_scale_1p0",
        "FoodStateEdit material-policy candidate": POLICY_CONDITION[case],
    }[method]
    return cells[(case, seed, condition)]


def finite_mean(values: list[float]) -> float:
    if all(math.isinf(value) for value in values):
        return math.inf
    return float(np.mean(values))


def format_num(value: float, digits: int) -> str:
    return "inf" if math.isinf(value) else f"{value:.{digits}f}"


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = resolve_day25_cells()
    records: list[dict] = []
    input_hashes: dict[str, str] = {}
    mask_hashes: dict[str, str] = {}

    for case in CASES:
        baseline_case = ROOT / "artifacts" / "day26_chordedit_direct_baseline_v1" / case
        canonical_input = baseline_case / "seed_1" / "input.png"
        canonical_mask = baseline_case / "seed_1" / "allowed_edit_region.png"
        input_hashes[case] = sha256(canonical_input)
        mask_hashes[case] = sha256(canonical_mask)
        input_pixel_hash = decoded_pixel_sha256(canonical_input, "RGB")
        mask_pixel_hash = decoded_pixel_sha256(canonical_mask, "L")
        with Image.open(canonical_input) as image:
            reference_size = image.size
        reference, _ = load_rgb(canonical_input)
        with Image.open(canonical_mask) as image:
            # The projectors treat every non-zero alpha value as editable; using
            # 127 here would incorrectly count anti-aliased support-edge pixels
            # as protected background.
            edit_mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if not edit_mask.any() or edit_mask.all():
            raise RuntimeError(f"Invalid support mask: {canonical_mask}")

        dataset_case = "noodle" if case == "ramen" else case
        dataset_input = ROOT / "artifacts" / "day19_multimaterial_dataset_v3" / dataset_case / "reference.png"
        dataset_mask = ROOT / "artifacts" / "day19_multimaterial_dataset_v3" / dataset_case / "edit_alpha.png"
        if decoded_pixel_sha256(dataset_input, "RGB") != input_pixel_hash or decoded_pixel_sha256(dataset_mask, "L") != mask_pixel_hash:
            raise RuntimeError(f"Baseline/dataset input or mask mismatch for {case}")

        for seed in SEEDS:
            seed_input = baseline_case / f"seed_{seed}" / "input.png"
            seed_mask = baseline_case / f"seed_{seed}" / "allowed_edit_region.png"
            if decoded_pixel_sha256(seed_input, "RGB") != input_pixel_hash or decoded_pixel_sha256(seed_mask, "L") != mask_pixel_hash:
                raise RuntimeError(f"Seed input or mask mismatch for {case} seed {seed}")
            qwen_input = ROOT / "artifacts" / "day27_qwen_image_edit_direct_baseline_v1" / case / f"seed_{seed}" / "input.png"
            qwen_mask = ROOT / "artifacts" / "day27_qwen_image_edit_direct_baseline_v1" / case / f"seed_{seed}" / "allowed_edit_region.png"
            if decoded_pixel_sha256(qwen_input, "RGB") != input_pixel_hash or decoded_pixel_sha256(qwen_mask, "L") != mask_pixel_hash:
                raise RuntimeError(f"Qwen input or mask mismatch for {case} seed {seed}")

            for method in METHODS:
                path = output_path(method, case, seed, cells)
                if not path.is_file():
                    raise FileNotFoundError(path)
                output, resized = load_rgb(path, reference_size)
                record = {
                    "method": method,
                    "case_id": case,
                    "seed": seed,
                    "output_path": path.relative_to(ROOT).as_posix(),
                    "output_sha256": sha256(path),
                    "resized_to_input_with_lanczos": resized,
                }
                record.update(metrics(reference, output, edit_mask))
                records.append(record)

    metric_keys = [
        "full_psnr_db",
        "full_ssim",
        "full_rgb_mae",
        "outside_psnr_db",
        "outside_ssim",
        "outside_rgb_mae",
        "outside_fraction_pixels_max_channel_gt_10",
        "outside_exact_pixel_fraction",
        "inside_rgb_mae_change_magnitude",
    ]
    aggregate = {}
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        aggregate[method] = {
            "n_outputs": len(subset),
            **{key: finite_mean([record[key] for record in subset]) for key in metric_keys},
            "clustered_bootstrap_95ci_by_case": {
                key: bootstrap_case_ci(subset, key)
                for key in metric_keys
                if not any(math.isinf(record[key]) for record in subset)
            },
        }

    # A paired contrast is descriptive only; the four case images are selected development inputs.
    contrasts = {}
    ours = [r for r in records if r["method"] == "FoodStateEdit material-policy candidate"]
    ours_map = {(r["case_id"], r["seed"]): r for r in ours}
    for comparator in ("ChordEdit", "Qwen-Image-Edit", "Native VACE", "2D planar proxy", "fixed relative3D"):
        comp = [r for r in records if r["method"] == comparator]
        contrasts[comparator] = {}
        for key in metric_keys:
            differences = []
            for row in comp:
                value = ours_map[(row["case_id"], row["seed"])][key] - row[key]
                differences.append(value)
            contrasts[comparator][key] = float(np.mean(differences)) if all(np.isfinite(differences)) else None

    payload = {
        "schema_version": "foodstateedit.day33_traditional_metrics.v1",
        "date": "2026-09-13",
        "status": "computed_from_hash_verified_selected_development_outputs",
        "scope": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "outputs_per_method": 12,
            "total_output_comparisons": len(records),
            "held_out": False,
        },
        "metric_definition": {
            "reference": "same input image for each case; output resized to input size with LANCZOS only when dimensions differ",
            "psnr": "RGB, 8-bit range, higher means closer to the input",
            "ssim": "RGB-channel mean, local Gaussian radius 1.5, constants K1=0.01 and K2=0.03, higher means closer to the input",
            "outside_support": "pixels outside the frozen edit_alpha support; SSIM excludes an 11-pixel dilated support boundary",
            "inside_rgb_mae_change_magnitude": "amount of change inside the editable support; it is not a quality score",
            "interpretation_boundary": "All values are input-fidelity or change-magnitude diagnostics. They do not measure action correctness, payload conservation, contact, or photo realism.",
        },
        "input_sha256": input_hashes,
        "mask_sha256": mask_hashes,
        "aggregate": aggregate,
        "paired_mean_differences_ours_minus_comparator": contrasts,
        "per_output": records,
        "claim_limit": "Four selected development cases and three seeds; cake is synthetic supplementary. No held-out, population, superiority, causal, physical-correctness, or human-preference claim is supported.",
    }
    JSON_PATH.write_text(
        json.dumps(json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    csv_path = OUT_DIR / "per_output_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Day 33 Traditional Image Metrics",
        "",
        "This table compares each edited output with its input. Higher PSNR/SSIM and lower MAE mean stronger input fidelity, not a more correct utensil action.",
        "",
        "| Method | n | Full PSNR ↑ | Full SSIM ↑ | Outside PSNR ↑ | Outside SSIM ↑ | Outside MAE ↓ | Outside exact ↑ | Inside change MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate[method]
        lines.append(
            f"| {method} | {row['n_outputs']} | {format_num(row['full_psnr_db'], 2)} | "
            f"{format_num(row['full_ssim'], 4)} | {format_num(row['outside_psnr_db'], 2)} | "
            f"{format_num(row['outside_ssim'], 4)} | {format_num(row['outside_rgb_mae'], 3)} | "
            f"{format_num(row['outside_exact_pixel_fraction'], 4)} | "
            f"{format_num(row['inside_rgb_mae_change_magnitude'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ChordEdit and all projected VACE variants preserve protected pixels exactly by construction; this is a preservation property, not proof that the requested action occurred.",
            "- Qwen-Image-Edit globally regenerates/reframes the image, so input-fidelity scores include registration and resize effects even when the output looks photographic.",
            "- Full-image metrics penalize the intended edit. They must be read together with Action Success, Photo Success, and Strict End-to-End Success.",
            "- The four case images are selected development examples. The clustered bootstrap intervals in the JSON resample cases and are descriptive only.",
            "",
            "## Reproducibility",
            "",
            "- Per-output CSV: `artifacts/day33_traditional_metrics_v1/per_output_metrics.csv`",
            "- Machine-readable result: `results/day33_traditional_metrics_v1.json`",
            "- Script: `scripts/compute_day33_traditional_metrics.py`",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
