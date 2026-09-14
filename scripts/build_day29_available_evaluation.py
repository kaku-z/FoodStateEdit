#!/usr/bin/env python3
"""Summarize available baselines and build a method-blind review package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


ENDPOINTS = (
    "action_success",
    "photo_success",
    "preservation_success",
    "strict_end_to_end_success",
)
CASES = ("ramen", "soup", "rice", "cake")
CASE_LABELS = {
    "ramen": "Use one pair of chopsticks to lift a supported noodle payload.",
    "soup": "Use one spoon to lift a retained soup payload without spilling.",
    "rice": "Use one serving spatula to lift a coherent fried-rice payload.",
    "cake": "Use one fork to lift the pre-cut cake bite, leaving a matching source gap.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def case_rates(records: list[dict[str, Any]], endpoint: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(float(bool(record[endpoint])))
    if set(grouped) != set(CASES):
        raise ValueError(f"Unexpected case set for {endpoint}: {sorted(grouped)}")
    return {case: sum(values) / len(values) for case, values in grouped.items()}


def cluster_bootstrap(
    rates: dict[str, float], *, iterations: int, seed: int
) -> dict[str, Any]:
    clusters = list(CASES)
    point = sum(rates.values()) / len(rates)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [rng.choice(clusters) for _ in clusters]
        draws.append(sum(rates[case] for case in sampled) / len(sampled))
    draws.sort()
    return {
        "estimate": point,
        "ci95_percentile": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "cluster_count": len(clusters),
        "iterations": iterations,
        "seed": seed,
    }


def paired_cluster_bootstrap(
    left: dict[str, float], right: dict[str, float], *, iterations: int, seed: int
) -> dict[str, Any]:
    differences = {case: left[case] - right[case] for case in CASES}
    result = cluster_bootstrap(differences, iterations=iterations, seed=seed)
    result["contrast"] = "qwen_minus_chordedit"
    return result


def load_method_records(path: Path, method_id: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    records = payload["provisional_internal_review"]["per_output"]
    normalized = []
    for record in records:
        item = dict(record)
        item["method_id"] = method_id
        item["strict_end_to_end_success"] = bool(
            item["action_success"]
            and item["photo_success"]
            and item["preservation_success"]
        )
        normalized.append(item)
    expected = {(case, seed) for case in CASES for seed in (1, 2, 3)}
    actual = {(item["case_id"], int(item["seed"])) for item in normalized}
    if actual != expected:
        raise ValueError(f"Unexpected case/seed inventory in {path}")
    return normalized


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def build_review_sheet(source: Path, output: Path, action_text: str, destination: Path) -> None:
    panel_size = (560, 560)
    header_height = 82
    label_height = 28
    canvas = Image.new("RGB", (panel_size[0] * 2, header_height + label_height + panel_size[1]), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 14), "Requested action:", fill="black", font=font)
    draw.text((16, 36), action_text, fill="black", font=font)
    draw.text((16, header_height + 8), "SOURCE", fill="black", font=font)
    draw.text((panel_size[0] + 16, header_height + 8), "OUTPUT", fill="black", font=font)
    canvas.paste(fit_panel(Image.open(source), panel_size), (0, header_height + label_height))
    canvas.paste(fit_panel(Image.open(output), panel_size), (panel_size[0], header_height + label_height))
    draw.line((panel_size[0], header_height, panel_size[0], canvas.height), fill=(128, 128, 128), width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def write_ballot(path: Path, reviewer_id: str, order: Iterable[str]) -> None:
    fields = [
        "reviewer_id",
        "opaque_id",
        *[f"final_q{i}" for i in range(1, 10)],
        "photo_realism_1_to_5",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for opaque_id in order:
            writer.writerow({"reviewer_id": reviewer_id, "opaque_id": opaque_id})


def build_blind_package(
    repo: Path,
    records_by_method: dict[str, list[dict[str, Any]]],
    package_root: Path,
    key_root: Path,
    reviewer_count: int,
    random_seed: int,
) -> dict[str, Any]:
    if package_root.exists() or key_root.exists():
        raise FileExistsError("Blind package or private key root already exists")
    package_root.mkdir(parents=True)
    key_root.mkdir(parents=True)
    salt = hashlib.sha256(f"day29-{random_seed}".encode("ascii")).hexdigest()
    mapping: dict[str, Any] = {}
    source_paths = {
        "chordedit": repo / "artifacts/day26_chordedit_direct_baseline_v1",
        "qwen": repo / "artifacts/day27_qwen_image_edit_direct_baseline_v1",
    }
    endpoint_names = {"chordedit": "final.png", "qwen": "raw_qwen.png"}
    for method_id, records in records_by_method.items():
        for record in records:
            case = record["case_id"]
            seed = int(record["seed"])
            token = f"{salt}:{method_id}:{case}:{seed}".encode("utf-8")
            opaque_id = "E" + hashlib.sha256(token).hexdigest()[:11].upper()
            case_root = source_paths[method_id] / case / f"seed_{seed}"
            source = case_root / "input.png"
            output = case_root / endpoint_names[method_id]
            if not source.is_file() or not output.is_file():
                raise FileNotFoundError(f"Missing review input: {source} or {output}")
            mapping[opaque_id] = {
                "method_id": method_id,
                "case_id": case,
                "seed": seed,
                "source_path": source.relative_to(repo).as_posix(),
                "source_sha256": sha256(source),
                "output_path": output.relative_to(repo).as_posix(),
                "output_sha256": sha256(output),
            }

    opaque_ids = sorted(mapping)
    reviewer_orders: dict[str, list[str]] = {}
    for reviewer_index in range(1, reviewer_count + 1):
        reviewer_id = f"R{reviewer_index}"
        order = list(opaque_ids)
        random.Random(random_seed + reviewer_index).shuffle(order)
        reviewer_orders[reviewer_id] = order
        reviewer_root = package_root / reviewer_id
        (reviewer_root / "items").mkdir(parents=True)
        for position, opaque_id in enumerate(order, start=1):
            item = mapping[opaque_id]
            build_review_sheet(
                repo / item["source_path"],
                repo / item["output_path"],
                CASE_LABELS[item["case_id"]],
                reviewer_root / "items" / f"{position:02d}_{opaque_id}.png",
            )
        write_ballot(reviewer_root / "ballot.csv", reviewer_id, order)

    protocol_source = repo / "benchmark/FOODSTATEEDIT_BLIND_EVALUATION_PROTOCOL_V1.md"
    shutil.copyfile(protocol_source, package_root / "PROTOCOL.md")
    readme = (
        "# Day 29 available-baseline blind review package\n\n"
        "This package contains 24 method-blind final-image items per reviewer. "
        "It covers two available development baselines, four selected cases, and three seeds. "
        "Do not open the separately stored private mapping before every ballot is locked.\n\n"
        "This is not the final held-out FoodStateEdit study: Ours and held-out cases are not yet included.\n"
    )
    (package_root / "README.md").write_text(readme, encoding="utf-8")
    key_payload = {
        "schema_version": "foodstateedit.day29_blind_key.v1",
        "status": "private_do_not_reveal_until_ballots_locked",
        "random_seed": random_seed,
        "mapping": mapping,
        "reviewer_orders": reviewer_orders,
    }
    (key_root / "private_mapping.json").write_text(
        json.dumps(key_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    package_files = [path for path in package_root.rglob("*") if path.is_file()]
    return {
        "reviewers": reviewer_count,
        "items_per_reviewer": len(opaque_ids),
        "method_count": len(records_by_method),
        "case_count": len(CASES),
        "seeds": [1, 2, 3],
        "package_file_count": len(package_files),
        "package_files": {
            path.relative_to(package_root).as_posix(): sha256(path) for path in sorted(package_files)
        },
        "private_mapping_sha256": sha256(key_root / "private_mapping.json"),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for method_id, method in summary["methods"].items():
        for endpoint in ENDPOINTS:
            metric = method["endpoints"][endpoint]
            low, high = metric["ci95_percentile"]
            rows.append(
                f"| {method_id} | {endpoint} | {metric['pass_count']}/{metric['total']} "
                f"({metric['estimate']:.3f}) | [{low:.3f}, {high:.3f}] |"
            )
    contrast_rows = []
    for endpoint, metric in summary["paired_contrasts"].items():
        low, high = metric["ci95_percentile"]
        contrast_rows.append(
            f"| {endpoint} | {metric['estimate']:+.3f} | [{low:+.3f}, {high:+.3f}] |"
        )
    return f"""# Day 29 available-baseline statistics and blind-package start

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {date.today().isoformat()}
- Verification Status: ANALYZED
- Version Label: day29_available_baseline_statistics_v1

## Scope

This analysis covers the already executed ChordEdit and Qwen development baselines only:
four selected cases, three fixed seeds, and one internal non-blind scoring pass. It is not
the held-out benchmark and does not include Ours. The percentile intervals resample only
four image clusters and describe uncertainty inside this selected set; they are not
population-level generalization intervals.

## Descriptive endpoint summary

| Method | Endpoint | Pass / total (rate) | Image-cluster bootstrap 95% interval |
| --- | --- | ---: | ---: |
{chr(10).join(rows)}

## Paired descriptive contrast

The contrast is Qwen minus ChordEdit, paired by case and clustered across the three seeds.

| Endpoint | Difference | Image-cluster bootstrap 95% interval |
| --- | ---: | ---: |
{chr(10).join(contrast_rows)}

No p-values are reported: four selected development clusters and one non-independent
internal review are insufficient for a confirmatory test.

## Blind-review readiness

- Three separately shuffled reviewer packages were generated.
- Each package contains 24 anonymized source/output comparisons and a blank ballot.
- The method key is stored separately and must remain sealed until all ballots are locked.
- The package is a workflow rehearsal only. A final blind study must add frozen Ours and
  the held-out test outputs before recruitment/scoring.

## Fallacy scan

- Coverage: 11/11 checked.
- Simpson's paradox: not assessable with one case per family; aggregate and per-case rates are retained.
- Ecological fallacy: no individual-level inference is made.
- Berkson/selection bias: CAUTION — cases are selected development anchors.
- Collider bias: no covariate adjustment is performed.
- Base-rate neglect: not applicable to these binary editing endpoints.
- Regression to the mean: no extreme-score enrollment or pre/post claim.
- Survivorship bias: no failed generation was removed from the two source result files.
- Look-elsewhere effect: CAUTION — many earlier development diagnostics exist.
- Garden of forking paths: CAUTION — this is exploratory; frozen Day 25 and held-out rules remain mandatory.
- Correlation/causation: no causal effectiveness claim is made from these counts.
- Reverse causality: not applicable to the paired generation design.

## Current gate

Day 25's 47-generation same-condition ablation must complete before choosing the frozen
material-adaptive Ours rule. The 40-image held-out test must remain unopened until that
rule is frozen. FLUX Kontext and ChronoEdit are documented as unavailable rather than
assigned fabricated results.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--result-md", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--key-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    config = read_json(args.config)
    if config.get("schema_version") != "foodstateedit.day29_available_evaluation.v1":
        raise ValueError("Unexpected config schema")
    for relative, expected in config["input_sha256"].items():
        if sha256(repo / relative) != expected:
            raise ValueError(f"Input hash mismatch: {relative}")

    source_paths = {
        "chordedit": repo / "results/day26_chordedit_direct_baseline_result_v1.json",
        "qwen": repo / "results/day27_qwen_image_edit_direct_baseline_result_v1.json",
    }
    records_by_method = {
        method_id: load_method_records(path, method_id) for method_id, path in source_paths.items()
    }
    iterations = int(config["bootstrap"]["iterations"])
    bootstrap_seed = int(config["bootstrap"]["seed"])
    summary: dict[str, Any] = {
        "schema_version": "foodstateedit.day29_available_baseline_statistics.v1",
        "status": "available_development_baselines_analyzed_blind_package_ready_not_scored",
        "verification_status": "ANALYZED",
        "scope": {
            "held_out": False,
            "independent_review": False,
            "methods": list(records_by_method),
            "cases": list(CASES),
            "seeds": [1, 2, 3],
            "outputs": 24,
        },
        "methods": {},
        "paired_contrasts": {},
        "bootstrap_warning": (
            "Intervals resample only four selected image clusters and are descriptive, "
            "not population-level generalization intervals."
        ),
        "fallacy_scan_coverage": "11/11",
        "claim_limit": (
            "These are selected development cases scored once internally and non-blind. "
            "They cannot establish superiority, held-out generalization, independent human preference, "
            "physical correctness, or paper-level benchmark performance."
        ),
    }
    method_case_rates: dict[str, dict[str, dict[str, float]]] = {}
    for method_index, (method_id, records) in enumerate(records_by_method.items()):
        method_case_rates[method_id] = {}
        endpoint_summary = {}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            rates = case_rates(records, endpoint)
            method_case_rates[method_id][endpoint] = rates
            metric = cluster_bootstrap(
                rates,
                iterations=iterations,
                seed=bootstrap_seed + method_index * 100 + endpoint_index,
            )
            metric.update(
                pass_count=sum(bool(record[endpoint]) for record in records),
                total=len(records),
                per_case_rate=rates,
            )
            endpoint_summary[endpoint] = metric
        summary["methods"][method_id] = {"endpoints": endpoint_summary}

    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        summary["paired_contrasts"][endpoint] = paired_cluster_bootstrap(
            method_case_rates["qwen"][endpoint],
            method_case_rates["chordedit"][endpoint],
            iterations=iterations,
            seed=bootstrap_seed + 500 + endpoint_index,
        )

    summary["blind_package"] = build_blind_package(
        repo,
        records_by_method,
        args.package_root,
        args.key_root,
        reviewer_count=int(config["blind_review"]["reviewer_count"]),
        random_seed=int(config["blind_review"]["random_seed"]),
    )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.result_md.write_text(build_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
