#!/usr/bin/env python3
"""Build an anonymized Day 13 review package from a completed evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config.resolve())
    if (config.get("schema_version") != "foodstateedit.flexible_completion_blind_review.v1"
            or not config.get("frozen_before_evaluation_output_was_seen")
            or not config.get("mapping_must_not_be_given_to_reviewers")):
        raise ValueError("Unexpected or non-frozen blind review config")
    aliases = config["aliases"]
    if set(aliases) != {f"R{i}" for i in range(1, 6)} or len(set(aliases.values())) != 5:
        raise ValueError("Aliases must be a one-to-one R1..R5 mapping")
    evaluation = args.evaluation_root.resolve()
    manifest_path = evaluation / "run_manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "complete_requires_two_blinded_semantic_and_photo_reviewers":
        raise ValueError("Evaluation is not complete and ready for independent review")
    if set(manifest.get("conditions", {})) != set(aliases.values()):
        raise ValueError("Blind mapping and evaluation conditions differ")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = []
    for alias, condition in aliases.items():
        for suffix in ("projected_review.png", "projected_final_hold.png"):
            source = evaluation / f"{condition}_{suffix}"
            destination = output / f"{alias}_{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)
            evidence.append({"alias": alias, "kind": suffix, "size_bytes": destination.stat().st_size,
                             "sha256": sha(destination)})
    instructions = """# FoodStateEdit 匿名评审表\n\n请不要查看条件映射或实验目录。R1–R5 的顺序不代表优劣。\n请独立评审，不与另一位评审者讨论。\n\n对每个 R 条件分别按 1（差）到 5（好）评分：\n\n- 筷子与面条的夹取接触\n- 面条连续性（没有断裂或漂浮）\n- 面条与碗内食物的连接\n- 抬升动作是否清楚\n- 最终保持姿态是否稳定\n- 照片真实感\n\n然后回答：\n\n1. 在夹取、连续抬升和最终保持上，R3 与 R4 哪个更好？可选 R3 / R4 / 相同 / 都失败。\n2. R3 与 R4 相比，照片真实感是否更差？可选 R3 更差 / R4 更差 / 相同。\n3. 写出最明显的伪影与判断理由。\n\n评审者编号：____  日期：____\n"""
    write_new(output / "REVIEW_FORM.md", instructions)
    audit = {"schema_version": "foodstateedit.flexible_completion_blind_package.v1",
             "source_manifest_sha256": sha(manifest_path), "blind_config_sha256": sha(args.config.resolve()),
             "files": evidence, "mapping_included": False, "reviewer_count_required": 2}
    write_new(output / "PACKAGE_AUDIT.json", json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
