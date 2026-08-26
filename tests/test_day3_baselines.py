import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "run_no_edit_baseline.py"
    spec = importlib.util.spec_from_file_location("run_no_edit_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


class Day3BaselineTests(unittest.TestCase):
    def test_frozen_baseline_registry_has_three_primary_controls(self):
        registry = json.loads((ROOT / "configs" / "baselines_v1.json").read_text())
        primary = [
            method for method in registry["methods"]
            if method["role"] in {"primary_control", "primary_baseline"} and method["inclusion"] == "frozen"
        ]
        self.assertEqual(
            {method["method_id"] for method in primary},
            {"input_no_edit", "vanilla_geoedit", "geoedit_unified_action_mask"},
        )
        self.assertEqual(registry["seed_policy"]["stochastic_formal_test"], [20260826, 20260827, 20260828])
        self.assertEqual(registry["day3_gate"]["status"], "OPEN_AWAITING_COMMON_PROXY_V1")

    def test_identity_runner_is_bit_exact_and_refuses_overwrite(self):
        runner, builder_path = load_script()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_csv = root / "canonical.csv"
            anchor_csv = root / "anchors.csv"
            canonical_fields = [
                "case_id", "family", "split", "canonical_input_path", "canonical_input_sha256",
                "canonical_width", "canonical_height"
            ]
            anchor_fields = ["anchor_id", "case_id", "canonical_input_sha256"]
            canonical_rows = []
            anchor_rows = []
            for index, family in enumerate(("liquid", "granular", "strand", "strand_contact"), start=1):
                image = root / f"source_{index}.jpg"
                image.write_bytes(f"fake-jpeg-payload-{index}".encode())
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                canonical_rows.append({
                    "case_id": f"case_{index}", "family": family, "split": "pilot",
                    "canonical_input_path": str(image), "canonical_input_sha256": digest,
                    "canonical_width": "16", "canonical_height": "16",
                })
                anchor_rows.append({
                    "anchor_id": f"anchor_{index}", "case_id": f"case_{index}",
                    "canonical_input_sha256": digest,
                })
            with canonical_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=canonical_fields)
                writer.writeheader()
                writer.writerows(canonical_rows)
            with anchor_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=anchor_fields)
                writer.writeheader()
                writer.writerows(anchor_rows)

            output_root = root / "outputs"
            summary_path = root / "summary.json"
            summary = runner.run_identity_baseline(
                canonical_csv, anchor_csv, output_root, summary_path,
                "test-commit", builder_path, "2026-08-26T00:00:00+09:00"
            )
            self.assertEqual(summary["complete_count"], 4)
            self.assertTrue(summary["all_outputs_bit_exact"])
            with self.assertRaises(FileExistsError):
                runner.run_identity_baseline(
                    canonical_csv, anchor_csv, output_root, summary_path,
                    "test-commit", builder_path, "2026-08-26T00:00:00+09:00"
                )


if __name__ == "__main__":
    unittest.main()
