import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


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
        self.assertEqual(registry["seed_policy"]["stochastic_formal_test"], [1, 2, 3])
        self.assertEqual(registry["day3_gate"]["four_anchor_batch_complete_count"], 3)
        self.assertEqual(registry["day3_gate"]["status"], "CLOSED_COMPLETE")

    def test_common_proxy_evidence_has_four_exact_protection_cases(self):
        evidence_root = ROOT / "results" / "day4_common_proxy_v1"
        summary = json.loads((evidence_root / "common_proxy_summary_v1.json").read_text())
        self.assertEqual(summary["proxy_version"], "common_proxy_v1")
        self.assertEqual(summary["case_count"], 4)
        self.assertTrue(summary["all_protected_pixels_exact"])
        expected = {
            "soup_spoon_001", "fried_rice_spatula_001",
            "ramen_chopsticks_001", "pasta_fork_001",
        }
        self.assertEqual({item["anchor_id"] for item in summary["cases"]}, expected)
        for item in summary["cases"]:
            self.assertEqual(item["outside_edit_alpha_max_pixel_difference"], 0)
            self.assertEqual(max(item["working_width"], item["working_height"]), 736)
            self.assertEqual(item["working_width"] % 16, 0)
            self.assertEqual(item["working_height"] % 16, 0)
            manifest_path = evidence_root / item["anchor_id"] / "proxy_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest, item)
            self.assertEqual(len(manifest["canonical_input_sha256"]), 64)
            self.assertEqual(len(manifest["files"]), 13)
            for file_item in manifest["files"].values():
                self.assertRegex(file_item["sha256"], r"^[0-9a-f]{64}$")

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

    def test_remote_no_edit_evidence_has_four_valid_run_manifests(self):
        evidence_root = ROOT / "results" / "day3_no_edit_v1"
        summary = json.loads((evidence_root / "no_edit_summary_v1.json").read_text())
        schema = json.loads((ROOT / "schemas" / "run_manifest.schema.json").read_text())
        validator = Draft202012Validator(schema)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["complete_count"], 4)
        self.assertTrue(summary["all_outputs_bit_exact"])
        self.assertEqual(len({row["anchor_id"] for row in summary["runs"]}), 4)
        for row in summary["runs"]:
            self.assertEqual(row["source_sha256"], row["output_sha256"])
            manifest = json.loads((evidence_root / "runs" / f"{row['anchor_id']}.json").read_text())
            validator.validate(manifest)
            self.assertEqual(manifest["method"], "input_no_edit")
            self.assertEqual(manifest["seed"], 0)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["outputs"][0]["sha256"], row["output_sha256"])

    def test_geoedit_smoke_evidence_has_eight_valid_run_manifests(self):
        schema = json.loads((ROOT / "schemas" / "run_manifest.schema.json").read_text())
        validator = Draft202012Validator(schema)
        expected_model_hashes = {
            "66c61b736c5674deeeef17861e494d3652cc9b1463a9656bf18c2c72d2c5f007",
            "0bf791adfb8330d451d2f5c03577b2a8fb780453f8ef05e23a8fa91f27a2134d",
            "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
            "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
        }
        batches = {
            "day3_vanilla_geoedit_smoke_v1": "vanilla_geoedit",
            "day3_unified_geoedit_smoke_v1": "geoedit_unified_action_mask",
        }
        for directory, method in batches.items():
            root = ROOT / "results" / directory
            summary = json.loads((root / "batch_summary_v1.json").read_text())
            self.assertEqual(summary["method"], method)
            self.assertEqual(summary["seed"], 1)
            self.assertEqual(summary["case_count"], 4)
            self.assertEqual(summary["complete_count"], 4)
            self.assertEqual(summary["technical_failure_count"], 0)
            self.assertTrue(summary["all_protected_pixels_exact"])
            for row in summary["runs"]:
                self.assertEqual(row["status"], "complete")
                self.assertEqual(row["outside_edit_alpha_max_pixel_difference"], 0)
                manifest = json.loads((root / "runs" / f"{row['anchor_id']}.json").read_text())
                validator.validate(manifest)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["seed"], 1)
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["inference"]["decoded_frames"], 21)
                self.assertEqual(
                    manifest["inference"]["outside_edit_alpha_max_pixel_difference"], 0
                )
                self.assertEqual(len(manifest["outputs"]), 3)
                self.assertEqual(
                    {item["sha256"] for item in manifest["model"]["files"]},
                    expected_model_hashes,
                )

    def test_model_hash_audit_and_internal_reviews_are_complete(self):
        audit = ROOT / "results" / "day3_model_hash_v1" / "wan_model_sha256.txt"
        lines = [line for line in audit.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(len(line.split()[0]) == 64 for line in lines))
        launcher_hashes = {
            "run_geoedit_anchor_v1.py": "9373002caf9924e35a8f9c1eda2e343972f5ded74dc0e152ad6a538b98933ea6",
            "run_geoedit_anchor_v2.py": "e839949a8bb2d1d6df30d352f210ed03c2192c27274a837a7ed5d0e8596b623c",
            "summarize_geoedit_batch_v1.py": "b28d13508a8b8ad4ac8b7bf2d710690b98c62feea67eff226f63bdfffaa4d4dd",
        }
        for name, expected_hash in launcher_hashes.items():
            payload = (ROOT / "results" / "day3_launchers" / name).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)
        expected = {
            "day3_vanilla_geoedit_smoke_v1": 1,
            "day3_unified_geoedit_smoke_v1": 0,
        }
        for directory, successes in expected.items():
            review = json.loads(
                (ROOT / "results" / directory / "internal_pilot_review_v1.json").read_text()
            )
            self.assertEqual(review["aggregate"]["technical_complete"], 4)
            self.assertEqual(review["aggregate"]["provisional_action_success"], successes)
            self.assertEqual(review["aggregate"]["exact_preservation_success"], 4)
            self.assertEqual(len(review["cases"]), 4)


if __name__ == "__main__":
    unittest.main()
