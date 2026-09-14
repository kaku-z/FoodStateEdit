import hashlib
import json
import unittest
from pathlib import Path

from scripts.run_benchmark_vace_shard import condition_specs


class HeldoutVaceShardTest(unittest.TestCase):
    def test_scale_one_reuses_fixed_relative3d_generation(self):
        config = {"material_scale_policy": {"liquid": 1.0}}
        specs = condition_specs(config, "liquid")
        self.assertEqual(len(specs), 3)
        self.assertEqual([item["condition"] for item in specs], [
            "native_scale_1p0", "planar_scale_1p0", "relative3d_scale_1p0"
        ])

    def test_nonunit_material_scale_adds_one_unique_generation(self):
        config = {"material_scale_policy": {"strand": 0.8}}
        specs = condition_specs(config, "strand")
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[-1], {
            "condition": "relative3d_material_adaptive",
            "control_mode": "relative3d",
            "vace_scale": 0.8,
        })

    def test_day32_config_covers_each_pilot_case_once(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "configs/day32_material_policy_pilot_v1.json").read_text(encoding="utf-8"))
        dataset = json.loads((repo / "artifacts/day31_family_template_controls_pilot_v1/dataset_manifest.json").read_text(encoding="utf-8"))
        members = [case_id for shard in config["shards"].values() for case_id in shard]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(members), {case["case_id"] for case in dataset["cases"]})
        runner = repo / "scripts/run_benchmark_vace_shard.py"
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(), config["implementation"]["runner_sha256"])
        expected_jobs = sum(
            3 * len(condition_specs(config, case["family"])) for case in dataset["cases"]
        )
        self.assertEqual(expected_jobs, 210)


if __name__ == "__main__":
    unittest.main()
