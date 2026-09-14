import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_benchmark_vace_collection import verify_shard


class BenchmarkVaceCollectionTest(unittest.TestCase):
    def test_complete_synthetic_shard_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_root = root / "case_a/seed_1/native_scale_1p0"
            job_root.mkdir(parents=True)
            output = job_root / "projected_final_hold.png"
            output.write_bytes(b"result")
            record = {
                "case_id": "case_a", "seed": 1, "condition": "native_scale_1p0",
                "frames": 21, "steps": 20,
                "outside_support_max_pixel_difference": 0,
                "cuda_max_memory_allocated_mib": 123.0,
                "cuda_max_memory_reserved_mib": 456.0,
                "files": {"projected_final_hold.png": {
                    "path": output.relative_to(root).as_posix(),
                    "size_bytes": output.stat().st_size,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }},
            }
            manifest = {
                "status": "complete_requires_blind_review",
                "pipeline_load_count": 1,
                "shard_id": "s0", "host": "test", "physical_gpu": 0,
                "config_sha256": "0" * 64, "case_ids": ["case_a"],
                "expected_jobs": [{"case_id": "case_a", "seed": 1, "condition": "native_scale_1p0"}],
                "completed_jobs": [record], "wall_time_seconds": 1.0,
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "COMPLETE").write_text("done\n", encoding="utf-8")
            result = verify_shard(root)
            self.assertEqual(result["completed_job_count"], 1)
            self.assertTrue(result["all_outside_support_exact"])


if __name__ == "__main__":
    unittest.main()
