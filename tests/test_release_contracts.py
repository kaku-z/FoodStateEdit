import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def validate_example(self, stem: str) -> None:
        schema = json.loads(
            (ROOT / "schemas" / f"{stem}.schema.json").read_text(encoding="utf-8")
        )
        example = json.loads(
            (ROOT / "examples" / f"{stem}.example.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(example)

    def test_case_manifest_contract(self) -> None:
        self.validate_example("case_manifest")

    def test_run_manifest_contract(self) -> None:
        self.validate_example("run_manifest")

    def test_metrics_contract(self) -> None:
        self.validate_example("metrics")

    def test_scope_config_is_valid_json(self) -> None:
        scope = json.loads(
            (ROOT / "configs" / "paper_scope_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(scope["target_case_count"], 60)
        self.assertEqual(scope["formal_seeds"], [1, 2, 3])
        self.assertEqual(len(scope["families"]), 4)


if __name__ == "__main__":
    unittest.main()
