import csv
import json
import math
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"liquid", "granular", "strand", "strand_contact"}


def load_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def png_dimensions(path: Path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise AssertionError(f"Not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def polyline_length(points, width, height):
    pixels = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in points]
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(pixels, pixels[1:]))


class Day2InputAndAnchorTests(unittest.TestCase):
    def test_canonical_inputs_are_traceable_aligned_and_structure_checked(self):
        frozen = load_csv(ROOT / "benchmark" / "data_manifest_v1.csv")
        canonical = load_csv(ROOT / "benchmark" / "canonical_input_manifest_v1.csv")
        self.assertEqual(len(canonical), 60)
        self.assertEqual({row["case_id"] for row in canonical}, {row["case_id"] for row in frozen})
        self.assertEqual(len({row["canonical_input_sha256"] for row in canonical}), 60)
        for row in canonical:
            source_width = int(row["source_width"])
            source_height = int(row["source_height"])
            self.assertEqual(int(row["canonical_width"]), source_width * 4 - (source_width * 4) % 8)
            self.assertEqual(int(row["canonical_height"]), source_height * 4 - (source_height * 4) % 8)
            self.assertLessEqual(int(row["dhash_distance"]), 8)
            self.assertEqual(row["scale_x"], f"{int(row['canonical_width']) / source_width:.1f}")
            self.assertEqual(row["scale_y"], f"{int(row['canonical_height']) / source_height:.1f}")

        summary = json.loads((ROOT / "benchmark" / "canonical_input_summary_v1.json").read_text())
        self.assertTrue(summary["all_source_hashes_match_frozen_manifest"])
        self.assertTrue(summary["all_canonical_hashes_unique"])
        self.assertTrue(summary["all_dhash_distances_within_limit"])
        self.assertEqual(summary["maximum_allowed_dhash_distance"], 8)

    def test_four_anchor_specs_are_pilot_only_and_complete(self):
        canonical = {row["case_id"]: row for row in load_csv(ROOT / "benchmark" / "canonical_input_manifest_v1.csv")}
        specs = json.loads((ROOT / "benchmark" / "anchor_specs_v1.json").read_text())
        self.assertEqual(specs["five_layer_order"], ["rigid", "contact", "material", "hole", "protect"])
        self.assertEqual(len(specs["anchors"]), 4)
        self.assertEqual({spec["family"] for spec in specs["anchors"]}, FAMILIES)
        self.assertEqual(len({spec["case_id"] for spec in specs["anchors"]}), 4)
        for spec in specs["anchors"]:
            self.assertEqual(canonical[spec["case_id"]]["split"], "pilot")
            self.assertEqual(canonical[spec["case_id"]]["family"], spec["family"])
            self.assertEqual(set(spec["layers"]), {"rigid", "contact", "material", "hole", "protect"})
            for layer in ("rigid", "contact", "material", "hole"):
                self.assertGreater(len(spec["layers"][layer]["primitives"]), 0)
            self.assertGreaterEqual(len(spec["hard_constraints"]), 5)

        noodle = next(spec for spec in specs["anchors"] if spec["family"] == "strand")
        row = canonical[noodle["case_id"]]
        width, height = int(row["canonical_width"]), int(row["canonical_height"])
        target = noodle["layers"]["material"]["primitives"][0]["points"]
        source = noodle["layers"]["hole"]["primitives"][0]["points"]
        ratio = polyline_length(target, width, height) / polyline_length(source, width, height)
        self.assertGreaterEqual(ratio, 0.95)
        self.assertLessEqual(ratio, 1.05)

    def test_dense_anchor_layers_and_case_manifests_match_canonical_inputs(self):
        canonical = {row["case_id"]: row for row in load_csv(ROOT / "benchmark" / "canonical_input_manifest_v1.csv")}
        anchor_rows = load_csv(ROOT / "benchmark" / "anchor_manifest_v1.csv")
        summary = json.loads((ROOT / "benchmark" / "anchor_annotation_summary_v1.json").read_text())
        summary_rows = {row["anchor_id"]: row for row in summary["anchors"]}
        self.assertEqual(len(anchor_rows), 4)
        for anchor in anchor_rows:
            anchor_dir = ROOT / "benchmark" / "anchors_v1" / anchor["anchor_id"]
            manifest = json.loads((anchor_dir / "case_manifest.json").read_text())
            canonical_row = canonical[anchor["case_id"]]
            expected_size = (int(canonical_row["canonical_width"]), int(canonical_row["canonical_height"]))
            self.assertEqual(manifest["source"]["sha256"], canonical_row["canonical_input_sha256"])
            self.assertFalse(manifest["source"]["has_target_utensil"])
            self.assertEqual(manifest["split"], "pilot")
            self.assertIn("exact_protection", {item["id"] for item in manifest["constraints"]})
            for filename in (
                "mask_rigid.png",
                "mask_contact.png",
                "mask_material.png",
                "mask_hole.png",
                "mask_protect.png",
                "edit_alpha.png",
            ):
                path = anchor_dir / filename
                self.assertTrue(path.is_file())
                self.assertEqual(png_dimensions(path), expected_size)

            audit = summary_rows[anchor["anchor_id"]]
            self.assertEqual(audit["status"], "five_layer_annotation_complete")
            self.assertGreater(audit["contact_rigid_intersection_pixels"], 0)
            self.assertGreater(audit["contact_material_intersection_pixels"], 0)
            self.assertAlmostEqual(audit["edit_alpha_nonzero_fraction"] + audit["protect_fraction"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
