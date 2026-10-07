"""Data-contract tests for synthetic, source-only MLD pretraining."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mld_pretraining_data.py"
_SPEC = importlib.util.spec_from_file_location("mld_pretraining_data", _PATH)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


class MLDPretrainingDataTest(unittest.TestCase):
    def test_fixed_queries_and_projection_are_independent_of_scene(self):
        xyz = data.canonical_grid(4)
        self.assertEqual(xyz.shape, (64, 3))
        np.testing.assert_array_equal(np.unique(xyz[:, 0]), [-.75, -.25, .25, .75])
        uv = data.known_camera_projection(xyz)
        independent = xyz.astype(float) @ data.camera_basis()[:2].T / data.CAMERA_SCALE
        independent[:, 1] *= -1
        np.testing.assert_allclose(uv, independent, atol=4e-8, rtol=0)

    def test_replay_and_internal_material_are_nonconstant(self):
        first, metadata = data.generate_scene(0, seed=123, grid_size=8, image_size=16)
        replay, repeated = data.generate_scene(0, seed=123, grid_size=8, image_size=16)
        for key in first:
            np.testing.assert_array_equal(first[key], replay[key])
        self.assertEqual(metadata, repeated)
        self.assertEqual(first["joint"].shape, (512, 4))
        self.assertTrue(np.isfinite(first["joint"]).all())
        self.assertLessEqual(float(np.max(np.abs(first["joint"]))), 1)
        rgb = (first["joint"][:, 1:].astype(float) + 1) / 2
        inside = first["occupancy"]
        self.assertGreater(float(np.std(rgb[inside])), .025)
        self.assertNotEqual(metadata["scene_id"], data.scene_identity(1, 123))
        # Generator labels explicitly contain hidden layers/inclusions; their
        # coefficients never appear in INPUT_KEYS or source projection.
        parameters = metadata["target_generator_parameters"]
        self.assertGreaterEqual(len(parameters["inclusion_centers"]), 4)
        self.assertTrue(set(data.INPUT_KEYS).isdisjoint(data.TARGET_KEYS))

    def test_subcell_partition_conserves_each_source_cell(self):
        record, metadata = data.generate_scene(3, seed=123, grid_size=6, image_size=16)
        xyz = data.canonical_grid(6)
        parameters = metadata["target_generator_parameters"]
        offsets = np.stack(np.meshgrid(*[[-.5 / 6, .5 / 6]] * 3, indexing="ij"),
                           axis=-1).reshape(8, 3)
        samples = xyz[:, None, :] + offsets[None, :, :]
        occupied = data.shape_sdf(samples, parameters) <= 0
        count = occupied.sum(axis=1)
        np.testing.assert_array_equal(record["cell_occupancy"], count / 8)
        np.testing.assert_array_equal(record["action_validmask"], count > 0)
        for index, command in enumerate(record["actions"]):
            selected = (occupied & (samples @ command[:3] > command[3])).sum(axis=1)
            if index == 0:
                selected[:] = 0
            expected = np.divide(selected, count, out=np.zeros(len(count), dtype=float),
                                 where=count > 0)
            np.testing.assert_allclose(record["carried_fraction"][index], expected,
                                       atol=.00025, rtol=0)
            fraction = record["carried_fraction"][index].astype(float)
            masses = record["cell_occupancy"].astype(float)
            np.testing.assert_allclose(masses * fraction + masses * (1 - fraction),
                                       masses, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(record["actions"][0], np.zeros(10))
        np.testing.assert_array_equal(record["carried_fraction"][0], np.zeros(216))
        np.testing.assert_array_equal(data.action_delta(xyz, record["actions"][0]),
                                      np.zeros((216, 3)))

    def test_action_commands_do_not_read_generator_geometry(self):
        identity = data.scene_identity(4, 123)
        first = data.frozen_actions(identity)
        data._scene_parameters(identity)
        replay = data.frozen_actions(identity)
        np.testing.assert_array_equal(first, replay)
        xyz = np.asarray([[.2, -.1, .3]], dtype=np.float32)
        command = np.zeros(10, dtype=np.float32)
        command[4:7] = [.1, .2, .3]
        np.testing.assert_allclose(data.action_delta(xyz, command), [[.1, .2, .3]], atol=1e-7)

    def test_split_is_identity_grouped_and_exact_at_frozen_size(self):
        split = data.split_assignment(8192)
        np.testing.assert_array_equal(np.bincount(split), [6144, 1024, 1024])
        identities = [data.scene_identity(i) for i in range(8192)]
        sets = [set(identity for identity, code in zip(identities, split) if code == value)
                for value in range(3)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_memmap_contract_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = data.prepare_dataset(directory, scenes=8, seed=123,
                                             grid_size=4, image_size=16)
            self.assertEqual(manifest["split_counts"], {"train": 6, "validation": 1, "test": 1})
            self.assertEqual(manifest["inputs"], ["source_rgb", "canonical_xyz", "source_uv", "actions"])
            self.assertEqual(np.load(Path(directory) / "source_rgb.npy", mmap_mode="r").shape,
                             (8, 16, 16, 3))
            for record in manifest["files"]:
                self.assertEqual(data.file_sha256(Path(directory) / record["path"]), record["sha256"])
            metadata = [json.loads(line) for line in (Path(directory) / "scenes.jsonl").read_text().splitlines()]
            self.assertEqual(len(metadata), 8)
            self.assertEqual([row["split"] for row in metadata], ["train"] * 6 + ["validation", "test"])
            with self.assertRaises(FileExistsError):
                data.prepare_dataset(directory, scenes=8, seed=123, grid_size=4, image_size=16)


if __name__ == "__main__":
    unittest.main()
