import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bite_remain_consistency import (build_cavity_anchor, composite,
                                     polygon_mask, visible_repair_support)


def test_shared_volume_excludes_moving_payload():
    shape = (80, 100)
    source = polygon_mask(np.array([[30, 25], [50, 25], [50, 55], [30, 55]]), shape)
    payload = polygon_mask(np.array([[48, 8], [68, 8], [68, 38], [48, 38]]), shape)
    support = visible_repair_support(source, np.ones(shape, bool), payload, .8,
                                     base_radius=2, growth_radius=12, exclusion_radius=3)
    assert support.any()
    assert not (support & payload).any()
    assert not visible_repair_support(source, np.ones(shape, bool), payload, 0,
                                      base_radius=2, growth_radius=12, exclusion_radius=3).any()


def test_cavity_faces_are_disjoint_semantic_surfaces():
    image = np.full((100, 120, 3), 180, np.uint8)
    vertices = np.array([[40, 25], [70, 30], [68, 55], [38, 50],
                         [40, 55], [70, 60], [68, 82], [38, 77]], float)
    donor = np.full((20, 30, 3), [190, 145, 80], np.uint8)
    plate_donor = np.zeros((100, 120), bool); plate_donor[70:95, 75:115] = True
    anchor, surfaces = build_cavity_anchor(image, vertices, donor, plate_donor)
    assert all(mask.any() for mask in surfaces.values())
    assert np.mean(anchor[surfaces['floor']]) > 0
    assert not np.array_equal(anchor[surfaces['wall_back']], image[surfaces['wall_back']])


def test_composite_is_exact_outside_alpha():
    base = np.full((10, 12, 3), 20, np.uint8)
    candidate = np.full_like(base, 240)
    alpha = np.zeros((10, 12), np.uint8); alpha[3:7, 4:9] = 255
    result = composite(base, candidate, alpha)
    assert np.array_equal(result[alpha == 0], base[alpha == 0])
    assert np.array_equal(result[alpha == 255], candidate[alpha == 255])
