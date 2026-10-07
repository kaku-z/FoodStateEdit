"""Synthetic state-error experiment with an immutable independent reference."""
import numpy as np
from PIL import Image
from .core import MaterialState


def make_fixture(seed, boundary_error=0, thickness_error=0):
    """All perturbations are source-only; the evaluation target is built separately.

    Thickness error moves the top along world z, NOT camera-ray depth error.
    Material IDs denote cells on the same fixed canonical lattice in every state.
    """
    rng = np.random.default_rng(seed)
    size = 56
    xyz = np.indices((size, size, size)).reshape(3, -1).T
    x, y, z = xyz.T
    width, length, height = (int(rng.integers(16, 23)), int(rng.integers(17, 25)),
                             int(rng.integers(8, 15)))
    x0, y0, z0 = 8, 12, 3
    family = seed % 3
    footprint = (x >= x0) & (x < x0 + width) & (y >= y0) & (y < y0 + length)
    if family == 1:
        footprint &= ((x + .5 - x0 - width / 2) / (width / 2)) ** 2 + \
                     ((y + .5 - y0 - length / 2) / (length / 2)) ** 2 <= 1
    top = z0 + height + (family == 2) * ((y - y0) // 5)
    occupied = footprint & (z >= z0) & (z < top + thickness_error)
    cut_x = x0 + width - int(rng.integers(5, 9))
    cut_y = y0 + length - int(rng.integers(8, 12))
    ids = np.flatnonzero(occupied)
    grid = np.full((size, size, size), -1, dtype=np.int32)
    grid.reshape(-1)[ids] = ids
    selected = np.flatnonzero(occupied & (x >= cut_x + boundary_error) & (y >= cut_y))
    if not len(selected):
        raise ValueError("Empty selected component")
    state = MaterialState(grid, xyz.astype(np.int16), selected)
    delta = np.array([13 + seed % 4, seed % 3, (seed // 3) % 3])
    meta = {"seed": seed, "family": ["cuboid", "elliptic", "stepped"][family],
            "width": width, "length": length, "height": height,
            "cut_x": cut_x, "cut_y": cut_y, "delta": delta.tolist(),
            "boundary_error": boundary_error, "thickness_error": thickness_error}
    return state, delta, meta


def edit_metrics(reference_source, reference_final, predicted_source, predicted_final):
    """All task correctness denominators refer to independent true state."""
    desired = reference_source.selected_ids
    selected = predicted_source.selected_ids
    intersection = len(np.intersect1d(desired, selected))
    original_positions = reference_source.canonical[desired]
    expected_positions = np.argwhere(np.isin(reference_final.grid, desired))
    expected_ids = reference_final.grid[tuple(expected_positions.T)]
    correctly_placed = np.sum(predicted_final.grid[tuple(expected_positions.T)] == expected_ids)
    a = reference_final.grid >= 0
    b = predicted_final.grid >= 0
    remaining_mask = (reference_source.grid >= 0) & ~np.isin(reference_source.grid, desired)
    predicted_count = np.sum(predicted_source.grid >= 0)
    return {
        "selection_precision": intersection / len(selected),
        "selection_recall": intersection / len(desired),
        "selection_iou": intersection / len(np.union1d(desired, selected)),
        "extra_transported_fraction": len(np.setdiff1d(selected, desired)) / len(desired),
        "true_source_residual_fraction": float(np.mean(np.isin(
            predicted_final.grid[tuple(original_positions.T)], desired))),
        "true_destination_error_fraction": float(1 - correctly_placed / len(desired)),
        "remaining_changed_fraction": float(np.mean(predicted_final.grid[remaining_mask] !=
                                                      reference_source.grid[remaining_mask])),
        "final_occupancy_iou": float(np.sum(a & b) / np.sum(a | b)),
        "self_volume_error": float(abs(np.sum(predicted_final.grid >= 0) - predicted_count) / predicted_count),
        "reference_volume_error": float(abs(np.sum(b) - np.sum(a)) / np.sum(a)),
    }


def translate_image(image, dx, dy):
    channels = [np.asarray(Image.fromarray(image[..., k].astype(np.float32)).transform(
        (image.shape[1], image.shape[0]), Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy), resample=Image.Resampling.BILINEAR))
                for k in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def image_plane_delta(delta, resolution):
    direction = np.array([.32, .42, -1.]); direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    return float(delta @ right * (resolution - 1) / 68), float(-delta @ up * (resolution - 1) / 68)


def planar_comparator(source_rgb, visible_selection, delta):
    """Source-only 2D cut/paste, bilinear shift, neighbor-propagation hole fill.

    No ground-truth final RGB/background is consumed. This weak diagnostic has
    less hidden appearance information than the oracle 3D renderer; it is NOT
    an information-matched competitive baseline.
    """
    filled = source_rgb.copy()
    known = ~visible_selection
    filled[~known] = 0
    for _ in range(sum(known.shape)):
        if known.all():
            break
        padded = np.pad(filled, ((1, 1), (1, 1), (0, 0)))
        valid = np.pad(known, 1)
        count = (valid[:-2, 1:-1].astype(int) + valid[2:, 1:-1] +
                 valid[1:-1, :-2] + valid[1:-1, 2:])
        total = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        frontier = ~known & (count > 0)
        if not frontier.any():
            raise ValueError("No observed pixels available for hole fill")
        filled[frontier] = total[frontier] / count[frontier, None]
        known[frontier] = True
    dx, dy = image_plane_delta(delta, len(source_rgb))
    alpha = translate_image(visible_selection[..., None].astype(float), dx, dy)
    payload = translate_image(source_rgb * visible_selection[..., None], dx, dy)
    return np.clip(filled * (1 - alpha) + payload, 0, 1)
