"""Shared bite-volume geometry and conservative local compositing for E5."""
import cv2
import numpy as np


def polygon_mask(points, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask > 0


def dilate(mask, radius):
    if radius < 0:
        raise ValueError("negative dilation")
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def inward_alpha(support, feather):
    if feather <= 0:
        raise ValueError("feather must be positive")
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.rint(255 * np.clip(distance / feather, 0, 1)).astype(np.uint8)
    alpha[~support] = 0
    return alpha


def visible_repair_support(removal_projection, cleanup_domain, payload_projection,
                           lift, *, base_radius, growth_radius, exclusion_radius):
    """Reveal one shared removal volume while never editing the moving payload."""
    if not 0 <= lift <= 1:
        raise ValueError("lift must be in [0, 1]")
    if lift <= 0:
        return np.zeros_like(removal_projection, dtype=bool)
    radius = base_radius + int(round(growth_radius * lift))
    support = dilate(removal_projection, radius) & cleanup_domain
    return support & ~dilate(payload_projection, exclusion_radius)


def fit_plate_field(image, donor_mask):
    """Fit a smooth affine RGB plate field from real neutral plate pixels."""
    ys, xs = np.nonzero(donor_mask)
    if len(xs) < 200:
        raise ValueError("not enough plate donor pixels")
    design = np.column_stack([np.ones(len(xs)), xs, ys])
    coefficients = np.linalg.lstsq(design, image[ys, xs].astype(float), rcond=None)[0]
    yy, xx = np.indices(image.shape[:2])
    field = np.stack([np.ones_like(xx), xx, yy], axis=-1) @ coefficients
    return np.clip(np.rint(field), 0, 255).astype(np.uint8)


def warp_patch_to_quad(patch, quad, shape):
    h, w = patch.shape[:2]
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    target = np.float32(quad)
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(patch, transform, (shape[1], shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def build_cavity_anchor(reference, vertices, crumb_patch, plate_donor_mask):
    """Render hidden cut walls and plate floor from one removal cuboid."""
    shape = reference.shape[:2]
    wall_back = polygon_mask(vertices[[0, 1, 5, 4]], shape)
    wall_left = polygon_mask(vertices[[3, 0, 4, 7]], shape)
    floor = polygon_mask(vertices[[4, 5, 6, 7]], shape)
    anchor = reference.copy()
    plate = fit_plate_field(reference, plate_donor_mask)
    anchor[floor] = plate[floor]
    donor_a = crumb_patch
    donor_b = np.ascontiguousarray(crumb_patch[:, ::-1])
    warped_a = warp_patch_to_quad(donor_a, vertices[[0, 1, 5, 4]], shape)
    warped_b = warp_patch_to_quad(donor_b, vertices[[3, 0, 4, 7]], shape)
    anchor[wall_back] = warped_a[wall_back]
    anchor[wall_left] = warped_b[wall_left]

    # Thin frosting rim and restrained contact shading are geometric seeds only;
    # the diffusion pass remains responsible for photographic completion.
    hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
    yy = np.indices(shape)[0]
    frosting_candidates = (yy < shape[0] * .55) & (hsv[..., 1] < 65) & (hsv[..., 2] > 150)
    pixels = reference[frosting_candidates]
    frosting = np.mean(pixels if len(pixels) else reference.reshape(-1, 3), axis=0)
    for a, b in [(0, 1), (3, 0)]:
        cv2.line(anchor, tuple(np.rint(vertices[a]).astype(int)),
                 tuple(np.rint(vertices[b]).astype(int)), tuple(map(int, frosting)), 4, cv2.LINE_AA)
    wall_union = wall_back | wall_left
    near_wall = cv2.dilate(wall_union.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    shadow = floor & near_wall
    anchor[shadow] = np.rint(anchor[shadow].astype(float) * 0.90).astype(np.uint8)
    return anchor, dict(wall_back=wall_back, wall_left=wall_left, floor=floor)


def composite(base, candidate, alpha):
    weight = alpha[..., None].astype(np.float32) / 255.0
    return np.rint(candidate * weight + base * (1 - weight)).clip(0, 255).astype(np.uint8)
