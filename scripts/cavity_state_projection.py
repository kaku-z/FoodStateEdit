"""E7 material-pure cavity rendering and decaying, feathered latent projection.

The renderer operates on explicit source/material masks; it does not infer true
hidden geometry. The projector is a training-free intervention, not a learned
condition encoder. Historical E5/E6 renderers remain unchanged.
"""
import cv2
import numpy as np

from bite_remain_consistency import fit_plate_field, polygon_mask, warp_patch_to_quad


def select_material_patch(image, material_mask, minimum_side=12):
    """Largest all-material axis-aligned square (dynamic programming, no padding).

    A rectangular crop spanning multiple materials must never be stretched onto
    a cut wall. Fails closed if a sufficiently large pure donor does not exist.
    """
    mask = np.asarray(material_mask, bool)
    if mask.shape != image.shape[:2]:
        raise ValueError('material mask/image shape mismatch')
    dp = np.zeros((mask.shape[0]+1, mask.shape[1]+1), np.int32)
    best = (0, 0, 0)
    for y, x in zip(*np.nonzero(mask)):
        side = 1 + min(dp[y, x], dp[y, x+1], dp[y+1, x])
        dp[y+1, x+1] = side
        if side > best[0]:
            best = (int(side), int(y+1), int(x+1))
    side, y1, x1 = best
    if side < minimum_side:
        raise ValueError('no sufficiently large pure material donor')
    box = [x1-side, y1-side, x1, y1]
    assert mask[box[1]:box[3], box[0]:box[2]].all()
    return image[box[1]:box[3], box[0]:box[2]].copy(), box


def render_material_pure_cavity(reference, vertices, material_mask, plate_mask):
    patch, box = select_material_patch(reference, material_mask)
    shape = reference.shape[:2]
    faces = {'wall_back': [0, 1, 5, 4], 'wall_left': [3, 0, 4, 7],
             'floor': [4, 5, 6, 7]}
    masks = {name: polygon_mask(vertices[ids], shape) for name, ids in faces.items()}
    result = reference.copy()
    plate = fit_plate_field(reference, plate_mask)
    result[masks['floor']] = plate[masks['floor']]
    for name in ('wall_back', 'wall_left'):
        quad = vertices[faces[name]]
        # Preserve approximately the observed crumb size. Stretching a small
        # pure donor to a tall face enlarges pores and creates a second defect.
        width = max(2, int(round(max(np.linalg.norm(quad[1]-quad[0]),
                                    np.linalg.norm(quad[2]-quad[3])))))
        height = max(2, int(round(max(np.linalg.norm(quad[3]-quad[0]),
                                     np.linalg.norm(quad[2]-quad[1])))))
        donor = cv2.copyMakeBorder(patch, 0, max(0, height-patch.shape[0]),
                                  0, max(0, width-patch.shape[1]), cv2.BORDER_REFLECT_101)
        texture = warp_patch_to_quad(donor[:height, :width], quad, shape)
        result[masks[name]] = texture[masks[name]]
    # Shade only visible floor, with continuous falloff from actual wall edges.
    visible_floor = masks['floor'] & ~masks['wall_back'] & ~masks['wall_left']
    edge = np.zeros(shape, np.uint8)
    for a, b in ((4, 5), (4, 7)):
        cv2.line(edge, tuple(np.rint(vertices[a]).astype(int)),
                 tuple(np.rint(vertices[b]).astype(int)), 1, 1)
    distance = cv2.distanceTransform(1-edge, cv2.DIST_L2, 5)
    shade = 1 - .06*np.exp(-distance/5.0)
    result[visible_floor] = np.rint(result[visible_floor]*shade[visible_floor, None]).astype(np.uint8)
    return result, masks, {'donor_box_xyxy': box,
                          'donor_pixels': int(patch.shape[0]*patch.shape[1]),
                          'donor_material_purity': 1.0}


def soft_cavity_projection(latents, noisy_reference, mask, step, end_step,
                           max_strength=.35, feather_kernel=3, trace=None):
    """Correct a bounded fraction of the reference residual, then release it.

    z' = z + lambda_k * m_inward * (q(z_ref,t_next) - z)
    lambda_k = max_strength * (1-k/end_step)^2 for k < end_step.
    Pool only the mask spatially; never smooth texture or mix time steps.
    Outside the binary support the update is exactly zero. Pixel-space payload
    preservation still requires the final compositor because VAE is nonlocal.
    """
    import torch
    import torch.nn.functional as F
    if not 0 <= max_strength <= 1 or end_step <= 0:
        raise ValueError('invalid projection strength or endpoint')
    if feather_kernel < 1 or feather_kernel % 2 != 1:
        raise ValueError('feather_kernel must be positive and odd')
    if latents.shape != noisy_reference.shape or mask.ndim != 5:
        raise ValueError('latent/reference or mask shape mismatch')
    if mask.shape[2:] != latents.shape[2:]:
        raise ValueError('projection mask must already be temporally aligned')
    if not torch.isfinite(mask).all() or torch.any((mask < 0) | (mask > 1)):
        raise ValueError('mask weights must be finite and in [0,1]')
    strength = max_strength * max(0.0, 1.0-step/end_step)**2 if step >= 0 else 0.0
    if strength == 0:
        return latents
    # float32 accumulation avoids quantizing small BF16 guidance updates away.
    m = mask.float()
    smooth = F.avg_pool3d(m, (1, feather_kernel, feather_kernel), stride=1,
                         padding=(0, feather_kernel//2, feather_kernel//2))
    weight = strength * smooth * m
    result = (latents.float() + weight * (noisy_reference.float()-latents.float())).to(latents.dtype)
    if trace is not None:
        trace.append({'step': int(step), 'strength': float(strength),
                      'mask_nonzero': int((weight > 0).sum().item()),
                      'max_weight': float(weight.max().item())})
    return result
