"""Camera-aware distal utensil placement; the food-contact bowl stays fixed."""
import numpy as np


def connect_handle_to_frame(vertices, spoon, camera, image_width, margin=10.):
    vertices = np.asarray(vertices)
    axes = np.asarray(spoon['spoon_axes_camera'])
    a = float(spoon['radius_ab'][0])
    x = (vertices @ axes)[:, 0] - spoon['center_local'][0]
    fixed = x <= a
    tip = x >= x.max() - 1.e-5*a
    point = vertices[tip].mean(axis=0)
    projected = camera @ point
    old_uv = projected[:2]/projected[2]
    result = vertices.copy()
    displacement = 0.
    if old_uv[0] < image_width:
        direction = axes[:, 0]
        target_u = image_width + margin
        displacement = (target_u*(camera[2]@point)-camera[0]@point)/(camera[0]@direction-target_u*(camera[2]@direction))
        t = np.clip((x-a)/(x.max()-a), 0., 1.)
        result += displacement*(t*t*(3.-2.*t))[:, None]*direction
        result[fixed] = vertices[fixed]
    q = camera @ result[tip].mean(axis=0)
    return result, dict(old_tip_uv=old_uv.tolist(), new_tip_uv=(q[:2]/q[2]).tolist(),
        distal_axial_extension=float(displacement), fixed_bowl_vertices=int(fixed.sum()),
        fixed_bowl_max_change=float(np.abs(result[fixed]-vertices[fixed]).max()),
        camera_margin_pixels=float(margin), already_beyond_right_frame=bool(old_uv[0]>=image_width),
        rule='C1 monotone distal axial extension; fixed x<=bowl radius; can leave through top before reaching the projected right boundary.',
        scope='Frame connection and unchanged bowl geometry; no evidence of an actual off-screen hand.')
