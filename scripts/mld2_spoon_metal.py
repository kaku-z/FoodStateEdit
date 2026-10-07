"""Analytic silver appearance for the utensil pass; source lighting is a hypothesis."""
import numpy as np


def reflective_metal_rgb(mesh, source_rgb):
    """Return uint8[N,3] spoon vertex RGB; camera origin is (0,0,0). No mesh mutation."""
    vertex = np.asarray(mesh.vertices, dtype=float)
    normal = np.asarray(mesh.vertex_normals, dtype=float)
    view = -vertex / np.maximum(np.linalg.norm(vertex, axis=1, keepdims=True), 1e-9)
    incidence = np.sum(normal*view, axis=1, keepdims=True)
    reflection = 2*incidence*normal-view
    source = (np.asarray(source_rgb, dtype=float).reshape(-1, 3)[::16]/255.)**2.2
    luminance = source.mean(1)
    weight = np.exp(-12*np.ptp(source, axis=1))*(.15+luminance)
    neutral = np.sum(source*weight[:, None], axis=0)/np.sum(weight)
    tint = np.clip(neutral/max(float(neutral.mean()), 1e-9), .85, 1.15)
    exposure = np.clip(np.quantile(luminance, .75), .32, .85)
    center = np.median(reflection[:, :2], axis=0)
    strip = np.exp(-((reflection[:, 1]-center[1]-.04)/.12)**2-((reflection[:, 0]-center[0])/.90)**8)
    fill = np.exp(-((reflection[:, 2]+.45)/.65)**2)
    rim = .22*(1-np.clip(np.abs(incidence[:, 0]), 0, 1))**8
    linear = np.clip((exposure*(.28+.90*strip+.18*fill)+rim)[:, None]*tint, 0, 1)
    srgb = np.where(linear<=.0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return np.clip(srgb*255, 0, 255).astype(np.uint8)
