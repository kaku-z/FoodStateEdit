"""Numerical operations that preserve the domain of observed source geometry."""
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_observed_points(points, valid, sigma=.7):
    """Normalize smoothing by finite support without filling missing observations."""
    points=np.asarray(points)
    observed=np.asarray(valid,dtype=bool)&np.isfinite(points).all(axis=-1)
    weight=gaussian_filter(observed.astype(points.dtype),sigma)
    smoothed=np.full_like(points,np.nan)
    supported=observed&(weight>np.finfo(points.dtype).eps)
    for k in range(points.shape[-1]):
        value=gaussian_filter(np.where(observed,points[...,k],0),sigma)
        np.divide(value,weight,out=smoothed[...,k],where=supported)
    return smoothed
