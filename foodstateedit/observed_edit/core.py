"""Source-only state scaffold and shared-observation compositor.

An integer image-plane translation is intentional in the first experiment.
It tests whether observed-source anchoring helps before introducing estimated
3D geometry. Pixel provenance is an internal invariant, not physical mass.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class EditState:
    source_mask: np.ndarray
    target_mask: np.ndarray
    source_ids: np.ndarray
    transported_rgb: np.ndarray
    editable: np.ndarray
    payload_core: np.ndarray
    control: np.ndarray
    scaffold: np.ndarray
    delta_xy: tuple


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))).astype(bool)


def build_state(source, mask, target_xy, handle_xy):
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Expected uint8 RGB image")
    mask = np.asarray(mask, bool)
    if mask.shape != source.shape[:2] or not mask.any():
        raise ValueError("Nonempty aligned source mask required")
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    dx = int(round(target_xy[0] - xx.mean()))
    dy = int(round(target_xy[1] - yy.mean()))
    tx, ty = xx + dx, yy + dy
    if tx.min() < 0 or tx.max() >= w or ty.min() < 0 or ty.max() >= h:
        raise ValueError("Transport is clipped by image boundary")
    target = np.zeros_like(mask)
    target[ty, tx] = True
    if np.any(target & mask):
        raise ValueError("This first experiment requires disjoint source/target footprints")
    ids = np.full((h, w), -1, np.int32)
    ids[ty, tx] = yy*w + xx
    payload = np.zeros_like(source)
    payload[ty, tx] = source[yy, xx]
    tool = np.zeros_like(mask, np.uint8)
    center = (int(round(tx.mean())), int(round(ty.mean())))
    cv2.line(tool, center, tuple(handle_xy), 255, 19, cv2.LINE_AA)
    # The ellipse is explicitly a schematic endpoint guide, not a tool mesh.
    axes = (int((tx.max()-tx.min())*.56)+8, int((ty.max()-ty.min())*.56)+8)
    cv2.ellipse(tool, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    tool_mask = tool > 0
    editable = dilate(mask, 20) | dilate(target, 26) | dilate(tool_mask, 20)
    core = cv2.erode(target.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if not core.any():
        raise ValueError("Payload too small for observation core")
    scaffold = cv2.inpaint(cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
                          dilate(mask, 8).astype(np.uint8)*255, 7, cv2.INPAINT_TELEA)
    scaffold = cv2.cvtColor(scaffold, cv2.COLOR_BGR2RGB)
    scaffold[tool_mask] = (168, 177, 186)
    scaffold[target] = payload[target]
    control = source.copy()
    cv2.drawContours(control, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                     cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 190, 0), 3)
    cv2.drawContours(control, cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL,
                     cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 220, 255), 3)
    cv2.line(control, center, tuple(handle_xy), (0, 220, 255), 4, cv2.LINE_AA)
    values = [mask.copy(), target, ids, payload, editable, core, control, scaffold]
    for value in values:
        value.setflags(write=False)
    return EditState(*values, (dx, dy))


def compose(source, proposal, state, anchor_observations):
    if proposal.shape != source.shape:
        raise ValueError("Proposal must be explicitly resized before composition")
    # No source alpha blend in the removed footprint; only the generated proposal.
    out = source.copy()
    out[state.editable] = proposal[state.editable]
    if anchor_observations:
        out[state.payload_core] = state.transported_rgb[state.payload_core]
    return out
