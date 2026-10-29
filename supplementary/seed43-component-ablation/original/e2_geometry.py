"""Isolated E2 scalar functions; no changes to production loss or assignment.

Caller supplies native foregrounds, applies lambda and actual batch once.
This module alone is not a production training integration.
"""
import math
import torch


def centre_components(pred, target, stride):
    if pred.ndim != 2 or pred.shape[1] != 4 or pred.shape != target.shape:
        raise ValueError("Expected aligned xyxy [M,4]")
    if stride.shape != (len(pred),):
        raise ValueError("Expected native stride [M]")
    p = pred.float()
    g = target.detach().float()
    s = stride.detach().float()
    wh = g[:, 2:] - g[:, :2]
    if not torch.isfinite(p).all() or not torch.isfinite(g).all():
        raise ValueError("Nonfinite boxes")
    if not torch.isfinite(s).all() or not (s > 0).all() or not (wh > 0).all():
        raise ValueError("Positive finite GT dimensions and strides required")
    denominator = torch.maximum(wh, s[:, None])
    displacement = ((p[:, :2] + p[:, 2:]) - (g[:, :2] + g[:, 2:])) * 0.5
    axis = 0.5 * (displacement.abs() / denominator).sum(-1)
    isotropic = displacement.abs().sum(-1) / (
        2 * torch.sqrt(denominator[:, 0] * denominator[:, 1])
    )
    return axis, isotropic


def foreground_geometry(arm, iou, graph_boxes, axis=None, isotropic=None, scale=None):
    if arm not in {"O_ONLY", "AXIS", "O_ISOTROPIC"}:
        raise ValueError("Unknown geometric arm")
    if iou.ndim != 1 or not torch.isfinite(iou).all():
        raise ValueError("Expected finite aligned IoU")
    if len(iou) == 0:
        return graph_boxes.sum() * 0
    overlap = 2.0 - 2.0 ** iou
    if arm == "O_ONLY":
        return overlap.mean()
    component = axis if arm == "AXIS" else isotropic
    if component is None or component.shape != iou.shape:
        raise ValueError("Centre component shape mismatch")
    if arm == "O_ISOTROPIC":
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise ValueError("A frozen positive calibration scale is required")
        component = component * scale
    return (overlap + component).mean()
