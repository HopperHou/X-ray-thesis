"""Candidate-level continuous geometry targets for the frozen V6-A2 protocol.

Boxes and known GT use the same pixel-space xyxy coordinate system.  Status
is one of E (exhaustive), P (positive only), U (unknown).  No GT is inferred
from an empty P or U cell.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BUCKET_EDGES = (0.0, 0.05, 0.20, 0.40, 0.70, 1.000001)


def pair_geometry(boxes: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N,G] plain IoU and normalized squared center distance."""
    if boxes.ndim != 2 or gt.ndim != 2 or boxes.shape[1] != 4 or gt.shape[1] != 4:
        raise ValueError("boxes and gt must have shape [N,4] and [G,4]")
    if gt.shape[0] == 0:
        empty = boxes.new_empty((len(boxes), 0))
        return empty, empty
    left = torch.maximum(boxes[:, None, :2], gt[None, :, :2])
    right = torch.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    intersection = (right - left).clamp_min(0).prod(-1)
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)[:, None]
    area_g = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)[None, :]
    iou = intersection / (area_b + area_g - intersection).clamp_min(1e-12)
    cb = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    cg = (gt[:, :2] + gt[:, 2:]) * 0.5
    diagonal2 = (gt[:, 2:] - gt[:, :2]).square().sum(-1).clamp_min(1e-12)
    d2 = (cb[:, None] - cg[None, :]).square().sum(-1) / diagonal2[None, :]
    return iou.clamp(0, 1), d2


def quality_targets(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    statuses: list[str] | tuple[str, ...],
    assigned_gt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build [slot,class] target and boolean mask without P-cell false negatives.

    ``assigned_gt`` is [N] and indexes ``gt_boxes``; -1 means unassigned.
    It is used only to add native assigned positives to P-cell supervision.
    """
    boxes, gt_boxes = boxes.detach().float(), gt_boxes.detach().float()
    gt_classes = gt_classes.long()
    n, nc = len(boxes), len(statuses)
    if gt_boxes.shape != (len(gt_classes), 4) or (assigned_gt is not None and assigned_gt.shape != (n,)):
        raise ValueError("GT or assigned_gt shape mismatch")
    if any(s not in {"E", "P", "U"} for s in statuses):
        raise ValueError("Unknown annotation status")
    target = boxes.new_zeros((n, nc))
    mask = torch.zeros((n, nc), dtype=torch.bool, device=boxes.device)
    best_gt = torch.full((n, nc), -1, dtype=torch.long, device=boxes.device)
    max_iou = boxes.new_zeros((n, nc))
    iou, d2 = pair_geometry(boxes, gt_boxes)
    for c, status in enumerate(statuses):
        if status == "U":
            continue
        indices = torch.where(gt_classes == c)[0]
        if len(indices):
            qc = ((2.0 ** iou[:, indices] - 1.0) * torch.exp(-d2[:, indices])).clamp(0, 1)
            target[:, c], local = qc.max(dim=1)
            best_gt[:, c] = indices[local]
            max_iou[:, c] = iou[:, indices].max(dim=1).values
        if status == "E":
            mask[:, c] = True  # includes the legitimate empty-GT cell
        elif len(indices):
            mask[:, c] = max_iou[:, c] >= 0.5
            if assigned_gt is not None:
                assigned = (assigned_gt >= 0) & (assigned_gt < len(gt_classes))
                if assigned.any():
                    safe = assigned_gt.clamp(0, max(len(gt_classes) - 1, 0))
                    mask[:, c] |= assigned & (gt_classes[safe] == c)
    if not torch.isfinite(target).all() or not bool(((target >= 0) & (target <= 1)).all()):
        raise RuntimeError("Nonfinite or out-of-range Q target")
    return {"target": target, "mask": mask, "best_gt": best_gt,
            "max_iou": max_iou, "iou": iou, "d2": d2}


def sample_q_rows(
    targets: dict[str, torch.Tensor], z0: torch.Tensor, gt_classes: torch.Tensor,
    assigned_gt: torch.Tensor, seed: int,
) -> dict[str, torch.Tensor | dict[str, int]]:
    """Sample ordinary rows by Q bucket and rotate good rows per GT.

    Unselected rows receive weight zero and stay ignored.  Each GT's sampled
    positive rows sum to weight one; ordinary bucket rows sum to weight one.
    """
    q, eligible, iou = targets["target"], targets["mask"], targets["iou"]
    n, nc = q.shape
    if z0.shape != q.shape or assigned_gt.shape != (n,):
        raise ValueError("Score or assignment shape mismatch")
    weight = torch.zeros_like(q)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    counts: dict[str, int] = {}
    def first_unique_limit(parts: tuple[torch.Tensor, ...], limit: int) -> torch.Tensor:
        seen: set[int] = set()
        ordered: list[int] = []
        for part in parts:
            for index in part.tolist():
                if index not in seen:
                    seen.add(index)
                    ordered.append(index)
                if len(ordered) == limit:
                    return torch.tensor(ordered, device=q.device, dtype=torch.long)
        return torch.tensor(ordered, device=q.device, dtype=torch.long)
    for c in range(nc):
        for b, (lo, hi) in enumerate(zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])):
            # Good duplicate candidates are governed by their per-GT cap below.
            pool = torch.where(eligible[:, c] & (targets["max_iou"][:, c] < 0.5)
                               & (q[:, c] >= lo) & (q[:, c] < hi))[0]
            # B0 prioritizes hard high-score E-cell negatives.
            if b == 0:
                pool = pool[torch.argsort(z0[pool, c], descending=True, stable=True)]
            elif len(pool):
                pool = pool[torch.randperm(len(pool), generator=gen).to(pool.device)]
            take = pool[:32]
            if len(take):
                weight[take, c] = torch.maximum(weight[take, c], q.new_full((len(take),), 1 / len(take)))
            counts[f"class_{c}_bucket_{b}"] = len(take)
        for g in torch.where(gt_classes == c)[0].tolist():
            good = torch.where(eligible[:, c] & (iou[:, g] >= 0.5))[0]
            assigned = torch.where((assigned_gt == g) & eligible[:, c])[0]
            if len(good):
                high_q = good[torch.argsort(q[good, c], descending=True, stable=True)]
                high_score = good[torch.argsort(z0[good, c], descending=True, stable=True)]
                rotation = seed % len(good)
                rotating = torch.roll(good, shifts=rotation)
                good = first_unique_limit((high_q[:11], high_score[:11], rotating), 32)
            chosen = torch.unique(torch.cat((good, assigned)))
            if len(chosen):
                weight[chosen, c] += 1 / len(chosen)
            counts[f"gt_{g}_class_{c}"] = len(chosen)
    return {"weight": weight, "counts": counts}


def continuous_q_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != weight.shape:
        raise ValueError("Q BCE shapes differ")
    if not bool((weight >= 0).all()):
        raise ValueError("Negative Q weight")
    return (F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none") * weight).sum() / weight.sum().clamp_min(1)
