"""Native O2O box loss plus detached-target Q and optional assigned-GT geometry.

The native 8.4.83 O2O method computes the box and DFL/L1 components.  A
second call to its own assigner exposes target_scores for auditing and for
Q masks; a strict equality check guards against assignment drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.cfg import DEFAULT_CFG

from v6a2_targets import quality_targets, sample_q_rows, continuous_q_bce

_PILOT_PROTOCOL = json.loads((Path(__file__).resolve().parent / "runtime_protocol.json").read_text())
POPULATION = Path(_PILOT_PROTOCOL["pilot"]["population_json"])
ELIGIBILITY = Path(_PILOT_PROTOCOL["eligibility_matrix"])


def aligned_geometry(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected aligned xyxy [M,4]")
    inter = (torch.minimum(pred[:, 2:], gt[:, 2:]) - torch.maximum(pred[:, :2], gt[:, :2])).clamp_min(0).prod(-1)
    area_p = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(-1)
    area_g = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
    iou = inter / (area_p + area_g - inter).clamp_min(1e-12)
    d2 = (((pred[:, :2] + pred[:, 2:]) - (gt[:, :2] + gt[:, 2:])) * 0.5).square().sum(-1)
    d2 = d2 / (gt[:, 2:] - gt[:, :2]).square().sum(-1).clamp_min(1e-12)
    return iou.clamp(0, 1), d2


def axis_center_term(pred: torch.Tensor, gt: torch.Tensor,
                     slot_stride: torch.Tensor) -> torch.Tensor:
    """Frozen axis center component in image pixels for aligned O2O positives."""
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected aligned positive xyxy [M,4]")
    if slot_stride.shape != (pred.shape[0],):
        raise ValueError("Expected one native slot stride per positive")
    pred32 = pred.float()
    gt32 = gt.detach().float()
    stride32 = slot_stride.detach().float()
    wh = gt32[:, 2:] - gt32[:, :2]
    if not bool(torch.isfinite(wh).all()) or not bool((wh > 0).all()):
        raise ValueError("Assigned GT width and height must be positive and finite")
    if not bool(torch.isfinite(stride32).all()) or not bool((stride32 > 0).all()):
        raise ValueError("Native slot stride must be positive and finite")
    delta = ((pred32[:, :2] + pred32[:, 2:]) -
             (gt32[:, :2] + gt32[:, 2:])) * 0.5
    denominators = torch.maximum(wh, stride32[:, None])
    return 0.5 * (delta.abs() / denominators).sum(-1)


def axis_decoded_pixel_boxes(native, anchor: torch.Tensor,
                             pred_dist: torch.Tensor,
                             stride: torch.Tensor) -> torch.Tensor:
    """Decode the same raw box/anchor in FP32 for the new center component only."""
    return native.bbox_decode(anchor.float(), pred_dist.float()) * stride.float()


def assigned_geometry_loss(iou: torch.Tensor, d2: torch.Tensor,
                           pixel_boxes: torch.Tensor, variant: str,
                           axis_center: torch.Tensor | None = None) -> torch.Tensor:
    """The production reduction for assigned positives, including the empty branch."""
    if variant not in {"original", "axis_center_v1"}:
        raise ValueError("Unknown geometry variant")
    if iou.shape != d2.shape or iou.ndim != 1:
        raise ValueError("Expected aligned positive IoU and d2 vectors")
    if iou.numel() == 0:
        return pixel_boxes.sum() * 0.0
    overlap = 2.0 - 2.0 ** iou
    if variant == "original":
        return (overlap + d2).mean()
    if axis_center is None or axis_center.shape != iou.shape:
        raise ValueError("Expected axis center term for each assigned positive")
    return (overlap + axis_center).mean()


class V6A2Loss:
    def __init__(self, model, geo_variant: str = "original"):
        if geo_variant not in {"original", "axis_center_v1"}:
            raise ValueError("Unknown geometry variant")
        self.geo_variant = geo_variant
        self.native = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
        if isinstance(self.native.hyp, dict):
            self.native.hyp = SimpleNamespace(**{**vars(DEFAULT_CFG), **self.native.hyp})
        if (float(self.native.hyp.box), float(self.native.hyp.cls), float(self.native.hyp.dfl)) != (7.5, 0.5, 1.5):
            raise RuntimeError("Native loss gains differ from frozen V6-A2 protocol")
        population = json.loads(POPULATION.read_text())
        self.sources = {}
        for split in ("train", "validation"):
            for row in population[split]:
                self.sources[str(row["path"])] = str(row["source"])
        eligibility = yaml.safe_load(ELIGIBILITY.read_text())
        self.classes = eligibility["classes"]
        self.matrix = eligibility["matrix"]

    def __call__(self, preds: dict, batch: dict, mode: str,
                 lambda_geo: float = 0.0, epoch: int = 0) -> tuple[torch.Tensor, dict]:
        if mode not in {"q_only", "box_native", "box_geo"}:
            raise ValueError("Invalid V6-A2 mode")
        raw = preds["one2one"]
        # The native assigner sees A0 O2O classification logits, never corrected scores.
        original = {"boxes": raw["boxes"], "scores": raw["original_logits"], "feats": raw["feats"]}
        (fg_mask, gt_idx, assigned_boxes, anchor, stride), native_terms, _ = self.native.get_assigned_targets_and_loss(original, batch)
        pred_dist = raw["boxes"].permute(0, 2, 1).contiguous()
        bs = pred_dist.shape[0]
        pixel_boxes = self.native.bbox_decode(anchor, pred_dist) * stride

        # Reuse the exact same native assigner only to obtain target_scores.
        imgsz = torch.tensor(raw["feats"][0].shape[2:], device=pred_dist.device, dtype=pred_dist.dtype) * self.native.stride[0]
        rows = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.native.preprocess(rows.to(pred_dist.device), bs, imgsz[[1, 0, 1, 0]])
        gt_labels, gt_boxes = targets.split((1, 4), 2)
        mask_gt = gt_boxes.sum(2, keepdim=True).gt(0)
        _, check_boxes, target_scores, check_fg, check_idx = self.native.assigner(
            raw["original_logits"].detach().permute(0, 2, 1).sigmoid(),
            pixel_boxes.detach().to(gt_boxes.dtype), anchor * stride,
            gt_labels, gt_boxes, mask_gt)
        if not (torch.equal(fg_mask, check_fg) and torch.equal(gt_idx, check_idx)
                and torch.equal(assigned_boxes, check_boxes)):
            raise RuntimeError("Native O2O assignment changed during audit replay")
        # Ultralytics returns zeros_like(pd_scores[..., 0]) for an entirely
        # GT-free batch, so its otherwise boolean fg_mask has score dtype.
        # Normalize only that documented empty-batch branch for our indexing.
        empty_gt_batch = not bool(mask_gt.any())
        if fg_mask.dtype != torch.bool:
            if not empty_gt_batch or bool(fg_mask.any()):
                raise RuntimeError("Nonboolean native foreground mask outside empty-GT branch")
            fg_mask = fg_mask.bool()
        gt_idx = gt_idx.long()

        qlog = raw["quality_logits"].permute(0, 2, 1)
        z0 = raw["original_logits"].detach().permute(0, 2, 1)
        q_losses, q_counts = [], []
        for bi in range(bs):
            valid = torch.where(mask_gt[bi, :, 0])[0]
            gt_b = gt_boxes[bi, valid]
            gt_c = gt_labels[bi, valid, 0].long()
            assigned = torch.full_like(gt_idx[bi], -1)
            assigned[fg_mask[bi]] = gt_idx[bi, fg_mask[bi]]
            image_file = str(batch["im_file"][bi])
            if image_file not in self.sources:
                raise KeyError(f"Training/validation source metadata missing: {image_file}")
            source = self.sources[image_file]
            statuses = [self.matrix[source][name] for name in self.classes]
            targets_q = quality_targets(pixel_boxes[bi], gt_b, gt_c, statuses, assigned)
            sampled = sample_q_rows(targets_q, z0[bi], gt_c, assigned, seed=42 + epoch)
            q_losses.append(continuous_q_bce(qlog[bi], targets_q["target"], sampled["weight"]))
            q_counts.append(sampled["counts"])
        q_loss = torch.stack(q_losses).mean()

        # Native v8DetectionLoss.loss multiplies its components by batch size.
        box_native = (native_terms[0] + native_terms[2]) * bs
        geo = assigned_geometry_loss(pixel_boxes.new_empty(0), pixel_boxes.new_empty(0),
                                     pixel_boxes, self.geo_variant)
        geo_stats = {}
        if fg_mask.any():
            # Native assigner returns target_bboxes in image pixels already.
            assigned_gt_pixel = assigned_boxes
            assigned_positive = assigned_gt_pixel[fg_mask]
            if (assigned_positive[:, [0, 2]] > imgsz[1] + 1).any() or (assigned_positive[:, [1, 3]] > imgsz[0] + 1).any():
                raise RuntimeError("Native assigned GT appears to have been scaled twice")
            iou, d2 = aligned_geometry(pixel_boxes[fg_mask], assigned_gt_pixel[fg_mask])
            overlap = 2.0 - 2.0 ** iou
            axis_center = None
            if self.geo_variant == "axis_center_v1":
                if stride.shape != (pixel_boxes.shape[1], 1):
                    raise RuntimeError("Native slot stride shape differs from [slots,1]")
                positive_stride = stride[:, 0].unsqueeze(0).expand(bs, -1)[fg_mask]
                axis_boxes = axis_decoded_pixel_boxes(self.native, anchor,
                                                       pred_dist, stride)
                if axis_boxes.shape != pixel_boxes.shape:
                    raise RuntimeError("FP32 axis decode shape differs from native boxes")
                axis_center = axis_center_term(axis_boxes[fg_mask],
                                               assigned_positive, positive_stride)
            geo = assigned_geometry_loss(iou, d2, pixel_boxes, self.geo_variant,
                                         axis_center)
            with torch.no_grad():
                for name, values in (("d2", d2), ("overlap", overlap),
                                     ("weight", 1.0 - iou.detach())):
                    geo_stats[name] = {key: float(val) for key, val in zip(
                        ("median", "p90", "p95", "p99", "max"),
                        (values.median(), *torch.quantile(values.float(), torch.tensor([.9, .95, .99], device=values.device)), values.max()))}
                    geo_stats[name]["mean"] = float(values.float().mean())
                geo_stats["weight"]["mean"] = float((1.0 - iou.detach()).float().mean())
                if axis_center is not None:
                    geo_stats["axis_center"] = {key: float(val) for key, val in zip(
                        ("median", "p90", "p95", "p99", "max"),
                        (axis_center.median(), *torch.quantile(axis_center.float(), torch.tensor([.9, .95, .99], device=axis_center.device)), axis_center.max()))}
                    geo_stats["axis_center"]["mean"] = float(axis_center.float().mean())
                    gt_wh = assigned_positive[:, 2:] - assigned_positive[:, :2]
                    axis_positive = axis_boxes[fg_mask]
                    delta_xy = ((axis_positive[:, :2] + axis_positive[:, 2:]) -
                                (assigned_positive[:, :2] + assigned_positive[:, 2:])) * 0.5
                    for name, values in (("gt_width", gt_wh[:, 0]),
                                         ("gt_height", gt_wh[:, 1]),
                                         ("gt_aspect_width_over_height", gt_wh[:, 0] / gt_wh[:, 1]),
                                         ("axis_abs_dx", delta_xy[:, 0].abs()),
                                         ("axis_abs_dy", delta_xy[:, 1].abs())):
                        geo_stats[name] = {key: float(val) for key, val in zip(
                            ("median", "p90", "p95", "p99", "max"),
                            (values.median(), *torch.quantile(values.float(), torch.tensor([.9, .95, .99], device=values.device)), values.max()))}
                        geo_stats[name]["mean"] = float(values.float().mean())
        total = q_loss if mode == "q_only" else box_native + q_loss
        if mode == "box_geo":
            if lambda_geo <= 0:
                raise ValueError("box_geo requires frozen positive lambda_geo")
            total = total + lambda_geo * geo
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite V6-A2 loss")
        audit = {"native_box": float(box_native.detach()), "native_cls_diagnostic": float(native_terms[1].detach()),
                 "q": float(q_loss.detach()), "geo": float(geo.detach()), "geo_stats": geo_stats,
                 "native_empty_gt_batch": empty_gt_batch,
                 "fg_count": int(fg_mask.sum()), "target_scores_sum": float(target_scores.sum()),
                 "assigned_positive_preview": [
                     {"image_index": int(bi), "slot": int(slot), "gt_index": int(gt_idx[bi, slot]),
                      "gt_xyxy_pixels": assigned_boxes[bi, slot].detach().float().tolist(),
                      "target_scores": target_scores[bi, slot].detach().float().tolist()}
                     for bi, slot in torch.nonzero(fg_mask, as_tuple=False)[:16].tolist()],
                 "q_sample_counts": q_counts}
        return total, audit
