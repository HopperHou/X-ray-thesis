"""Official full validation plus raw/Top300/final and E-cell duplicate diagnostics."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from v6a2_eval import two_stage_topk
from v6a2_model import install_v6a2_head

ROOT = Path(__file__).resolve().parent


def summarize_ap_matrix(class_ids, all_ap, actual_iouv, official_ap50,
                        official_map, class_names):
    class_ids = [int(cid) for cid in class_ids]
    all_ap = np.asarray(all_ap, dtype=np.float64)
    if sorted(class_ids) != list(range(6)) or all_ap.shape != (6, 10):
        raise RuntimeError(f"Incomplete six-class by ten-IoU AP matrix: {class_ids}, {all_ap.shape}")
    if not np.isfinite(all_ap).all():
        raise RuntimeError("Nonfinite per-IoU AP")
    frozen_iouv = np.asarray([0.50 + 0.05 * j for j in range(10)])
    if not np.allclose(np.asarray(actual_iouv), frozen_iouv, rtol=0, atol=1e-6):
        raise RuntimeError("Actual validator IoU thresholds differ from .50:.05:.95")
    ordered_ap = np.empty((6, 10), dtype=np.float64)
    for cid, ap_row in zip(class_ids, all_ap):
        ordered_ap[cid] = ap_row
    macro_ap = ordered_ap.mean(axis=0)
    if not np.isclose(macro_ap[0], float(official_ap50), rtol=1e-9, atol=1e-10):
        raise RuntimeError("Exported AP50 differs from official AP50")
    if not np.isclose(macro_ap.mean(), float(official_map), rtol=1e-9, atol=1e-10):
        raise RuntimeError("Exported mAP50-95 differs from official mAP50-95")
    thresholds = [f"{t:.2f}" for t in frozen_iouv]
    return {"thresholds": thresholds, "class_ids": list(range(6)),
            "class_names": class_names, "ap_by_class": ordered_ap.tolist(),
            "macro_ap": {t: float(macro_ap[j]) for j, t in enumerate(thresholds)},
            "H": float(macro_ap[5:].mean())}


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if len(boxes) == 0:
        return boxes.new_zeros((0,))
    intersection = (torch.minimum(box[2:], boxes[:, 2:]) - torch.maximum(box[:2], boxes[:, :2])).clamp_min(0).prod(-1)
    a = (box[2:] - box[:2]).clamp_min(0).prod()
    b = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    return intersection / (a + b - intersection).clamp_min(1e-12)


def evaluate(protocol: dict, checkpoint: Path, output: Path, scorer: tuple[float, float, float, float],
             device: str = "cuda:0", batch: int = 8, workers: int = 8,
             validation_data: str | None = None, expected_images: int = 1660) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    is_a0 = str(checkpoint.resolve()) == str(Path(protocol["a0_checkpoint"]).resolve())
    if not is_a0 and "model_state" not in payload:
        raise RuntimeError("Checkpoint lacks full model state")
    detector = YOLO(protocol["a0_checkpoint"])
    install_v6a2_head(detector.model, scorer=scorer)
    if not is_a0:
        detector.model.load_state_dict(payload["model_state"], strict=True)
    head = detector.model.model[-1]
    head.fuse = types.MethodType(lambda self: None, head)
    population_path = Path(protocol["pilot"]["population_json"])
    population = json.loads(population_path.read_text())
    source_by_path = {str(row["path"]): str(row["source"]) for split in ("train", "validation")
                      for row in population[split]}
    eligibility = yaml.safe_load(Path(protocol["eligibility_matrix"]).read_text())["matrix"]
    class_names = protocol["class_names"]
    records = []
    boot_records = []
    raw_by_path = {}
    actual_iouv = []

    class AuditValidator(DetectionValidator):
        def preprocess(self, input_batch):
            self.audit_paths = [str(p) for p in input_batch["im_file"]]
            return super().preprocess(input_batch)

        def postprocess(self, preds):
            raw = preds[1]["one2one"]
            decoded = head._inference(raw).permute(0, 2, 1)
            for bi, image in enumerate(self.audit_paths):
                trace = two_stage_topk(decoded[bi, :, :4], decoded[bi, :, 4:], 300, 0.001)
                raw_by_path[image] = {"boxes": decoded[bi, :, :4].detach().float().cpu(),
                                      "first": trace.first_slots.detach().cpu(),
                                      "final_boxes": trace.final_boxes.detach().float().cpu(),
                                      "final_cls": trace.final_classes.detach().cpu(),
                                      "first_sha256": tensor_sha(trace.first_slots),
                                      "second_slots_sha256": tensor_sha(trace.second_slots),
                                      "second_classes_sha256": tensor_sha(trace.second_classes),
                                      "final_scores_sha256": tensor_sha(trace.final_scores),
                                      "final_count": int(len(trace.final_scores))}
            return super().postprocess(preds)

        def update_metrics(self, preds, input_batch):
            if not actual_iouv:
                actual_iouv.extend(self.iouv.detach().cpu().tolist())
            super().update_metrics(preds, input_batch)
            for bi, pred in enumerate(preds):
                prepared = self._prepare_batch(bi, input_batch)
                predn = self._prepare_pred(pred)
                image = str(prepared["im_file"])
                raw = raw_by_path.pop(image)
                if image not in source_by_path:
                    raise RuntimeError(f"Validation source metadata missing: {image}")
                source = source_by_path[image]
                gt_boxes = prepared["bboxes"].detach().float().cpu()
                gt_classes = prepared["cls"].detach().long().cpu()
                pred_boxes = predn["bboxes"].detach().float().cpu()
                pred_classes = predn["cls"].detach().long().cpu()
                official_tp = self._process_batch(predn, prepared)["tp"]
                official_tp = torch.as_tensor(official_tp, dtype=torch.bool).cpu().numpy()
                boot_records.append({"image": image, "tp": official_tp,
                                     "conf": predn["conf"].detach().cpu().numpy(),
                                     "pred_cls": pred_classes.numpy(),
                                     "target_cls": gt_classes.numpy()})
                tp = official_tp[:, 0]
                tp = torch.as_tensor(tp, dtype=torch.bool).cpu()
                e_gt = raw_good = first_good = final_good = 0
                duplicate_indices = set()
                for gi, (box, cls_tensor) in enumerate(zip(gt_boxes, gt_classes)):
                    cls = int(cls_tensor)
                    if eligibility[source][class_names[cls]] != "E":
                        continue
                    e_gt += 1
                    raw_iou = iou_one_to_many(box, raw["boxes"])
                    first_iou = raw_iou[raw["first"]]
                    final_iou = iou_one_to_many(box, raw["final_boxes"])
                    final_same = raw["final_cls"] == cls
                    raw_good += int(bool((raw_iou >= 0.5).any()))
                    first_good += int(bool((first_iou >= 0.5).any()))
                    final_good += int(bool(((final_iou >= 0.5) & final_same).any()))
                    pred_iou = iou_one_to_many(box, pred_boxes)
                    near = (pred_iou >= 0.5) & (pred_classes == cls)
                    if bool((near & tp).any()):
                        duplicate_indices.update(torch.where(near & ~tp)[0].tolist())
                records.append({"image": image, "source": source, "E_GT": e_gt,
                                "raw_good_GT": raw_good, "first_top300_good_GT": first_good,
                                "final_good_GT": final_good, "duplicate_FP": len(duplicate_indices),
                                "output_count": raw["final_count"],
                                "first_slots_sha256": raw["first_sha256"],
                                "second_slots_sha256": raw["second_slots_sha256"],
                                "second_classes_sha256": raw["second_classes_sha256"],
                                "final_scores_sha256": raw["final_scores_sha256"]})

    result = detector.val(data=validation_data or protocol["data_yaml"], split="val", device=device,
                          imgsz=1024, batch=batch, workers=workers, conf=0.001,
                          max_det=300, agnostic_nms=False, plots=False, verbose=False,
                          project=str(output), name="official", exist_ok=False,
                          validator=AuditValidator)
    if len(records) != expected_images or raw_by_path:
        raise RuntimeError(f"Validation trace covered {len(records)} instead of {expected_images} images")
    if expected_images == 1660:
        expected_paths = [str(Path(p).resolve()) for p in
                          Path(protocol["development_validation_list"]).read_text().splitlines()]
        actual_paths = [str(Path(row["image"]).resolve()) for row in records]
        if (len(set(expected_paths)) != 1660 or len(set(actual_paths)) != 1660 or
                sorted(expected_paths) != sorted(actual_paths)):
            raise RuntimeError("Official evaluation images differ from frozen development list")
    with (output / "trace.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    save_bootstrap_records(output, boot_records)
    totals = {key: sum(row[key] for row in records)
              for key in ("E_GT", "raw_good_GT", "first_top300_good_GT", "final_good_GT", "duplicate_FP")}
    per_iou = summarize_ap_matrix(result.box.ap_class_index, result.box.all_ap,
                                  actual_iouv, result.box.map50, result.box.map,
                                  class_names)
    (output / "per_iou_ap.json").write_text(json.dumps(per_iou, indent=2) + "\n")
    report = {"status": "PASS", "split": "development_validation" if expected_images == 1660 else "development_smoke",
              "images": len(records),
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "scorer": {"delta_plus": scorer[0], "delta_minus": scorer[1], "b": scorer[2], "k": scorer[3]},
              "mAP50": float(result.box.map50), "mAP50_95": float(result.box.map),
              "per_iou_ap": str(output / "per_iou_ap.json"), "H": per_iou["H"],
              "per_class": {str(int(cid)): {"name": result.names[int(cid)], "AP50": float(ap[0]),
                                                "AP50_95": float(ap.mean())}
                            for cid, ap in zip(result.box.ap_class_index, result.box.all_ap)},
              "mechanism": {**totals, "output_count_median": float(np.median([row["output_count"] for row in records])),
                            "output_count_max": max(row["output_count"] for row in records)},
              "trace": str(output / "trace.jsonl"), "TEST_DATA_READ": 0}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def save_bootstrap_records(output, rows):
    pred_offsets=np.cumsum([0]+[len(r['conf']) for r in rows])
    gt_offsets=np.cumsum([0]+[len(r['target_cls']) for r in rows])
    np.savez_compressed(output/'bootstrap_records.npz', images=np.asarray([r['image'] for r in rows]),
        pred_offsets=pred_offsets, gt_offsets=gt_offsets,
        tp=np.concatenate([r['tp'] for r in rows]), conf=np.concatenate([r['conf'] for r in rows]),
        pred_cls=np.concatenate([r['pred_cls'] for r in rows]),target_cls=np.concatenate([r['target_cls'] for r in rows]))
