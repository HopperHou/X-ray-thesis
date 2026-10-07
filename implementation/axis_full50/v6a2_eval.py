"""Exact native two-stage O2O Top300 trace for full raw-slot score replay."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import types
from pathlib import Path

import torch

from v6a2_model import corrected_logits


@dataclass
class TopKTrace:
    first_slots: torch.Tensor
    second_slots: torch.Tensor
    second_classes: torch.Tensor
    second_scores: torch.Tensor
    final_slots: torch.Tensor
    final_classes: torch.Tensor
    final_scores: torch.Tensor
    final_boxes: torch.Tensor


def two_stage_topk(boxes: torch.Tensor, scores: torch.Tensor, k: int = 300,
                   floor: float = 0.001) -> TopKTrace:
    """Mirror Detect.get_topk_index, preserving torch.topk tie behavior."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 2 or len(boxes) != len(scores):
        raise ValueError("Expected boxes [N,4], scores [N,C]")
    if not 0 < k <= len(boxes):
        raise ValueError("Invalid TopK")
    nc = scores.shape[1]
    first = scores.max(dim=1).values.topk(k).indices
    selected = scores[first].flatten()
    second_score, flat_index = selected.topk(k)
    second_slot = first[flat_index // nc]
    second_class = flat_index % nc
    keep = second_score > floor
    return TopKTrace(first, second_slot, second_class, second_score,
                     second_slot[keep], second_class[keep], second_score[keep], boxes[second_slot[keep]])


def replay_raw(boxes: torch.Tensor, z0: torch.Tensor, q_logits: torch.Tensor,
               scorer: tuple[float, float, float, float], k: int = 300,
               floor: float = 0.001) -> TopKTrace:
    """Correct every slot and every class before either Top300 operation."""
    z = corrected_logits(z0, q_logits, *scorer)
    return two_stage_topk(boxes, z.sigmoid(), k, floor)


def compare_native(head, raw: dict, replay: TopKTrace, floor: float = 0.001) -> None:
    """Assert both actual native index stages, boxes, classes and scores match."""
    decoded = head._inference(raw).permute(0, 2, 1)
    if decoded.shape[0] != 1:
        raise ValueError("Use batch size one for index comparison")
    boxes, scores = decoded[0, :, :4], decoded[0, :, 4:]
    k = min(head.max_det, len(boxes))
    first_native = scores.max(dim=1).values.topk(k).indices
    final_score, final_class, final_slot = head.get_topk_index(scores.unsqueeze(0), head.max_det)
    final_score = final_score[0, :, 0]
    final_class = final_class[0, :, 0].long()
    final_slot = final_slot[0, :, 0]
    if not torch.equal(first_native, replay.first_slots):
        raise AssertionError("First TopK raw slot indices differ")
    if not torch.equal(final_slot, replay.second_slots) or not torch.equal(final_class, replay.second_classes):
        raise AssertionError("Second TopK slot/class indices differ")
    if not torch.equal(final_score, replay.second_scores):
        raise AssertionError("Second TopK scores differ")
    keep = final_score > floor
    if not torch.equal(boxes[final_slot[keep]], replay.final_boxes):
        raise AssertionError("Final boxes differ")


def assert_correction_before_first_topk() -> None:
    """Construct a raw rank-301 slot promoted by the frozen grid scorer."""
    n, nc = 301, 6
    boxes = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
    z0 = torch.full((n, nc), -12.0)
    z0[:300, 0] = 0.0
    z0[300, 0] = -0.1
    qlog = torch.full_like(z0, -12.0)
    qlog[300, 0] = 12.0
    original = replay_raw(boxes, z0, qlog, (0, 0, 0.35, 4))
    corrected = replay_raw(boxes, z0, qlog, (2, 0, 0.35, 4))
    assert 300 not in original.first_slots.tolist()
    assert 300 in corrected.first_slots.tolist()
    assert 300 in corrected.second_slots.tolist()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _result_metrics(result) -> dict:
    return {"mAP50": float(result.box.map50), "mAP50_95": float(result.box.map),
            "per_class": {str(int(c)): {"name": result.names[int(c)],
                            "AP50": float(row[0]), "AP50_95": float(row.mean())}
                          for c, row in zip(result.box.ap_class_index, result.box.all_ap)}}


def run_identity_validation(protocol_path: Path, device: str, batch: int, workers: int,
                            smoke_images: int = 0) -> dict:
    """Full canonical validation twice, with per-image native index/score hashes."""
    from ultralytics import YOLO, __version__
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from v6a2_model import install_v6a2_head

    protocol = json.loads(protocol_path.read_text())
    if __version__ != protocol["ultralytics_version"]:
        raise RuntimeError("Ultralytics version drift")
    if protocol["inference"] != {"branch": "one2one", "nms": "none", "first_topk": 300,
                                  "second_topk": 300, "score_floor_strict_gt": 0.001}:
        raise RuntimeError("Identity validation protocol drift")
    output_dir = protocol_path.parent / ("p1_identity_validation_smoke" if smoke_images else "p1_identity_validation")
    output_dir.mkdir(exist_ok=True)
    data_yaml = protocol["data_yaml"]
    expected_images = 1660
    if smoke_images:
        import yaml
        images = Path(protocol["development_validation_list"]).read_text().splitlines()[:smoke_images]
        if len(images) != smoke_images:
            raise RuntimeError("Insufficient validation images for smoke")
        subset = output_dir / "validation_smoke.txt"
        subset.write_text("\n".join(images) + "\n")
        data = yaml.safe_load(Path(data_yaml).read_text())
        data["val"] = str(subset)
        data_yaml = str(output_dir / "data_smoke.yaml")
        Path(data_yaml).write_text(yaml.safe_dump(data, sort_keys=False))
        expected_images = smoke_images
    baseline_rows = {}
    results = {}

    for label in ("native", "v6a2_zero"):
        detector = YOLO(protocol["a0_checkpoint"])
        head = detector.model.model[-1]
        head.fuse = types.MethodType(lambda self: None, head)
        if label == "v6a2_zero":
            install_v6a2_head(detector.model)
            head = detector.model.model[-1]
            head.fuse = types.MethodType(lambda self: None, head)
        trace_path = output_dir / f"{label}_trace.jsonl"
        if trace_path.exists():
            raise FileExistsError(f"Refuse to overwrite identity trace: {trace_path}")

        class TraceValidator(DetectionValidator):
            def preprocess(self, input_batch):
                self.trace_paths = list(input_batch["im_file"])
                return super().preprocess(input_batch)

            def postprocess(self, preds):
                raw = preds[1]["one2one"]
                decoded = head._inference(raw).permute(0, 2, 1)
                with trace_path.open("a") as out:
                    for bi, path in enumerate(self.trace_paths):
                        trace = two_stage_topk(decoded[bi, :, :4], decoded[bi, :, 4:], 300)
                        row = {"image": str(path), "first_slots_sha256": _tensor_hash(trace.first_slots),
                               "second_slots_sha256": _tensor_hash(trace.second_slots),
                               "second_classes_sha256": _tensor_hash(trace.second_classes),
                               "second_scores_sha256": _tensor_hash(trace.second_scores),
                               "final_scores_sha256": _tensor_hash(trace.final_scores),
                               "final_boxes_sha256": _tensor_hash(trace.final_boxes),
                               "final_count": int(len(trace.final_scores))}
                        if label == "native":
                            if path in baseline_rows:
                                raise RuntimeError(f"Duplicate validation image: {path}")
                            baseline_rows[path] = row
                        elif row != baseline_rows.get(path):
                            raise AssertionError(f"V6-A2 zero scorer differs from native at {path}")
                        out.write(json.dumps(row, sort_keys=True) + "\n")
                return super().postprocess(preds)

        result = detector.val(data=data_yaml, split="val", device=device,
                              imgsz=1024, batch=batch, workers=workers, conf=0.001,
                              max_det=300, agnostic_nms=False, plots=False, verbose=False,
                              project=str(output_dir), name=label, exist_ok=False,
                              validator=TraceValidator)
        results[label] = _result_metrics(result)
    if len(baseline_rows) != expected_images:
        raise AssertionError(f"Expected {expected_images} validation images, saw {len(baseline_rows)}")
    if results["native"] != results["v6a2_zero"]:
        raise AssertionError("Native versus zero-correction full AP differs")
    report = {"status": "SMOKE_PASS" if smoke_images else "PASS", "images": len(baseline_rows), "device": device,
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "batch": batch, "workers": workers, "results": results,
              "per_image_two_topk_scores_boxes_identity": True, "TEST_DATA_READ": 0,
              "trace_paths": [str(output_dir / f"{label}_trace.jsonl") for label in ("native", "v6a2_zero")]}
    report_path = output_dir / ("P1_IDENTITY_SMOKE_RESULT.json" if smoke_images else "P1_IDENTITY_RESULT.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("identity_validation",), required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--smoke-images", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_identity_validation(args.protocol, args.device, args.batch, args.workers,
                                             args.smoke_images), indent=2))
