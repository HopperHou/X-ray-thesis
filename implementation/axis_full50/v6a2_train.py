"""V6-A2 Q-only training gates and patient-fold training, no site-package edits."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset, DATASET_CACHE_VERSION
from ultralytics.data.utils import (check_det_dataset, get_hash, img2label_paths,
                                    load_dataset_cache_file)
from ultralytics.data.build import seed_worker

from v6a2_loss import V6A2Loss
from v6a2_model import install_v6a2_head, set_trainable

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if ultralytics_version != protocol["ultralytics_version"]:
        raise RuntimeError("Ultralytics version drift")
    amended = path.resolve() == (ROOT / "protocol.m2_fold12_b60w30.json").resolve()
    frozen = ROOT / ("protocol.m2_fold12_b60w30.frozen.json" if amended else "protocol.pre_m2_frozen.json")
    if not frozen.is_file() or sha(frozen) != sha(path):
        raise RuntimeError("M2 protocol drift from frozen pre-training snapshot")
    if amended:
        original = json.loads((ROOT / "protocol.pre_m2_frozen.json").read_text())
        expected = copy.deepcopy(original)
        expected["training_data_rule"].update({"global_batch": 60, "workers": 30})
        expected["m2_amendment"] = {
            "scope": "outer_folds_1_and_2_only",
            "fold0_protocol_sha256": sha(ROOT / "protocol.pre_m2_frozen.json"),
            "reason": "user_requested_batch60_workers30_from_second_fold",
            "fold0_artifacts_immutable": True}
        if protocol != expected:
            raise RuntimeError("Fold1/2 amendment changed fields beyond batch/workers")
    splits = json.loads((ROOT / "splits/M2_SPLITS.json").read_text())
    split_protocol_hash = sha(ROOT / "protocol.pre_m2_frozen.json") if amended else sha(path)
    if splits["status"] != "PASS" or splits["protocol_sha256"] != split_protocol_hash:
        raise RuntimeError("M2 patient split protocol mismatch")
    p1 = json.loads((ROOT / "p1_identity_validation/P1_IDENTITY_RESULT.json").read_text())
    if p1["status"] != "PASS" or p1["images"] != 1660:
        raise RuntimeError("M0 full identity gate missing")
    return protocol


class FrozenPopulationDataset(YOLODataset):
    """Subset native cached A0 labels without rewriting the shared train.cache."""

    full_train_list: Path
    cache_path: Path

    def get_labels(self):
        all_paths = self.full_train_list.read_text().splitlines()
        cache = load_dataset_cache_file(self.cache_path)
        if (cache["version"] != DATASET_CACHE_VERSION or
                cache["hash"] != get_hash(img2label_paths(all_paths) + all_paths) or
                len(cache["labels"]) != len(all_paths)):
            raise RuntimeError("Canonical train label cache does not match A0 train list")
        by_path = {item["im_file"]: item for item in cache["labels"]}
        if len(by_path) != len(all_paths):
            raise RuntimeError("Duplicate path in canonical train label cache")
        missing = set(self.im_files) - set(by_path)
        if missing:
            raise RuntimeError(f"Missing {len(missing)} subset labels in train cache")
        chosen = [copy.deepcopy(by_path[path]) for path in self.im_files]
        self.label_files = img2label_paths(self.im_files)
        return chosen


def make_loader(protocol: dict, list_path: Path, *, augment: bool, batch: int, workers: int,
                shuffle: bool, seed: int, pin_memory: bool = False) -> DataLoader:
    if not list_path.is_file():
        raise FileNotFoundError(list_path)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 1024, "rect": False, "cache": False,
                             "batch": batch, "workers": workers, "mosaic": 0.0, "mixup": 0.0,
                             "cutmix": 0.0, "copy_paste": 0.0, "deterministic": True, "seed": seed})
    data = check_det_dataset(protocol["data_yaml"])
    if [data["names"][i] for i in range(6)] != protocol["class_names"]:
        raise RuntimeError("Class order changed")
    FrozenPopulationDataset.full_train_list = Path(protocol["train_list"])
    FrozenPopulationDataset.cache_path = FrozenPopulationDataset.full_train_list.parent / "labels/train.cache"
    dataset = FrozenPopulationDataset(img_path=str(list_path), imgsz=1024, batch_size=batch,
                                      augment=augment, hyp=cfg, rect=False, cache=False,
                                      single_cls=False, stride=32, pad=0.0 if augment else 0.5,
                                      task="detect", data=data, fraction=1.0, prefix="v6a2: ")
    expected = len(list_path.read_text().splitlines())
    if len(dataset) != expected:
        raise RuntimeError(f"Dataset skipped images: {len(dataset)} versus {expected}")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      collate_fn=dataset.collate_fn, pin_memory=pin_memory,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=workers > 0)


def model_and_loss(protocol: dict, device: torch.device, mode: str):
    torch.manual_seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    random.seed(protocol["seed"])
    model = YOLO(protocol["a0_checkpoint"]).model
    actual = install_v6a2_head(model)
    model.to(device)
    q = protocol["quality_head"]
    if actual["h_cls_channels"] != q["h_cls_channels_verified"] or actual["h_box_channels"] != q["h_box_channels_verified"]:
        raise RuntimeError("Q head checkpoint channels changed")
    model.train()
    set_trainable(model, mode)
    criterion = V6A2Loss(model)
    return model, criterion


def move_batch(batch: dict, device: torch.device) -> dict:
    result = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
    result["img"] = result["img"].float() / 255
    return result


def amp_context(protocol: dict, device: torch.device):
    if protocol["training_data_rule"]["amp"] and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def cpu_dataset_gate(protocol: dict) -> dict:
    path = ROOT / "splits/fold0/inner_train.txt"
    loader = make_loader(protocol, path, augment=True, batch=1, workers=0, shuffle=False, seed=42)
    batch = move_batch(next(iter(loader)), torch.device("cpu"))
    if batch["img"].shape != (1, 3, 1024, 1024):
        raise RuntimeError("Training augmentation changed fixed 1024 image shape")
    if batch["im_file"][0] not in Path(protocol["train_list"]).read_text().splitlines():
        raise RuntimeError("Training loader emitted a foreign image")
    model, criterion = model_and_loss(protocol, torch.device("cpu"), "q_only")
    loss, audit = criterion(model(batch["img"]), batch, "q_only", epoch=1)
    loss.backward()
    q_grad = sum(float(p.grad.float().norm()) for p in model.model[-1].v6a2_quality_heads.parameters()
                 if p.grad is not None)
    if not torch.isfinite(loss) or q_grad <= 0:
        raise RuntimeError("CPU dataset Q-only loss/gradient failed")
    if any(p.grad is not None for p in model.model[-1].one2one_cv2.parameters()):
        raise RuntimeError("CPU dataset Q-only changed box gradient path")
    result = {"status": "PASS", "image": batch["im_file"][0], "loss": float(loss.detach()),
              "q_gradient_norm_sum": q_grad, "native_audit": audit,
              "TEST_DATA_READ": 0, "GPU_ALLOCATED": False}
    (ROOT / "M2_CPU_DATASET_GATE.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: result[k] for k in ("status", "image", "loss", "q_gradient_norm_sum")}


def benchmark(protocol: dict, protocol_path: Path, device: torch.device, steps: int, batch_size: int, workers: int) -> dict:
    gate_path = ROOT / "M2_CPU_DATASET_GATE.json"
    if not gate_path.is_file() or json.loads(gate_path.read_text())["status"] != "PASS":
        raise RuntimeError("CPU dataset gate must PASS before GPU optimizer benchmark")
    if steps != 100:
        raise ValueError("Frozen M0/M2 budget gate requires 100 optimizer steps")
    fold = 1 if "m2_amendment" in protocol else 0
    loader = make_loader(protocol, ROOT / "splits" / f"fold{fold}" / "inner_train.txt", augment=True,
                         batch=batch_size, workers=workers, shuffle=True, seed=42,
                         pin_memory=device.type == "cuda")
    model, criterion = model_and_loss(protocol, device, "q_only")
    optimizer = torch.optim.AdamW(model.model[-1].v6a2_quality_heads.parameters(), lr=1e-4, weight_decay=5e-4)
    warmup = None
    start = time.monotonic()
    records = []
    for step, unprocessed in enumerate(loader):
        batch = move_batch(unprocessed, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(protocol, device):
            loss, audit = criterion(model(batch["img"]), batch, "q_only", epoch=1)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step == 9:
            warmup = time.monotonic() - start
        records.append({"step": step + 1, "loss": float(loss.detach()), "fg_count": audit["fg_count"]})
        if step + 1 == steps:
            break
    elapsed = time.monotonic() - start
    if len(records) != steps:
        raise RuntimeError("Insufficient training batches for 100-step benchmark")
    report = {"status": "PASS", "device": str(device), "batch": batch_size,
              "workers": workers, "optimizer_steps": steps, "elapsed_seconds": elapsed,
              "first_10_seconds": warmup, "seconds_per_step_after_warmup": (elapsed - warmup) / (steps - 10),
              "records": records, "checkpoint_saved": False, "TEST_DATA_READ": 0,
              "amp": protocol["training_data_rule"]["amp"], "amp_dtype": "bfloat16" if device.type == "cuda" else None,
              "protocol_sha256": sha(protocol_path)}
    report_path = ROOT / ("M2_FOLD12_B60W30_100_STEP_BENCHMARK.json" if "m2_amendment" in protocol
                          else "M2_100_STEP_BENCHMARK.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {k: report[k] for k in ("status", "batch", "optimizer_steps", "elapsed_seconds",
                                  "seconds_per_step_after_warmup")}


def _check_benchmark(protocol: dict) -> dict:
    path = ROOT / ("M2_FOLD12_B60W30_100_STEP_BENCHMARK.json" if "m2_amendment" in protocol
                   else "M2_100_STEP_BENCHMARK.json")
    if not path.exists():
        raise RuntimeError("100-step measured GPU budget gate missing")
    report = json.loads(path.read_text())
    if report.get("status") != "PASS" or report.get("optimizer_steps") != 100:
        raise RuntimeError("100-step benchmark did not complete")
    if report.get("batch") != protocol["training_data_rule"]["global_batch"]:
        raise RuntimeError("Benchmark batch differs from frozen training batch")
    if report.get("amp") != protocol["training_data_rule"]["amp"] or report.get("amp_dtype") != "bfloat16":
        raise RuntimeError("Benchmark AMP mode differs from frozen M2 training")
    expected_protocol = ROOT / ("protocol.m2_fold12_b60w30.json" if "m2_amendment" in protocol else "protocol.json")
    if report.get("protocol_sha256") != sha(expected_protocol):
        raise RuntimeError("Benchmark protocol hash differs from frozen M2 protocol")
    return report


def _parameter_hash(parameters: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _save_checkpoint(path: Path, model, epoch: int, protocol_path: Path, phase: str,
                     outer_fold: int | None, q_initial_hash: str) -> None:
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload = {"model_state": state, "epoch": epoch, "protocol_sha256": sha(protocol_path),
               "phase": phase, "outer_fold": outer_fold, "q_initial_hash": q_initial_hash,
               "saved_utc": datetime.now(timezone.utc).isoformat()}
    tmp = path.with_suffix(".pending")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path, model, protocol_path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["protocol_sha256"] != sha(protocol_path):
        raise RuntimeError("Checkpoint protocol hash differs from frozen M2 protocol")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def _epoch_loop(protocol, model, criterion, loader, optimizer, device, epoch: int,
                *, max_steps: int | None = None) -> tuple[float, int, list[dict]]:
    model.train()
    set_trainable(model, "q_only")
    weighted_loss = 0.0
    image_count = 0
    audits = []
    for step, data in enumerate(loader):
        batch = move_batch(data, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(protocol, device):
            loss, audit = criterion(model(batch["img"]), batch, "q_only", epoch=epoch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite Q loss at epoch {epoch}, step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.model[-1].v6a2_quality_heads.parameters()):
            raise RuntimeError(f"Nonfinite Q gradient at epoch {epoch}, step {step}")
        optimizer.step()
        bs = len(batch["im_file"])
        weighted_loss += float(loss.detach()) * bs
        image_count += bs
        if step < 3:
            audits.append({"step": step + 1, "fg_count": audit["fg_count"],
                           "q_sample_counts": audit["q_sample_counts"]})
        if max_steps is not None and step + 1 >= max_steps:
            break
    return weighted_loss / image_count, image_count, audits


@torch.no_grad()
def _q_validation(protocol, model, criterion, loader, device) -> tuple[float, int]:
    model.eval()
    weighted_loss = 0.0
    image_count = 0
    for data in loader:
        batch = move_batch(data, device)
        with amp_context(protocol, device):
            output = model(batch["img"])
            raw = output[1] if isinstance(output, tuple) else output
            loss, _ = criterion(raw, batch, "q_only", epoch=0)
        bs = len(batch["im_file"])
        weighted_loss += float(loss) * bs
        image_count += bs
    return weighted_loss / image_count, image_count


def _training_phase(protocol: dict, protocol_path: Path, stage: str, fold: int,
                    device: torch.device, batch_size: int, workers: int,
                    run_id: str | None = None) -> dict:
    _check_benchmark(protocol)
    if stage not in {"inner_select", "outer_refit"} or fold not in (0, 1, 2):
        raise ValueError("Invalid Q-only fold phase")
    if "m2_amendment" in protocol and fold == 0:
        raise RuntimeError("Batch60 amendment is restricted to outer folds 1 and 2")
    if run_id is not None and (stage != "outer_refit" or fold != 1
                               or run_id != "outer_refit_empty_gt_fix_v2"):
        raise RuntimeError("Versioned retry id is restricted to failed fold1 refit")
    folder = ROOT / "runs" / f"fold{fold}" / (run_id or stage)
    if folder.exists():
        raise FileExistsError(f"Refuse to overwrite versioned fold run: {folder}")
    if batch_size != protocol["training_data_rule"]["global_batch"]:
        raise RuntimeError("Global batch differs from frozen M2 protocol")
    selected_epoch = None
    if stage == "outer_refit":
        inner_path = ROOT / "runs" / f"fold{fold}" / "inner_select" / "result.json"
        if not inner_path.exists():
            raise RuntimeError("Inner patient holdout must select e_i first")
        inner = json.loads(inner_path.read_text())
        if inner["status"] != "PASS" or len(inner["epochs"]) != 20:
            raise RuntimeError("Inner epoch selection did not complete 20 epochs")
        selected_epoch = int(inner["selected_epoch"])
    folder.mkdir(parents=True)
    frozen = {"status": "RUNNING", "phase": stage, "run_id": run_id or stage,
              "restart_from": "A0_and_seed42_Q_initialization" if run_id else None,
              "outer_fold": fold,
              "protocol_sha256": sha(protocol_path), "batch": batch_size, "workers": workers,
              "seed": protocol["seed"], "selected_epoch_from_inner": selected_epoch,
              "TEST_DATA_READ": 0, "epochs": []}
    (folder / "status.json").write_text(json.dumps(frozen, indent=2) + "\n")
    train_name = "inner_train" if stage == "inner_select" else "outer_train"
    train_loader = make_loader(protocol, ROOT / "splits" / f"fold{fold}" / f"{train_name}.txt",
                               augment=True, batch=batch_size, workers=workers, shuffle=True,
                               seed=protocol["seed"], pin_memory=device.type == "cuda")
    val_loader = None
    if stage == "inner_select":
        val_loader = make_loader(protocol, ROOT / "splits" / f"fold{fold}" / "inner_holdout.txt",
                                 augment=False, batch=batch_size, workers=workers, shuffle=False,
                                 seed=protocol["seed"], pin_memory=device.type == "cuda")
    model, criterion = model_and_loss(protocol, device, "q_only")
    q_params = model.model[-1].v6a2_quality_heads.state_dict()
    q_initial_hash = _parameter_hash(q_params)
    if run_id == "outer_refit_empty_gt_fix_v2":
        inner = json.loads((ROOT / "runs/fold1/inner_select/result.json").read_text())
        if q_initial_hash != inner["q_initial_hash"]:
            raise RuntimeError("Fold1 retry did not restart from the original Q initialization")
    optimizer = torch.optim.AdamW(model.model[-1].v6a2_quality_heads.parameters(), lr=1e-4, weight_decay=5e-4)
    max_epochs = 20 if stage == "inner_select" else selected_epoch
    best_val = float("inf")
    best_epoch = None
    try:
        for epoch in range(1, max_epochs + 1):
            start = time.monotonic()
            train_loss, train_images, sample_audits = _epoch_loop(protocol, model, criterion, train_loader,
                                                                   optimizer, device, epoch)
            if stage == "inner_select":
                val_loss, val_images = _q_validation(protocol, model, criterion, val_loader, device)
                if val_loss < best_val:
                    best_val, best_epoch = val_loss, epoch
                    _save_checkpoint(folder / "best.pt", model, epoch, protocol_path,
                                     stage, fold, q_initial_hash)
            else:
                val_loss, val_images = None, 0
            row = {"epoch": epoch, "train_q_loss": train_loss, "train_images": train_images,
                   "inner_holdout_q_loss": val_loss, "inner_holdout_images": val_images,
                   "elapsed_seconds": time.monotonic() - start, "sample_audits": sample_audits}
            frozen["epochs"].append(row)
            frozen["last_epoch"] = epoch
            (folder / "status.json").write_text(json.dumps(frozen, indent=2) + "\n")
            print(json.dumps({"fold": fold, "phase": stage, **{k: row[k] for k in
                               ("epoch", "train_q_loss", "inner_holdout_q_loss", "elapsed_seconds")}}), flush=True)
        if stage == "outer_refit":
            _save_checkpoint(folder / "final.pt", model, max_epochs, protocol_path,
                             stage, fold, q_initial_hash)
        frozen.update({"status": "PASS", "selected_epoch": best_epoch if stage == "inner_select" else selected_epoch,
                       "selected_inner_q_loss": best_val if stage == "inner_select" else None,
                       "q_initial_hash": q_initial_hash,
                       "checkpoint": str(folder / ("best.pt" if stage == "inner_select" else "final.pt"))})
        (folder / "result.json").write_text(json.dumps(frozen, indent=2) + "\n")
        return {k: frozen[k] for k in ("status", "phase", "outer_fold", "selected_epoch", "checkpoint")}
    except BaseException as exc:
        frozen.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        (folder / "status.json").write_text(json.dumps(frozen, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpu_dataset", "benchmark100", "inner_select", "outer_refit"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.stage == "cpu_dataset":
        output = cpu_dataset_gate(protocol)
    elif args.stage == "benchmark100":
        output = benchmark(protocol, args.protocol, torch.device(args.device), 100, args.batch, args.workers)
    else:
        if args.fold is None:
            raise ValueError("--fold is required for fold training")
        output = _training_phase(protocol, args.protocol, args.stage, args.fold,
                                 torch.device(args.device), args.batch, args.workers,
                                 args.run_id)
    print(json.dumps(output, indent=2))
