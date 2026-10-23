#!/usr/bin/env python3
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom


DEFAULT_DATASET_ROOT = Path("/hpc/zhou228/x-ray/Dataset/SIIM-ACR")
DEFAULT_OUTPUT_ROOT = Path("/hpc/zhou228/x-ray/trail-SIIM")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SIIM-ACR DICOM files to PNG and YOLO labels."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per split.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def dicom_to_uint8(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    low, high = np.percentile(arr, (0.5, 99.5))
    if high <= low:
        low, high = float(arr.min()), float(arr.max())
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip((arr - low) / (high - low), 0, 1)
    return (arr * 255).astype(np.uint8)


def build_dicom_index(split_dir):
    index = {}
    for path in split_dir.rglob("*.dcm"):
        index[path.stem] = path
    return index


def convert_one(item):
    image_id, split, dicom_path, png_path, force = item
    if png_path.exists() and not force:
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Existing PNG is unreadable: {png_path}")
        height, width = img.shape[:2]
        return image_id, split, width, height, str(dicom_path), str(png_path), "kept"

    ds = pydicom.dcmread(str(dicom_path))
    image = dicom_to_uint8(ds)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(png_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {png_path}")
    height, width = image.shape[:2]
    return image_id, split, width, height, str(dicom_path), str(png_path), "written"


def rows_to_labels(rows, width, height):
    labels = []
    for row in rows:
        disease = str(row.get("disease", ""))
        if disease == "no_findings" or pd.isna(row.get("x_min")):
            continue

        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])
        x_min = max(0.0, min(x_min, width))
        x_max = max(0.0, min(x_max, width))
        y_min = max(0.0, min(y_min, height))
        y_max = max(0.0, min(y_max, height))
        box_w = x_max - x_min
        box_h = y_max - y_min
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        labels.append(f"0 {x_center:.8f} {y_center:.8f} {box_w / width:.8f} {box_h / height:.8f}")
    return labels


def load_annotations(csv_path):
    df = pd.read_csv(csv_path, dtype={"ID": str})
    return {image_id: group.to_dict("records") for image_id, group in df.groupby("ID", sort=False)}


def load_annotation_frame(csv_path, split):
    df = pd.read_csv(csv_path, dtype={"ID": str})
    df.insert(0, "split", split)
    df.insert(1, "source_csv", str(csv_path))
    return df


def write_data_yaml(dataset_dir):
    data_yaml = dataset_dir / "siim_yolo.yaml"
    text = "\n".join(
        [
            f"path: {dataset_dir}",
            "train: images/train",
            "val: images/val",
            "test: images/val",
            "",
            "names:",
            "  0: pneumothorax",
            "",
        ]
    )
    data_yaml.write_text(text)
    return data_yaml


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    output_root = args.output_root
    yolo_root = output_root / "Dataset"

    train_csv = args.train_csv or dataset_root / "train-rle-box.csv"
    val_csv = args.val_csv or dataset_root / "test-rle-box.csv"
    split_specs = {
        "train": (dataset_root / "dicom-images-train", train_csv),
        "val": (dataset_root / "dicom-images-test", val_csv),
    }

    all_items = []
    annotations = {}
    annotation_frames = []
    for split, (dicom_dir, csv_path) in split_specs.items():
        dicom_index = build_dicom_index(dicom_dir)
        annotations[split] = load_annotations(csv_path)
        annotation_frames.append(load_annotation_frame(csv_path, split))
        image_ids = list(annotations[split].keys())
        if args.limit is not None:
            image_ids = image_ids[: args.limit]

        missing = [image_id for image_id in image_ids if image_id not in dicom_index]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"{len(missing)} {split} DICOM files missing. Examples: {preview}")

        for image_id in image_ids:
            all_items.append(
                (
                    image_id,
                    split,
                    dicom_index[image_id],
                    yolo_root / "images" / split / f"{image_id}.png",
                    args.force,
                )
            )

    yolo_root.mkdir(parents=True, exist_ok=True)
    manifest_path = yolo_root / "conversion_manifest.csv"
    dims = {}
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(convert_one, all_items))

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "split", "width", "height", "dicom_path", "png_path", "status"])
        writer.writerows(results)

    for image_id, split, width, height, *_ in results:
        dims[(split, image_id)] = (width, height)

    manifest_df = pd.read_csv(manifest_path)
    labels_csv = yolo_root / "labels.csv"
    combined_labels = pd.concat(annotation_frames, ignore_index=True)
    combined_labels = combined_labels.merge(
        manifest_df[["image_id", "split", "width", "height", "png_path"]],
        left_on=["ID", "split"],
        right_on=["image_id", "split"],
        how="left",
    )
    combined_labels["label_path"] = combined_labels.apply(
        lambda row: str(yolo_root / "labels" / row["split"] / f"{row['ID']}.txt"),
        axis=1,
    )
    combined_labels.drop(columns=["image_id"], inplace=True)
    combined_labels.to_csv(labels_csv, index=False)

    label_count = 0
    image_count = 0
    positive_count = 0
    for split in split_specs:
        label_dir = yolo_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id, rows in annotations[split].items():
            if args.limit is not None and (split, image_id) not in dims:
                continue
            width, height = dims[(split, image_id)]
            labels = rows_to_labels(rows, width, height)
            (label_dir / f"{image_id}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
            image_count += 1
            label_count += len(labels)
            positive_count += int(bool(labels))

    data_yaml = write_data_yaml(yolo_root)
    print(f"Prepared YOLO dataset: {yolo_root}")
    print(f"Images: {image_count}, positive images: {positive_count}, boxes: {label_count}")
    print(f"Combined label CSV: {labels_csv}")
    print(f"Manifest: {manifest_path}")
    print(f"Data YAML: {data_yaml}")


if __name__ == "__main__":
    main()
