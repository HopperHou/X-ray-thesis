#!/usr/bin/env python3
"""Build six-class simplified train.csv or test.csv from source split tables."""

from __future__ import annotations

import csv
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pydicom
from PIL import Image


DATASET_ROOT = Path("/hpc/zhou228/x-ray/Dataset")
OUTPUT_DIR = DATASET_ROOT / "Simplified_dataset"

FIELDNAMES = [
    "id",
    "type_of_abnormalities",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "source",
    "width",
    "height",
]

SELECTED = {
    "Pneumonia",
    "Pneumothorax",
    "Cardiomegaly",
    "Aortic enlargement",
    "Pleural thickening",
    "Pulmonary fibrosis",
}

CANONICAL = {name.lower(): name for name in SELECTED}

SOURCE_FILES = {
    "train": {
        "MS-CXR": [DATASET_ROOT / "MS-CXR" / "train-rle-box-ms-cxr.csv"],
        "RSNA": [DATASET_ROOT / "RSNA" / "train-rle-box-rsnna.csv"],
        "SIIM-ACR": [DATASET_ROOT / "SIIM-ACR" / "train-rle-box-siim-acr.csv"],
        "VinDR-CXR": [DATASET_ROOT / "VinDR-CXR" / "train-rle-box-vindr-cxr.csv"],
    },
    "test": {
        "MS-CXR": [DATASET_ROOT / "MS-CXR" / "test-rle-box-ms-cxr.csv"],
        "RSNA": [DATASET_ROOT / "RSNA" / "test-rle-box-rsnna.csv"],
        "SIIM-ACR": [DATASET_ROOT / "SIIM-ACR" / "test-rle-box-siim-acr.csv"],
        "VinDR-CXR": [DATASET_ROOT / "VinDR-CXR" / "test-rle-box-vindr-cxr.csv"],
    },
}

EXPECTED = {
    "train": {
        "positive_rows": 25600,
        "positive_images": 11469,
        "negative_images": 2294,
        "total_rows": 27894,
        "removed_multilabel": 81,
    },
    "test": {
        "positive_rows": 3816,
        "positive_images": 2492,
        "negative_images": 498,
        "total_rows": 4314,
        "removed_multilabel": 0,
    },
}

NEGATIVE_SOURCES = ("RSNA", "SIIM-ACR", "VinDR-CXR")
NEGATIVE_RATIO = 0.20
NEGATIVE_SEED = "simplified-dataset-no-finding-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def selected_atoms(disease: str) -> set[str]:
    atoms = {part.strip() for part in disease.split("|") if part.strip()}
    return {CANONICAL[atom.lower()] for atom in atoms if atom.lower() in CANONICAL}


def build_ms_dimensions() -> dict[str, tuple[int, int]]:
    rows = read_csv(DATASET_ROOT / "MS-CXR" / "MS_CXR_Local_Alignment_v1.0.0.csv")
    dimensions: dict[str, tuple[int, int]] = {}
    for row in rows:
        value = (int(row["image_width"]), int(row["image_height"]))
        previous = dimensions.setdefault(row["dicom_id"], value)
        if previous != value:
            raise ValueError(f"Conflicting MS-CXR dimensions for {row['dicom_id']}")
    return dimensions


def index_siim_dicoms(split: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    root = DATASET_ROOT / "SIIM-ACR"
    directory = root / f"dicom-images-{split}"
    for path in directory.rglob("*.dcm"):
        if path.stem in paths:
            raise ValueError(f"Duplicate SIIM-ACR DICOM ID: {path.stem}")
        paths[path.stem] = path
    return paths


def dicom_dimensions(path: Path) -> tuple[int, int]:
    dataset = pydicom.dcmread(
        str(path), stop_before_pixels=True, specific_tags=["Rows", "Columns"]
    )
    return int(dataset.Columns), int(dataset.Rows)


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def locate_vindr_jpg(image_id: str) -> Path:
    root = DATASET_ROOT / "VinDR-CXR" / "images"
    train = root / "train" / f"{image_id}.jpg"
    test = root / "test" / f"{image_id}.jpg"
    matches = [path for path in (train, test) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one VinDR-CXR JPG for {image_id}, found {len(matches)}"
        )
    return matches[0]


def get_dimensions(
    source: str,
    image_id: str,
    ms_dimensions: dict[str, tuple[int, int]],
    siim_paths: dict[str, Path],
    dimension_cache: dict[tuple[str, str], tuple[int, int]],
) -> tuple[int, int]:
    cache_key = (source, image_id)
    if cache_key in dimension_cache:
        return dimension_cache[cache_key]
    if source == "MS-CXR":
        if image_id not in ms_dimensions:
            raise KeyError(f"Missing MS-CXR dimensions for {image_id}")
        dimensions = ms_dimensions[image_id]
    elif source == "RSNA":
        dcm = DATASET_ROOT / "RSNA" / "train-dcm" / f"{image_id}.dcm"
        if not dcm.is_file():
            raise FileNotFoundError(dcm)
        dimensions = dicom_dimensions(dcm)
    elif source == "SIIM-ACR":
        if image_id not in siim_paths:
            raise FileNotFoundError(f"Missing SIIM-ACR DICOM for {image_id}")
        dimensions = dicom_dimensions(siim_paths[image_id])
    elif source == "VinDR-CXR":
        dimensions = image_dimensions(locate_vindr_jpg(image_id))
    else:
        raise ValueError(source)
    dimension_cache[cache_key] = dimensions
    return dimensions


def is_no_finding(value: str) -> bool:
    normalized = value.strip().lower().replace(" ", "_")
    return normalized in {"no_finding", "no_findings"}


def deterministic_negative_sample(
    split: str,
    candidates: dict[str, set[str]],
    total: int,
) -> dict[str, list[str]]:
    base, remainder = divmod(total, len(NEGATIVE_SOURCES))
    selected: dict[str, list[str]] = {}
    for index, source in enumerate(NEGATIVE_SOURCES):
        count = base + int(index < remainder)
        ranked = sorted(
            candidates[source],
            key=lambda image_id: hashlib.sha256(
                f"{NEGATIVE_SEED}|{split}|{source}|{image_id}".encode("utf-8")
            ).digest(),
        )
        if len(ranked) < count:
            raise ValueError(
                f"Not enough {source} no-finding candidates for {split}: "
                f"need {count}, have {len(ranked)}"
            )
        selected[source] = ranked[:count]
    return selected


def main() -> None:
    args = parse_args()
    output_path = OUTPUT_DIR / f"{args.split}.csv"
    ms_dimensions = build_ms_dimensions()
    siim_paths = index_siim_dicoms(args.split)
    dimension_cache: dict[tuple[str, str], tuple[int, int]] = {}

    output_rows: list[dict[str, object]] = []
    removed_multilabel: list[tuple[str, tuple[str, ...]]] = []
    negative_candidates = {source: set() for source in NEGATIVE_SOURCES}

    for source, files in SOURCE_FILES[args.split].items():
        for path in files:
            for row in read_csv(path):
                y_max = row.get("y_max") or row.get("Iy_max") or ""
                coordinates = [row.get("x_min", ""), row.get("y_min", ""), row.get("x_max", ""), y_max]
                if source in negative_candidates and is_no_finding(row.get("disease", "")):
                    if any(str(value).strip() for value in coordinates):
                        raise ValueError(f"No-finding row unexpectedly has coordinates: {row}")
                    negative_candidates[source].add(row["ID"])
                if not all(str(value).strip() for value in coordinates):
                    continue

                selected = selected_atoms(row.get("disease", ""))
                if not selected:
                    continue
                if len(selected) >= 2:
                    if source != "VinDR-CXR":
                        raise ValueError(f"Unexpected multi-label box outside VinDR-CXR: {row}")
                    removed_multilabel.append((row["ID"], tuple(sorted(selected))))
                    continue

                image_id = row["ID"]
                width, height = get_dimensions(
                    source, image_id, ms_dimensions, siim_paths, dimension_cache
                )
                numeric = [float(value) for value in coordinates]
                x_min, y_min, x_max, y_max_number = numeric
                if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max_number <= height):
                    raise ValueError(
                        f"Out-of-bounds box for {source}/{image_id}: {numeric}, image={width}x{height}"
                    )

                output_rows.append(
                    {
                        "id": image_id,
                        "type_of_abnormalities": next(iter(selected)),
                        "x_min": coordinates[0],
                        "y_min": coordinates[1],
                        "x_max": coordinates[2],
                        "y_max": coordinates[3],
                        "source": source,
                        "width": width,
                        "height": height,
                    }
                )

    expected = EXPECTED[args.split]
    if len(removed_multilabel) != expected["removed_multilabel"]:
        raise ValueError(
            f"Expected to remove {expected['removed_multilabel']} multi-label boxes, "
            f"got {len(removed_multilabel)}"
        )
    if len(output_rows) != expected["positive_rows"]:
        raise ValueError(
            f"Expected {expected['positive_rows']:,} positive rows, got {len(output_rows):,}"
        )

    positive_ids = {(str(row["source"]), str(row["id"])) for row in output_rows}
    if len(positive_ids) != expected["positive_images"]:
        raise ValueError(
            f"Expected {expected['positive_images']:,} positive images, got {len(positive_ids):,}"
        )
    calculated_negative_total = round(len(positive_ids) * NEGATIVE_RATIO)
    if calculated_negative_total != expected["negative_images"]:
        raise ValueError(
            f"Expected {expected['negative_images']:,} negatives from the 20% rule, "
            f"got {calculated_negative_total:,}"
        )
    negative_sample = deterministic_negative_sample(
        args.split, negative_candidates, calculated_negative_total
    )
    for source in NEGATIVE_SOURCES:
        for image_id in negative_sample[source]:
            if (source, image_id) in positive_ids:
                raise ValueError(f"Negative image is also positive: {source}/{image_id}")
            width, height = get_dimensions(
                source, image_id, ms_dimensions, siim_paths, dimension_cache
            )
            output_rows.append(
                {
                    "id": image_id,
                    "type_of_abnormalities": "no_finding",
                    "x_min": "",
                    "y_min": "",
                    "x_max": "",
                    "y_max": "",
                    "source": source,
                    "width": width,
                    "height": height,
                }
            )

    if len(output_rows) != expected["total_rows"]:
        raise ValueError(
            f"Expected {expected['total_rows']:,} total rows, got {len(output_rows):,}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    class_counts = Counter(str(row["type_of_abnormalities"]) for row in output_rows)
    source_counts = Counter(str(row["source"]) for row in output_rows)
    unique_images = {(str(row["source"]), str(row["id"])) for row in output_rows}
    print(f"split={args.split}")
    print(f"output={output_path}")
    print(f"rows={len(output_rows)}")
    print(f"unique_source_qualified_images={len(unique_images)}")
    print(f"negative_ratio_to_positive_images={NEGATIVE_RATIO:.2f}")
    print(f"negative_counts={dict(sorted((s, len(v)) for s, v in negative_sample.items()))}")
    print(f"removed_multilabel_boxes={len(removed_multilabel)}")
    print(f"class_counts={dict(sorted(class_counts.items()))}")
    print(f"source_counts={dict(sorted(source_counts.items()))}")


if __name__ == "__main__":
    main()
