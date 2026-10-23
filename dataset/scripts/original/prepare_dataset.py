#!/usr/bin/env python3
"""Validate and optionally materialize the current patient-level YOLO split.

Without --materialize this script is read-only.  Materialized image files are
anonymous symlinks; labels are derived directly from the reviewed CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
CSV_ROOT = Path("/hpc/zhou228/x-ray/Dataset/Simplified_dataset")
ALL_DATA = CSV_ROOT / "All_data.csv"
SPLIT_CSVS = {
    "train": CSV_ROOT / "new_train.csv",
    "validation": CSV_ROOT / "new_validation.csv",
    "test": CSV_ROOT / "new_test.csv",
}
PATIENT_ASSIGNMENTS = CSV_ROOT / "patient_level_split_image_assignments.csv"

EXPECTED_HASHES = {
    "All_data.csv": "a44d2e86b847da005a6b4ede1135606fa79f2aa0d08bd0652a377e2713c75b50",
    "new_train.csv": "c09b6b7862ea259031a478399adecaeeeae82a8d878baee5a4a77ee59fcecdaf",
    "new_validation.csv": "d53e83e80132a236084a433a25a455a7b94fa3c7e7152e2a92139fe7517711e8",
    "new_test.csv": "e5ebdc082d874cc5e9e6550b5d4cf36aae2aa6781c8cc49e5d0845c34a4935fb",
    "patient_level_split_image_assignments.csv": "4ddb0f76e9a5f2bb81be1a29a96e4d4d968e1ba277a576d7e20686b0fcabf4ac",
}
EXPECTED = {
    "train": {"rows": 25211, "images": 14934, "boxes": 22477, "negative_images": 2734},
    "validation": {"rows": 2806, "images": 1660, "boxes": 2502, "negative_images": 304},
    "test": {"rows": 3107, "images": 1844, "boxes": 2770, "negative_images": 337},
}
EXPECTED_ALL = {"rows": 31124, "images": 18438, "boxes": 27749, "negative_images": 3375}
EXPECTED_COLUMNS = [
    "id", "type_of_abnormalities", "x_min", "y_min", "x_max", "y_max",
    "source", "width", "height",
]
BOX_COLUMNS = ("x_min", "y_min", "x_max", "y_max")
CLASS_TO_ID = {
    "Pneumonia": 0,
    "Pneumothorax": 1,
    "Cardiomegaly": 2,
    "Aortic enlargement": 3,
    "Pleural thickening": 4,
    "Pulmonary fibrosis": 5,
}
NO_FINDING = "no_finding"
SOURCE_DIRS = {
    "MS-CXR": (Path("/hpc/zhou228/x-ray/Dataset/MS-CXR/train"),),
    "RSNA": (Path("/hpc/zhou228/x-ray/Dataset/RSNA/train"),),
    # Consolidated overlap rows retain the RSNA UUID and therefore use RSNA pixels.
    "RSNA/ChestX-Det": (Path("/hpc/zhou228/x-ray/Dataset/RSNA/train"),),
    "ChestX-Det": (Path("/hpc/zhou228/x-ray/Dataset/ChestX-Det/image"),),
    "SIIM-ACR": (
        Path("/hpc/zhou228/x-ray/Trials/trail-SIIM/Dataset/images/train"),
        Path("/hpc/zhou228/x-ray/Trials/trail-SIIM/Dataset/images/val"),
    ),
    "VinDR-CXR": (
        Path("/hpc/zhou228/x-ray/Dataset/VinDR-CXR/images/train"),
        Path("/hpc/zhou228/x-ray/Dataset/VinDR-CXR/images/test"),
    ),
}


@dataclass(frozen=True)
class Sample:
    partition: str
    source: str
    image_id: str
    width: int
    height: int
    rows: tuple[dict[str, str], ...]
    image_path: Path
    anonymous_key: str

    @property
    def negative(self) -> bool:
        return self.rows[0]["type_of_abnormalities"] == NO_FINDING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != EXPECTED_COLUMNS:
            raise RuntimeError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def row_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in EXPECTED_COLUMNS)


def anonymous_key(source: str, image_id: str) -> str:
    return hashlib.sha256(f"{source}\0{image_id}".encode()).hexdigest()[:32]


def build_source_index(required: dict[str, set[str]]) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for source, ids in required.items():
        for directory in SOURCE_DIRS[source]:
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=True):
                        continue
                    stem = Path(entry.name).stem
                    if stem not in ids:
                        continue
                    key = (source, stem)
                    path = Path(entry.path)
                    previous = result.setdefault(key, path)
                    if previous != path:
                        raise RuntimeError(f"Multiple source images found for {key}: {previous}, {path}")
        missing = ids - {image_id for src, image_id in result if src == source}
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} {source} images; first={sorted(missing)[:5]}")
    return result


def validate_group(identity: tuple[str, str], rows: list[dict[str, str]]) -> tuple[int, int, bool]:
    dimensions = {(int(row["width"]), int(row["height"])) for row in rows}
    if len(dimensions) != 1:
        raise RuntimeError(f"Conflicting dimensions: {identity}")
    width, height = next(iter(dimensions))
    labels = {row["type_of_abnormalities"] for row in rows}
    unknown = labels - set(CLASS_TO_ID) - {NO_FINDING}
    if unknown:
        raise RuntimeError(f"Unknown labels for {identity}: {unknown}")
    negative = labels == {NO_FINDING}
    if NO_FINDING in labels and not negative:
        raise RuntimeError(f"Mixed positive/no_finding rows: {identity}")
    if negative and len(rows) != 1:
        raise RuntimeError(f"Negative image does not have exactly one row: {identity}")
    exact_boxes = set()
    for row in rows:
        coordinates = [row[column].strip() for column in BOX_COLUMNS]
        if row["type_of_abnormalities"] == NO_FINDING:
            if any(coordinates):
                raise RuntimeError(f"no_finding row has coordinates: {identity}")
            continue
        if not all(coordinates):
            raise RuntimeError(f"Positive row lacks coordinates: {identity}")
        x1, y1, x2, y2 = map(float, coordinates)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise RuntimeError(f"Invalid box for {identity}: {coordinates}")
        box = (row["type_of_abnormalities"], x1, y1, x2, y2)
        if box in exact_boxes:
            raise RuntimeError(f"Duplicate class/box row: {identity} {box}")
        exact_boxes.add(box)
    return width, height, negative


def load_samples() -> tuple[list[Sample], dict[str, object]]:
    hash_inputs = {
        "All_data.csv": ALL_DATA,
        **{path.name: path for path in SPLIT_CSVS.values()},
        PATIENT_ASSIGNMENTS.name: PATIENT_ASSIGNMENTS,
    }
    for name, path in hash_inputs.items():
        actual_hash = sha256(path)
        if actual_hash != EXPECTED_HASHES[name]:
            raise RuntimeError(f"Unreviewed input version: {path} SHA256={actual_hash}")

    all_rows = read_rows(ALL_DATA)
    split_rows = {partition: read_rows(path) for partition, path in SPLIT_CSVS.items()}
    if Counter(map(row_signature, all_rows)) != Counter(
        row_signature(row) for rows in split_rows.values() for row in rows
    ):
        raise RuntimeError("train + validation + test is not the exact All_data row multiset")

    required: dict[str, set[str]] = defaultdict(set)
    grouped_by_partition: dict[str, dict[tuple[str, str], list[dict[str, str]]]] = {}
    identities_by_partition: dict[str, set[tuple[str, str]]] = {}
    summary: dict[str, object] = {}
    for partition, rows in split_rows.items():
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            identity = (row["source"], row["id"])
            grouped[identity].append(row)
            required[row["source"]].add(row["id"])
        grouped_by_partition[partition] = grouped
        identities_by_partition[partition] = set(grouped)
        negatives = 0
        for identity, image_rows in grouped.items():
            _, _, negative = validate_group(identity, image_rows)
            negatives += int(negative)
        actual = {
            "rows": len(rows), "images": len(grouped),
            "boxes": sum(row["type_of_abnormalities"] != NO_FINDING for row in rows),
            "negative_images": negatives,
        }
        if actual != EXPECTED[partition]:
            raise RuntimeError(f"Unexpected {partition} totals: {actual}")
        summary[partition] = actual

    partitions = list(SPLIT_CSVS)
    for index, left in enumerate(partitions):
        for right in partitions[index + 1:]:
            if identities_by_partition[left] & identities_by_partition[right]:
                raise RuntimeError(f"Image overlap between {left} and {right}")

    source_index = build_source_index(required)
    seen_anonymous: dict[str, tuple[str, str]] = {}
    samples: list[Sample] = []
    for partition, grouped in grouped_by_partition.items():
        for identity, image_rows in sorted(grouped.items()):
            width, height, _ = validate_group(identity, image_rows)
            key = anonymous_key(*identity)
            previous = seen_anonymous.setdefault(key, identity)
            if previous != identity:
                raise RuntimeError(f"Anonymous key collision: {previous}, {identity}")
            samples.append(Sample(
                partition, identity[0], identity[1], width, height,
                tuple(image_rows), source_index[identity], key,
            ))

    assignment_partition: dict[tuple[str, str], str] = {}
    patient_partition: dict[str, str] = {}
    with PATIENT_ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            identity = (row["source"], row["id"])
            assignment_partition[identity] = row["partition"]
            previous = patient_partition.setdefault(row["patient_group"], row["partition"])
            if previous != row["partition"]:
                raise RuntimeError(f"Patient leakage: {row['patient_group']}")
    sample_partitions = {(sample.source, sample.image_id): sample.partition for sample in samples}
    if sample_partitions != assignment_partition:
        raise RuntimeError("CSV partitions differ from reviewed patient-level assignments")
    summary["all_data"] = EXPECTED_ALL
    summary["patient_groups"] = len(patient_partition)
    summary["source_images"] = dict(sorted(Counter(s.source for s in samples).items()))
    return samples, summary


def yolo_text(sample: Sample) -> str:
    output = []
    for row in sample.rows:
        label = row["type_of_abnormalities"]
        if label == NO_FINDING:
            continue
        x1, y1, x2, y2 = (float(row[column]) for column in BOX_COLUMNS)
        output.append(
            f"{CLASS_TO_ID[label]} "
            f"{((x1 + x2) / 2 / sample.width):.8f} "
            f"{((y1 + y2) / 2 / sample.height):.8f} "
            f"{((x2 - x1) / sample.width):.8f} "
            f"{((y2 - y1) / sample.height):.8f}"
        )
    return "\n".join(output) + ("\n" if output else "")


def check_image(sample: Sample) -> None:
    with Image.open(sample.image_path) as image:
        if image.size != (sample.width, sample.height):
            raise RuntimeError(
                f"Dimension mismatch for {sample.source}/{sample.image_id}: "
                f"{image.size} != {(sample.width, sample.height)}"
            )
        image.verify()


def materialize(samples: list[Sample], workers: int) -> None:
    for partition in SPLIT_CSVS:
        (ROOT / "images" / partition).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / partition).mkdir(parents=True, exist_ok=True)

    def one(sample: Sample) -> dict[str, object]:
        check_image(sample)
        image_output = ROOT / "images" / sample.partition / (
            sample.anonymous_key + sample.image_path.suffix.lower()
        )
        label_output = ROOT / "labels" / sample.partition / f"{sample.anonymous_key}.txt"
        if image_output.is_symlink():
            if image_output.resolve() != sample.image_path.resolve():
                raise RuntimeError(f"Wrong existing symlink: {image_output}")
        elif image_output.exists():
            raise RuntimeError(f"Expected symlink, found regular file: {image_output}")
        else:
            image_output.symlink_to(sample.image_path)
        label_output.write_text(yolo_text(sample), encoding="ascii")
        return {
            "partition": sample.partition,
            "anonymous_key": sample.anonymous_key,
            "source": sample.source,
            "id": sample.image_id,
            "image_path": str(image_output),
            "label_path": str(label_output),
            "original_path": str(sample.image_path),
            "width": sample.width,
            "height": sample.height,
            "boxes": len(yolo_text(sample).splitlines()),
            "negative": int(sample.negative),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        manifest = list(executor.map(one, samples))
    manifest.sort(key=lambda row: (str(row["partition"]), str(row["anonymous_key"])))
    fields = [
        "partition", "anonymous_key", "source", "id", "image_path", "label_path",
        "original_path", "width", "height", "boxes", "negative",
    ]
    with (ROOT / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    for partition in SPLIT_CSVS:
        paths = [str(row["image_path"]) for row in manifest if row["partition"] == partition]
        (ROOT / f"{partition}.txt").write_text("\n".join(paths) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    samples, summary = load_samples()
    if args.check_images and not args.materialize:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(check_image, samples))
    if args.materialize:
        materialize(samples, args.workers)
        summary["materialized"] = True
    else:
        summary["materialized"] = False
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("CURRENT_SPLIT_VALIDATION_PASS=1")
    print(f"MATERIALIZED={int(args.materialize)}")


if __name__ == "__main__":
    main()
