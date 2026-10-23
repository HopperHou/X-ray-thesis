#!/usr/bin/env python3
"""Independently verify configuration and, optionally, materialized YOLO data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
CSV_ROOT = Path("/hpc/zhou228/x-ray/Dataset/Simplified_dataset")
CSV_FILES = {
    "train": CSV_ROOT / "new_train.csv",
    "validation": CSV_ROOT / "new_validation.csv",
    "test": CSV_ROOT / "new_test.csv",
}
ALL_DATA = CSV_ROOT / "All_data.csv"
ASSIGNMENTS = CSV_ROOT / "patient_level_split_image_assignments.csv"
EXPECTED_HASHES = {
    "All_data.csv": "a44d2e86b847da005a6b4ede1135606fa79f2aa0d08bd0652a377e2713c75b50",
    "new_train.csv": "c09b6b7862ea259031a478399adecaeeeae82a8d878baee5a4a77ee59fcecdaf",
    "new_validation.csv": "d53e83e80132a236084a433a25a455a7b94fa3c7e7152e2a92139fe7517711e8",
    "new_test.csv": "e5ebdc082d874cc5e9e6550b5d4cf36aae2aa6781c8cc49e5d0845c34a4935fb",
    "patient_level_split_image_assignments.csv": "4ddb0f76e9a5f2bb81be1a29a96e4d4d968e1ba277a576d7e20686b0fcabf4ac",
}
EXPECTED_IMAGES = {"train": 14934, "validation": 1660, "test": 1844}
EXPECTED_BOXES = {"train": 22477, "validation": 2502, "test": 2770}
EXPECTED_NEGATIVES = {"train": 2734, "validation": 304, "test": 337}
FIELDS = [
    "id", "type_of_abnormalities", "x_min", "y_min", "x_max", "y_max",
    "source", "width", "height",
]
CLASS_TO_ID = {
    "Pneumonia": 0, "Pneumothorax": 1, "Cardiomegaly": 2,
    "Aortic enlargement": 3, "Pleural thickening": 4, "Pulmonary fibrosis": 5,
}
HEX32 = re.compile(r"^[0-9a-f]{32}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != FIELDS:
            raise RuntimeError(f"Unexpected columns: {path}")
        return list(reader)


def anonymous(source: str, image_id: str) -> str:
    return hashlib.sha256(f"{source}\0{image_id}".encode()).hexdigest()[:32]


def expected_label(rows: list[dict[str, str]]) -> str:
    width, height = int(rows[0]["width"]), int(rows[0]["height"])
    output = []
    for row in rows:
        if row["type_of_abnormalities"] == "no_finding":
            continue
        x1, y1, x2, y2 = map(float, (row["x_min"], row["y_min"], row["x_max"], row["y_max"]))
        output.append(
            f"{CLASS_TO_ID[row['type_of_abnormalities']]} "
            f"{((x1+x2)/2/width):.8f} {((y1+y2)/2/height):.8f} "
            f"{((x2-x1)/width):.8f} {((y2-y1)/height):.8f}"
        )
    return "\n".join(output) + ("\n" if output else "")


def main() -> None:
    args = parse_args()
    paths = {ALL_DATA.name: ALL_DATA, ASSIGNMENTS.name: ASSIGNMENTS}
    paths.update({path.name: path for path in CSV_FILES.values()})
    for name, path in paths.items():
        if sha256(path) != EXPECTED_HASHES[name]:
            raise RuntimeError(f"Input hash differs: {path}")

    all_rows = read(ALL_DATA)
    rows_by_partition = {partition: read(path) for partition, path in CSV_FILES.items()}
    signature = lambda row: tuple(row[field] for field in FIELDS)
    if Counter(map(signature, all_rows)) != Counter(
        signature(row) for rows in rows_by_partition.values() for row in rows
    ):
        raise RuntimeError("Split rows do not exactly reproduce All_data")

    grouped = {}
    identities = {}
    for partition, rows in rows_by_partition.items():
        part = defaultdict(list)
        for row in rows:
            part[(row["source"], row["id"])].append(row)
        grouped[partition] = part
        identities[partition] = set(part)
        boxes = sum(row["type_of_abnormalities"] != "no_finding" for row in rows)
        negatives = sum(
            {row["type_of_abnormalities"] for row in image_rows} == {"no_finding"}
            for image_rows in part.values()
        )
        if (len(part), boxes, negatives) != (
            EXPECTED_IMAGES[partition], EXPECTED_BOXES[partition], EXPECTED_NEGATIVES[partition]
        ):
            raise RuntimeError(f"Wrong {partition} totals")
    names = list(CSV_FILES)
    for index, left in enumerate(names):
        for right in names[index+1:]:
            if identities[left] & identities[right]:
                raise RuntimeError(f"Image leakage between {left} and {right}")

    csv_partition = {
        identity: partition for partition, items in identities.items() for identity in items
    }
    patient_partition = {}
    assignment_partition = {}
    with ASSIGNMENTS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            identity = (row["source"], row["id"])
            assignment_partition[identity] = row["partition"]
            previous = patient_partition.setdefault(row["patient_group"], row["partition"])
            if previous != row["partition"]:
                raise RuntimeError(f"Patient leakage: {row['patient_group']}")
    if assignment_partition != csv_partition:
        raise RuntimeError("Patient assignment and split CSVs differ")

    config = yaml.safe_load((ROOT / "data.yaml").read_text())
    expected_config = {
        "path": str(ROOT), "train": "train.txt", "val": "validation.txt", "test": "test.txt",
        "nc": 6, "names": {value: name for name, value in CLASS_TO_ID.items()},
    }
    if config != expected_config:
        raise RuntimeError(f"data.yaml differs: {config}")

    if args.materialized:
        manifest_path = ROOT / "dataset_manifest.csv"
        if not manifest_path.is_file():
            raise RuntimeError("Materialized manifest missing")
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            manifest = list(csv.DictReader(handle))
        if len(manifest) != sum(EXPECTED_IMAGES.values()):
            raise RuntimeError("Manifest image count differs")
        seen = set()
        expected_files = {p: {"images": set(), "labels": set()} for p in CSV_FILES}
        for row in manifest:
            partition = row["partition"]
            identity = (row["source"], row["id"])
            if identity not in grouped[partition] or identity in seen:
                raise RuntimeError(f"Unexpected/duplicate manifest identity: {identity}")
            seen.add(identity)
            key = anonymous(*identity)
            if row["anonymous_key"] != key or not HEX32.fullmatch(key):
                raise RuntimeError(f"Invalid anonymous key: {identity}")
            image_path, label_path = Path(row["image_path"]), Path(row["label_path"])
            if not image_path.is_symlink() or image_path.resolve() != Path(row["original_path"]).resolve():
                raise RuntimeError(f"Invalid source symlink: {image_path}")
            if label_path.read_text(encoding="ascii") != expected_label(grouped[partition][identity]):
                raise RuntimeError(f"Label mismatch: {identity}")
            with Image.open(image_path) as image:
                if image.size != (int(row["width"]), int(row["height"])):
                    raise RuntimeError(f"Dimension mismatch: {identity}")
                image.verify()
            expected_files[partition]["images"].add(image_path.name)
            expected_files[partition]["labels"].add(label_path.name)
        if seen != set(csv_partition):
            raise RuntimeError("Manifest does not cover all split images")
        for partition in CSV_FILES:
            image_dir, label_dir = ROOT / "images" / partition, ROOT / "labels" / partition
            if {p.name for p in image_dir.iterdir()} != expected_files[partition]["images"]:
                raise RuntimeError(f"Unexpected image file set: {partition}")
            if {p.name for p in label_dir.iterdir()} != expected_files[partition]["labels"]:
                raise RuntimeError(f"Unexpected label file set: {partition}")
            listed = (ROOT / f"{partition}.txt").read_text().splitlines()
            if {Path(item).name for item in listed} != expected_files[partition]["images"]:
                raise RuntimeError(f"Split list differs: {partition}")

    print(f"all_images={len(csv_partition)} patient_groups={len(patient_partition)}")
    print("image_overlap=0 patient_group_overlap=0 exact_All_data_row_union=1")
    print("CONFIG_AND_SPLIT_CHECKS_PASS=1")
    print(f"MATERIALIZED_CHECKED={int(args.materialized)}")


if __name__ == "__main__":
    main()
