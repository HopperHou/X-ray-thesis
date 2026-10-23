#!/usr/bin/env python3
"""Read-only, standard-library verification of the released frozen metadata.

This release verifier was added on 2026-10-23; it is not a historical training
script. It reads neither image files nor medical-image pixel data.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["id", "type_of_abnormalities", "x_min", "y_min", "x_max", "y_max", "source", "width", "height"]
CLASSES = {"Pneumonia", "Pneumothorax", "Cardiomegaly", "Aortic enlargement", "Pleural thickening", "Pulmonary fibrosis"}
EXPECTED = {"train": (14934, 22477, 2734), "validation": (1660, 2502, 304), "test": (1844, 2770, 337)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def identity(row):
    return row["source"], row["id"]


def multiset(rows):
    return Counter(tuple(row[k] for k in FIELDS) for row in rows)


def verify(root):
    root = root.resolve()
    checksum_files = set()
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = root / name
        require(root in path.resolve().parents, f"Checksum path outside release: {name}")
        require(name not in checksum_files, f"Repeated checksum path: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"SHA256 mismatch: {name}")
        checksum_files.add(name)

    # The historical preparation script is parsed, never imported or executed.
    parsed = ast.parse((root / "scripts/original/prepare_dataset.py").read_text(encoding="utf-8"))
    original_hashes = next(ast.literal_eval(node.value) for node in parsed.body
                           if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXPECTED_HASHES" for t in node.targets))
    for name, expected in original_hashes.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f"Historical input hash mismatch: {name}")

    all_rows = read_csv(root / "All_data.csv")
    require(list(all_rows[0]) == FIELDS, "Unexpected annotation schema")
    partitions = {p: read_csv(root / f"new_{p}.csv") for p in EXPECTED}
    require(multiset(all_rows) == multiset([r for rows in partitions.values() for r in rows]), "Split row union differs from All_data.csv")
    expected_partition = {}
    image_groups = defaultdict(list)
    counts = {}
    for partition, rows in partitions.items():
        require(rows and list(rows[0]) == FIELDS, f"Unexpected schema: {partition}")
        images = {identity(r) for r in rows}
        boxes = sum(r["type_of_abnormalities"] != "no_finding" for r in rows)
        negatives = len(rows) - boxes
        require((len(images), boxes, negatives) == EXPECTED[partition], f"Unexpected counts: {partition}")
        counts[partition] = {"image_records": len(images), "boxes": boxes, "negative_records": negatives}
        for key in images:
            require(key not in expected_partition, "Image record occurs in multiple partitions")
            expected_partition[key] = partition
        for row in rows:
            image_groups[identity(row)].append(row)

    for key, rows in image_groups.items():
        dimensions = {(int(r["width"]), int(r["height"])) for r in rows}
        require(len(dimensions) == 1, f"Inconsistent dimensions: {key}")
        width, height = next(iter(dimensions))
        require(width > 0 and height > 0, f"Invalid image dimensions: {key}")
        labels = {r["type_of_abnormalities"] for r in rows}
        if "no_finding" in labels:
            require(len(rows) == 1 and labels == {"no_finding"}, f"Mixed or repeated negative: {key}")
            require(not any(rows[0][k].strip() for k in FIELDS[2:6]), f"Negative contains coordinates: {key}")
        else:
            require(labels <= CLASSES, f"Unknown class: {key}")
            seen = set()
            for row in rows:
                box = tuple(float(row[k]) for k in FIELDS[2:6])
                require(all(math.isfinite(v) for v in box), f"Nonfinite box: {key}")
                x1, y1, x2, y2 = box
                require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"Invalid box: {key}")
                item = row["type_of_abnormalities"], box
                require(item not in seen, f"Repeated class/box: {key}")
                seen.add(item)

    assignment_rows = read_csv(root / "patient_level_split_image_assignments.csv")
    assignments = {identity(r): r["partition"] for r in assignment_rows}
    require(len(assignment_rows) == len(assignments) == 18438, "Repeated/missing assignment records")
    require(assignments == expected_partition, "Assignment mapping differs from split CSV files")
    group_partitions = defaultdict(set)
    for row in assignment_rows:
        group_partitions[row["patient_group"]].add(row["partition"])
    require(all(len(v) == 1 for v in group_partitions.values()), "Analysis group crosses partitions")
    require(len(group_partitions) == 10985, "Unexpected number of analysis groups")

    mapping_rows = read_csv(root / "records/patient_group_mapping.csv")
    require(len(mapping_rows) == 18438 and {identity(r): r["patient_group"] for r in mapping_rows} ==
            {identity(r): r["patient_group"] for r in assignment_rows}, "Patient group mapping differs from assignments")
    manifest = read_csv(root / "dataset_manifest.csv")
    require(len(manifest) == 18438 and len({identity(r) for r in manifest}) == 18438, "Repeated/missing manifest records")
    require({identity(r): r["partition"] for r in manifest} == expected_partition, "Training manifest differs from splits")
    for row in manifest:
        key = identity(row)
        annotations = image_groups[key]
        first = annotations[0]
        anonymous = hashlib.sha256(f"{key[0]}\0{key[1]}".encode()).hexdigest()[:32]
        negative = first["type_of_abnormalities"] == "no_finding"
        require(row["anonymous_key"] == anonymous, "Manifest anonymous key differs from original convention")
        require((int(row["width"]), int(row["height"])) == (int(first["width"]), int(first["height"])), "Manifest dimensions disagree")
        require(int(row["boxes"]) == (0 if negative else len(annotations)) and int(row["negative"]) == int(negative), "Manifest box count/status disagrees")

    audit = read_csv(root / "records/vindr_overlap60_merge_audit.csv")
    require(len(audit) == 3009, "Unexpected number of verified enclosing-box groups")
    require(sum(int(r["input_box_count"]) for r in audit) == 6505, "Unexpected enclosing-box input count")
    return {"status": "PASS", "checksum_files": len(checksum_files), "annotation_rows": len(all_rows),
            "image_records": len(image_groups), "analysis_groups": len(group_partitions), "partitions": counts,
            "split_union_equals_final_table": True, "group_partition_overlap": False,
            "manifest_matches_annotations_and_assignments": True, "verified_enclosing_box_groups": len(audit),
            "image_pixels_checked": False, "historical_scripts_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset_root), indent=2))
