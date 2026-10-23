#!/usr/bin/env python3
"""Merge highly overlapping same-class VinDR-CXR boxes in train.csv.

Boxes are linked when intersection / min(box areas) >= threshold. Each connected
component is replaced by its enclosing rectangle. The procedure is repeated
until no newly-created enclosing rectangles satisfy the criterion.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = [
    "id", "type_of_abnormalities", "x_min", "y_min", "x_max", "y_max",
    "source", "width", "height",
]
COORDS = ("x_min", "y_min", "x_max", "y_max")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def overlap_over_smaller(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


def components(nodes, threshold):
    n = len(nodes)
    adjacent = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_over_smaller(nodes[i]["box"], nodes[j]["box"]) + 1e-12 >= threshold:
                adjacent[i].add(j)
                adjacent[j].add(i)

    result, seen = [], set()
    for start in range(n):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacent[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def merge_until_stable(nodes, threshold):
    rounds = 0
    while True:
        groups = components(nodes, threshold)
        if all(len(group) == 1 for group in groups):
            return nodes, rounds
        merged = []
        for group in groups:
            selected = [nodes[i] for i in group]
            if len(selected) == 1:
                merged.append(selected[0])
                continue
            members = sorted(m for node in selected for m in node["members"])
            boxes = [node["box"] for node in selected]
            merged.append({
                "members": members,
                "box": (
                    min(box[0] for box in boxes), min(box[1] for box in boxes),
                    max(box[2] for box in boxes), max(box[3] for box in boxes),
                ),
            })
        nodes = sorted(merged, key=lambda node: node["members"][0])
        rounds += 1


def format_number(value):
    return format(value, ".15g")


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected columns: {reader.fieldnames!r}")
        return list(reader)


def validate_rows(rows):
    for line_no, row in enumerate(rows, 2):
        if row["source"] == "VinDR-CXR" and row["type_of_abnormalities"] != "no_finding":
            box = tuple(float(row[key]) for key in COORDS)
            width, height = float(row["width"]), float(row["height"])
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid or out-of-bounds box at CSV line {line_no}: {row}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("train.csv"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    backup = (args.backup or source.with_name("train.before-vindr-overlap60.csv")).resolve()
    audit = (args.audit or source.with_name("vindr_overlap60_merge_audit.csv")).resolve()
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")

    original = read_rows(source)
    validate_rows(original)
    grouped = defaultdict(list)
    for index, row in enumerate(original):
        if row["source"] == "VinDR-CXR" and row["type_of_abnormalities"] != "no_finding":
            grouped[(row["id"], row["type_of_abnormalities"])].append({
                "members": [index],
                "box": tuple(float(row[key]) for key in COORDS),
            })

    replacement, removed, max_rounds = {}, set(), 0
    audit_rows = []
    per_class_removed = Counter()
    for (image_id, abnormality), nodes in grouped.items():
        final_nodes, rounds = merge_until_stable(nodes, args.threshold)
        max_rounds = max(max_rounds, rounds)
        for node in final_nodes:
            if len(node["members"]) == 1:
                continue
            first = node["members"][0]
            replacement[first] = node["box"]
            removed.update(node["members"][1:])
            per_class_removed[abnormality] += len(node["members"]) - 1
            input_boxes = [
                [original[i][key] for key in COORDS] for i in node["members"]
            ]
            audit_rows.append({
                "id": image_id,
                "type_of_abnormalities": abnormality,
                "input_box_count": str(len(node["members"])),
                "input_row_numbers": json.dumps([i + 2 for i in node["members"]]),
                "input_boxes": json.dumps(input_boxes, separators=(",", ":")),
                "merged_x_min": format_number(node["box"][0]),
                "merged_y_min": format_number(node["box"][1]),
                "merged_x_max": format_number(node["box"][2]),
                "merged_y_max": format_number(node["box"][3]),
            })

    cleaned = []
    for index, old_row in enumerate(original):
        if index in removed:
            continue
        row = dict(old_row)
        if index in replacement:
            for key, value in zip(COORDS, replacement[index]):
                row[key] = format_number(value)
        cleaned.append(row)
    validate_rows(cleaned)

    # Strict preservation checks for all untouched records and dataset membership.
    expected = []
    for i, row in enumerate(original):
        if i not in removed:
            expected.append(i)
    if len(cleaned) != len(original) - len(removed):
        raise AssertionError("row count mismatch")
    for out_row, original_index in zip(cleaned, expected):
        old_row = original[original_index]
        if original_index not in replacement and out_row != old_row:
            raise AssertionError(f"untouched row changed: original index {original_index}")
        for key in ("id", "type_of_abnormalities", "source", "width", "height"):
            if out_row[key] != old_row[key]:
                raise AssertionError(f"metadata changed at original index {original_index}: {key}")
    if {(r["id"], r["source"]) for r in cleaned} != {(r["id"], r["source"]) for r in original}:
        raise AssertionError("image/source membership changed")

    # The requested invariant must hold after writing the merged rectangles.
    final_groups = defaultdict(list)
    for row in cleaned:
        if row["source"] == "VinDR-CXR" and row["type_of_abnormalities"] != "no_finding":
            final_groups[(row["id"], row["type_of_abnormalities"])].append(
                tuple(float(row[key]) for key in COORDS)
            )
    remaining = 0
    for boxes in final_groups.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                remaining += overlap_over_smaller(boxes[i], boxes[j]) + 1e-12 >= args.threshold
    if remaining:
        raise AssertionError(f"{remaining} qualifying pairs remain")

    shutil.copy2(source, backup)
    if sha256(source) != sha256(backup):
        raise AssertionError("backup checksum mismatch")
    original_mode = stat.S_IMODE(source.stat().st_mode)
    fd, temp_name = tempfile.mkstemp(prefix=source.name + ".", suffix=".tmp", dir=source.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(cleaned)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp_name, original_mode)
        os.replace(temp_name, source)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    audit_fields = [
        "id", "type_of_abnormalities", "input_box_count", "input_row_numbers",
        "input_boxes", "merged_x_min", "merged_y_min", "merged_x_max", "merged_y_max",
    ]
    with open(audit, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(sorted(audit_rows, key=lambda r: (r["id"], r["type_of_abnormalities"])))

    print(f"input_rows={len(original)}")
    print(f"output_rows={len(cleaned)}")
    print(f"removed_boxes={len(removed)}")
    print(f"merged_groups={len(audit_rows)}")
    print(f"max_merge_rounds={max_rounds}")
    print(f"removed_by_class={dict(sorted(per_class_removed.items()))}")
    print("remaining_qualifying_pairs=0")
    print(f"backup={backup}")
    print(f"backup_sha256={sha256(backup)}")
    print(f"output_sha256={sha256(source)}")
    print(f"audit={audit}")


if __name__ == "__main__":
    main()
