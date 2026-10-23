#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/hpc/zhou228/x-ray/Dataset/VinDR-CXR")
INPUT_CSV = ROOT / "annotations_train.csv"
OUTPUT_CSV = ROOT / "annotations_train_WBF.csv"
IMAGE_DIR = ROOT / "images" / "train"
IOU_THRESHOLD = 0.60
MAX_BOX_AREA_RATIO = 1 / 6
BOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
OUTPUT_COLUMNS = ["image_id", "rad_id", "class_name", *BOX_COLUMNS]


@dataclass(frozen=True)
class BoxRecord:
    row_order: int
    rad_id: str
    class_names: tuple[str, ...]
    box: np.ndarray
    source_rows: int


def rad_sort_key(rad_id: str) -> tuple[int, str]:
    suffix = rad_id.removeprefix("R")
    return (int(suffix), rad_id) if suffix.isdigit() else (10**9, rad_id)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-12)


def wbf(records: list[BoxRecord]) -> np.ndarray:
    # All radiologists have equal weight, so WBF is the coordinate-wise mean.
    return np.mean([record.box for record in records], axis=0)


def joined_class_names(records: list[BoxRecord]) -> str:
    names = []
    for record in sorted(records, key=lambda item: item.row_order):
        for name in record.class_names:
            if name not in names:
                names.append(name)
    return " | ".join(names)


def build_position_records(group: pd.DataFrame, counters: dict[str, int]) -> list[BoxRecord]:
    records = []
    keys = ["rad_id", *BOX_COLUMNS]
    for _, position_group in group.groupby(keys, sort=False, dropna=False):
        ordered = position_group.sort_values("row_order", kind="stable")
        class_names = tuple(dict.fromkeys(ordered["class_name"].astype(str)))
        first = ordered.iloc[0]
        records.append(
            BoxRecord(
                row_order=int(first["row_order"]),
                rad_id=str(first["rad_id"]),
                class_names=class_names,
                box=first[BOX_COLUMNS].to_numpy(dtype=float),
                source_rows=len(ordered),
            )
        )
        counters["same_doctor_same_position_rows_collapsed"] += len(ordered) - 1
    return records


def box_area_ratio(record: BoxRecord, image_area: float) -> float:
    width = record.box[2] - record.box[0]
    height = record.box[3] - record.box[1]
    return width * height / image_area


def fuse_image_boxes(group: pd.DataFrame, image_area: float, counters: dict[str, int]) -> list[dict]:
    records = build_position_records(group, counters)
    available = set(range(len(records)))
    fused_groups: list[list[int]] = []

    while True:
        candidates = []
        ordered = sorted(available)
        for offset, first_index in enumerate(ordered):
            for second_index in ordered[offset + 1 :]:
                first = records[first_index]
                second = records[second_index]
                if first.rad_id == second.rad_id:
                    continue
                overlap = box_iou(first.box, second.box)
                if overlap >= IOU_THRESHOLD:
                    candidates.append((overlap, first_index, second_index))

        if not candidates:
            break

        _, first_index, second_index = max(
            candidates,
            key=lambda item: (item[0], -records[item[1]].row_order, -records[item[2]].row_order),
        )
        cluster = [first_index, second_index]
        available.remove(first_index)
        available.remove(second_index)

        cluster_rads = {records[index].rad_id for index in cluster}
        third_candidates = []
        for index in available:
            record = records[index]
            if record.rad_id in cluster_rads:
                continue
            overlaps = [box_iou(record.box, records[member].box) for member in cluster]
            if min(overlaps) >= IOU_THRESHOLD:
                third_candidates.append((min(overlaps), np.mean(overlaps), index))

        if third_candidates:
            _, _, third_index = max(
                third_candidates,
                key=lambda item: (item[0], item[1], -records[item[2]].row_order),
            )
            cluster.append(third_index)
            available.remove(third_index)

        fused_groups.append(cluster)

    output = []
    for cluster in fused_groups:
        members = [records[index] for index in cluster]
        fused_box = wbf(members)
        rad_ids = sorted((member.rad_id for member in members), key=rad_sort_key)
        output.append(
            {
                "row_order": min(member.row_order for member in members),
                "rad_id": "WBF_" + "_".join(rad_ids),
                "class_name": joined_class_names(members),
                **dict(zip(BOX_COLUMNS, fused_box)),
            }
        )
        counters["wbf_groups"] += 1
        counters[f"wbf_groups_{len(members)}_doctors"] += 1
        counters["source_position_boxes_fused"] += len(members)
        for member in members:
            if box_area_ratio(member, image_area) >= MAX_BOX_AREA_RATIO:
                counters["consensus_large_position_boxes_retained"] += 1
                counters["consensus_large_box_rows_retained"] += member.source_rows

    for index in available:
        record = records[index]
        if box_area_ratio(record, image_area) >= MAX_BOX_AREA_RATIO:
            counters["large_position_boxes_removed"] += 1
            counters["large_box_rows_removed"] += record.source_rows
            continue
        output.append(
            {
                "row_order": record.row_order,
                "rad_id": record.rad_id,
                "class_name": joined_class_names([record]),
                **dict(zip(BOX_COLUMNS, record.box)),
            }
        )
        counters["unmatched_boxes_retained"] += 1

    return output


def main() -> None:
    data = pd.read_csv(INPUT_CSV, dtype={"image_id": str, "rad_id": str})
    if list(data.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(f"Unexpected input columns: {list(data.columns)}")

    data = data.reset_index(names="row_order")
    data["has_box"] = data[BOX_COLUMNS].notna().all(axis=1)
    data["is_no_finding"] = data["class_name"].eq("No finding")

    doctor_state = data.groupby(["image_id", "rad_id"], sort=False).agg(
        has_box=("has_box", "any"),
        is_no_finding=("is_no_finding", "any"),
    )
    if (doctor_state["has_box"] & doctor_state["is_no_finding"]).any():
        raise RuntimeError("A radiologist has both No finding and box annotations for the same image")

    doctors_per_image = doctor_state.groupby("image_id").size()
    if not doctors_per_image.eq(3).all():
        raise RuntimeError("Not every image has exactly three radiologists")

    no_finding_votes = doctor_state.groupby("image_id")["is_no_finding"].sum()
    counters = {
        "no_finding_images": 0,
        "positive_images": 0,
        "box_rows_removed_by_majority_vote": 0,
        "large_box_rows_removed": 0,
        "large_position_boxes_removed": 0,
        "images_with_large_boxes_removed": 0,
        "consensus_large_box_rows_retained": 0,
        "consensus_large_position_boxes_retained": 0,
        "positive_images_without_boxes_after_filter": 0,
        "wbf_groups": 0,
        "wbf_groups_2_doctors": 0,
        "wbf_groups_3_doctors": 0,
        "same_doctor_same_position_rows_collapsed": 0,
        "source_position_boxes_fused": 0,
        "unmatched_boxes_retained": 0,
    }
    output_rows = []
    excluded_positive_images = []

    for image_id, image_group in data.groupby("image_id", sort=False):
        if no_finding_votes.loc[image_id] >= 2:
            no_finding_rads = sorted(
                image_group.loc[image_group["is_no_finding"], "rad_id"].unique(), key=rad_sort_key
            )
            output_rows.append(
                {
                    "row_order": int(image_group["row_order"].min()),
                    "image_id": image_id,
                    "rad_id": "WBF_" + "_".join(no_finding_rads),
                    "class_name": "No finding",
                    **{column: np.nan for column in BOX_COLUMNS},
                }
            )
            counters["no_finding_images"] += 1
            counters["box_rows_removed_by_majority_vote"] += int(image_group["has_box"].sum())
            continue

        counters["positive_images"] += 1
        positive_rows = image_group[image_group["has_box"]].copy()
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing train image: {image_path}")
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        removed_before = counters["large_position_boxes_removed"]
        fused_rows = fuse_image_boxes(
            positive_rows,
            image_area=float(image_width * image_height),
            counters=counters,
        )
        counters["images_with_large_boxes_removed"] += int(
            counters["large_position_boxes_removed"] > removed_before
        )
        if not fused_rows:
            counters["positive_images_without_boxes_after_filter"] += 1
            excluded_positive_images.append(image_id)
            continue
        for row in fused_rows:
            row["image_id"] = image_id
            output_rows.append(row)

    output = pd.DataFrame(output_rows)
    output = output.sort_values(["row_order", "image_id"], kind="stable")[OUTPUT_COLUMNS]
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"Input rows: {len(data)}")
    print(f"Input images: {data['image_id'].nunique()}")
    print(f"Output rows: {len(output)}")
    print(f"Output images: {output['image_id'].nunique()}")
    for key, value in counters.items():
        print(f"{key}: {value}")
    if excluded_positive_images:
        print("Excluded positive images with no boxes after filtering:")
        for image_id in excluded_positive_images:
            print(f"  {image_id}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
