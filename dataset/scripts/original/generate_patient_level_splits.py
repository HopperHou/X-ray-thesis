#!/usr/bin/env python3
"""Create deterministic patient-grouped train/validation/test CSV files.

The requested global image targets are 81%, 9%, and 10%, implemented as
14,934 training, 1,660 validation, and 1,844 test images for the current
18,438-image All_data.csv.  Patient groups are kept intact.  Splits are
balanced greedily across source, positive/negative state, image-level class
presence, and box-level class counts.

VinDr-CXR does not publish a patient identifier; each VinDr image is therefore
treated as a documented singleton fallback group.  SIIM-ACR images use the
cross-dataset NIH Patient ID when a strict pixel-content mapping is available;
otherwise their released DICOM PatientID is retained as a source-local group.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT.parent
XRAY_ROOT = DATASET_ROOT.parent
CHECK_ROOT = XRAY_ROOT / "Check_overlap"

INPUT = ROOT / "All_data.csv"
TRAIN_OUTPUT = ROOT / "new_train.csv"
VALIDATION_OUTPUT = ROOT / "new_validation.csv"
TEST_OUTPUT = ROOT / "new_test.csv"
ASSIGNMENTS_OUTPUT = ROOT / "patient_level_split_image_assignments.csv"
AUDIT_OUTPUT = ROOT / "patient_level_split_audit.json"
PATIENT_MAP_OUTPUT = ROOT / "patient_group_mapping.csv"

NIH_DERIVED_MAPPING = CHECK_ROOT / "nih_patient_mapping_for_retained_images.csv"
NIH_METADATA = CHECK_ROOT / "Data_Entry_2017_v2020.csv"
RSNA_MAPPING = CHECK_ROOT / "rsna_pneumonia_mapping.json"
MS_CXR_ANNOTATIONS = DATASET_ROOT / "MS-CXR" / "MS_CXR_Local_Alignment_v1.0.0.csv"

BACKUP_SUFFIX = ".before_patient_level_split_20260813"
BACKUP_TARGETS = (
    TRAIN_OUTPUT,
    TEST_OUTPUT,
    ROOT / "final_new_split_image_assignments.csv",
    ROOT / "final_new_split_audit.json",
    ROOT / "new_train_test_summary.png",
)

FIELDS = [
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
BOX_FIELDS = ("x_min", "y_min", "x_max", "y_max")
NO_FINDING = "no_finding"
PARTITIONS = ("train", "validation", "test")
TARGET_IMAGES = {"train": 14934, "validation": 1660, "test": 1844}
FRACTIONS = {"train": 0.81, "validation": 0.09, "test": 0.10}
SPLIT_SEED = 42
PATIENT_AVAILABLE_SOURCES = {
    "MS-CXR",
    "RSNA",
    "ChestX-Det",
    "RSNA/ChestX-Det",
    "SIIM-ACR",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> bytes:
    value = "|".join((str(SPLIT_SEED), *parts))
    return hashlib.sha256(value.encode("utf-8")).digest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def validate_and_group(
    rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, str]]],
    dict[tuple[str, str], tuple[str, ...]],
]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        source, image_id = row["source"], row["id"]
        if not source or not image_id or source != source.strip() or image_id != image_id.strip():
            raise ValueError(f"Invalid source/id at line {line_number}")
        grouped[(source, image_id)].append(row)

    signatures: dict[tuple[str, str], tuple[str, ...]] = {}
    for identity, image_rows in grouped.items():
        dimensions = {(row["width"], row["height"]) for row in image_rows}
        if len(dimensions) != 1:
            raise ValueError(f"Conflicting dimensions for {identity}: {dimensions}")
        labels: set[str] = set()
        negative_rows = 0
        for row in image_rows:
            label = row["type_of_abnormalities"].strip()
            coordinates = [row[field].strip() for field in BOX_FIELDS]
            if any(coordinates) and not all(coordinates):
                raise ValueError(f"Partial box for {identity}")
            if label == NO_FINDING:
                if any(coordinates):
                    raise ValueError(f"no_finding has a box for {identity}")
                negative_rows += 1
                labels.add(label)
            else:
                if not label or not all(coordinates):
                    raise ValueError(f"Incomplete positive row for {identity}")
                x_min, y_min, x_max, y_max = map(float, coordinates)
                width, height = int(row["width"]), int(row["height"])
                if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                    raise ValueError(f"Out-of-bounds box for {identity}")
                labels.add(label)
        if NO_FINDING in labels:
            if labels != {NO_FINDING} or negative_rows != 1 or len(image_rows) != 1:
                raise ValueError(f"Invalid no_finding state for {identity}")
            signatures[identity] = (NO_FINDING,)
        else:
            signatures[identity] = tuple(sorted(labels, key=str.casefold))
    return grouped, signatures


def build_ms_patient_map() -> dict[str, str]:
    result: dict[str, str] = {}
    with MS_CXR_ANNOTATIONS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parts = row["path"].split("/")
            patient_parts = [
                part for part in parts if part.startswith("p") and part[1:].isdigit()
            ]
            if not patient_parts:
                raise ValueError(f"No MIMIC patient component in {row['path']}")
            patient_id = patient_parts[-1]
            previous = result.setdefault(row["dicom_id"], patient_id)
            if previous != patient_id:
                raise RuntimeError(f"Conflicting MS-CXR patient for {row['dicom_id']}")
    return result


def build_nih_patient_maps() -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    metadata: dict[str, str] = {}
    with NIH_METADATA.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            metadata[row["Image Index"]] = row["Patient ID"]
    with RSNA_MAPPING.open(encoding="utf-8") as handle:
        rsna_rows = json.load(handle)
    rsna: dict[str, str] = {}
    for row in rsna_rows:
        nih_image = row["img_id"]
        if nih_image not in metadata:
            raise KeyError(f"RSNA NIH image missing from metadata: {nih_image}")
        rsna[row["subset_img_id"]] = metadata[nih_image]

    derived: dict[tuple[str, str], str] = {}
    with NIH_DERIVED_MAPPING.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            # A small number of SIIM images cannot be linked back to the NIH
            # release under the strict full-resolution confirmation rule.
            # Their released DICOM PatientID is handled explicitly below.
            if not row["nih_patient_id"] or not row["patient_group"]:
                continue
            key = (row["source"], row["id"])
            previous = derived.setdefault(key, row["nih_patient_id"])
            if previous != row["nih_patient_id"]:
                raise RuntimeError(f"Conflicting derived NIH patient for {key}")
    return rsna, derived


def build_siim_dicom_patient_map(required_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in DATASET_ROOT.glob("SIIM-ACR/dicom-images-*/*/*/*.dcm"):
        if path.stem not in required_ids:
            continue
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        patient_id = str(getattr(dataset, "PatientID", "")).strip()
        if not patient_id:
            raise RuntimeError(f"SIIM DICOM PatientID missing: {path}")
        previous = result.setdefault(path.stem, patient_id)
        if previous != patient_id:
            raise RuntimeError(f"Conflicting SIIM PatientID for {path.stem}")
    missing = required_ids - set(result)
    if missing:
        raise FileNotFoundError(f"SIIM DICOM files missing for {len(missing)} images")
    return result


def patient_groups(
    identities: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    ms = build_ms_patient_map()
    rsna, derived = build_nih_patient_maps()
    siim_ids = {image_id for source, image_id in identities if source == "SIIM-ACR"}
    siim_dicom = build_siim_dicom_patient_map(siim_ids)
    group_map: dict[tuple[str, str], str] = {}
    basis_map: dict[tuple[str, str], str] = {}
    for identity in sorted(identities):
        source, image_id = identity
        if source == "MS-CXR":
            patient_id = ms.get(image_id)
            if patient_id is None:
                raise KeyError(f"MS-CXR patient mapping missing: {identity}")
            group_map[identity] = f"MIMIC:{patient_id}"
            basis_map[identity] = "MIMIC subject_id from MS-CXR path"
        elif source in {"RSNA", "RSNA/ChestX-Det"}:
            patient_id = rsna.get(image_id)
            if patient_id is None:
                raise KeyError(f"RSNA NIH patient mapping missing: {identity}")
            group_map[identity] = f"NIH:{patient_id}"
            basis_map[identity] = "NIH Patient ID via official RSNA image mapping"
        elif source == "ChestX-Det":
            patient_id = derived.get(identity)
            if patient_id is None:
                raise KeyError(f"Derived NIH patient mapping missing: {identity}")
            group_map[identity] = f"NIH:{patient_id}"
            basis_map[identity] = "NIH Patient ID via confirmed pixel-content mapping"
        elif source == "SIIM-ACR":
            patient_id = derived.get(identity)
            if patient_id is not None:
                group_map[identity] = f"NIH:{patient_id}"
                basis_map[identity] = (
                    "NIH Patient ID via confirmed pixel-content mapping"
                )
            else:
                patient_id = siim_dicom[image_id]
                group_map[identity] = f"SIIM:{patient_id}"
                basis_map[identity] = (
                    "SIIM DICOM PatientID; cross-dataset NIH mapping unavailable"
                )
        elif source == "VinDR-CXR":
            group_map[identity] = f"VINDR_SINGLETON:{image_id}"
            basis_map[identity] = "singleton fallback; VinDr-CXR patient ID unavailable"
        else:
            raise ValueError(f"Unsupported source: {source}")
    return group_map, basis_map


def group_features(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    signatures: dict[tuple[str, str], tuple[str, ...]],
    image_to_group: dict[tuple[str, str], str],
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, Counter[str]]]:
    group_images: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for identity, patient_group in image_to_group.items():
        group_images[patient_group].add(identity)
    features: dict[str, Counter[str]] = {}
    for patient_group, identities in group_images.items():
        values: Counter[str] = Counter()
        values["images:total"] = len(identities)
        # Sets deliberately model membership, but their iteration order varies
        # between Python processes. Sorting here makes feature insertion and
        # floating-point cost accumulation reproducible for the fixed seed.
        for identity in sorted(identities):
            source = identity[0]
            signature = signatures[identity]
            values[f"source:{source}"] += 1
            values[
                "state:negative" if signature == (NO_FINDING,) else "state:positive"
            ] += 1
            for label in signature:
                values[f"class_image:{label}"] += 1
            for row in grouped[identity]:
                if row["type_of_abnormalities"] != NO_FINDING:
                    values[f"class_box:{row['type_of_abnormalities']}"] += 1
        features[patient_group] = values
    return group_images, features


def feature_weight(feature: str) -> float:
    if feature.startswith("source:"):
        return 5.0
    if feature.startswith("state:"):
        return 4.0
    if feature.startswith("class_image:"):
        return 3.0
    if feature.startswith("class_box:"):
        return 1.0
    return 0.0


def choose_patient_partitions(
    features: dict[str, Counter[str]],
) -> dict[str, str]:
    totals: Counter[str] = Counter()
    for values in features.values():
        totals.update(values)
    if totals["images:total"] != sum(TARGET_IMAGES.values()):
        raise RuntimeError(
            f"Target image total mismatch: {totals['images:total']} != {sum(TARGET_IMAGES.values())}"
        )
    targets = {
        partition: {
            feature: total * FRACTIONS[partition]
            for feature, total in totals.items()
        }
        for partition in PARTITIONS
    }
    current = {partition: Counter() for partition in PARTITIONS}
    assignment: dict[str, str] = {}

    def rarity_score(patient_group: str) -> float:
        values = features[patient_group]
        return sum(
            feature_weight(feature) * count / math.sqrt(max(totals[feature], 1))
            for feature, count in sorted(values.items())
            if feature != "images:total"
        )

    ordered = sorted(
        features,
        key=lambda group: (
            -features[group]["images:total"],
            -rarity_score(group),
            stable_hash("order", group),
        ),
    )

    def incremental_cost(partition: str, values: Counter[str]) -> float:
        cost = 0.0
        for feature, addition in sorted(values.items()):
            if feature == "images:total":
                continue
            target = targets[partition][feature]
            scale = max(math.sqrt(target), 1.0)
            before = (current[partition][feature] - target) / scale
            after = (current[partition][feature] + addition - target) / scale
            cost += feature_weight(feature) * (after * after - before * before)
        target_total = TARGET_IMAGES[partition]
        before_fill = current[partition]["images:total"] / target_total
        after_fill = (
            current[partition]["images:total"] + values["images:total"]
        ) / target_total
        cost += 25.0 * (after_fill * after_fill - before_fill * before_fill)
        return cost

    for patient_group in ordered:
        values = features[patient_group]
        size = values["images:total"]
        candidates = [
            partition
            for partition in PARTITIONS
            if current[partition]["images:total"] + size <= TARGET_IMAGES[partition]
        ]
        if not candidates:
            raise RuntimeError(
                f"Patient group cannot fit any remaining image target: {patient_group}, size={size}"
            )
        chosen = min(
            candidates,
            key=lambda partition: (
                incremental_cost(partition, values),
                stable_hash("partition", patient_group, partition),
            ),
        )
        assignment[patient_group] = chosen
        current[chosen].update(values)

    actual = {
        partition: current[partition]["images:total"] for partition in PARTITIONS
    }
    if actual != TARGET_IMAGES:
        raise RuntimeError(f"Exact image targets were not reached: {actual}")
    return assignment


def atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if read_rows(temporary) != rows:
            raise RuntimeError(f"Read-back validation failed for {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def split_summary(
    rows: list[dict[str, str]], signatures: dict[tuple[str, str], tuple[str, ...]]
) -> dict[str, object]:
    identities = {(row["source"], row["id"]) for row in rows}
    source_images = Counter(source for source, _ in identities)
    negatives = sum(signatures[identity] == (NO_FINDING,) for identity in identities)
    class_boxes = Counter(
        row["type_of_abnormalities"]
        for row in rows
        if row["type_of_abnormalities"] != NO_FINDING
    )
    class_images: Counter[str] = Counter()
    for identity in identities:
        class_images.update(signatures[identity])
    return {
        "rows": len(rows),
        "images": len(identities),
        "positive_images": len(identities) - negatives,
        "negative_images": negatives,
        "boxes": sum(
            row["type_of_abnormalities"] != NO_FINDING for row in rows
        ),
        "source_images": dict(sorted(source_images.items())),
        "class_images": dict(sorted(class_images.items())),
        "class_boxes": dict(sorted(class_boxes.items())),
    }


def preserve_existing_outputs() -> dict[str, str]:
    backups: dict[str, str] = {}
    for original in BACKUP_TARGETS:
        if not original.is_file():
            continue
        backup = original.with_name(
            f"{original.stem}{BACKUP_SUFFIX}{original.suffix}"
        )
        if not backup.exists():
            shutil.copy2(original, backup)
        backups[str(original)] = str(backup)
    return backups


def main() -> None:
    for required in (
        INPUT,
        NIH_DERIVED_MAPPING,
        NIH_METADATA,
        RSNA_MAPPING,
        MS_CXR_ANNOTATIONS,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows = read_rows(INPUT)
    grouped, signatures = validate_and_group(rows)
    all_ids = set(grouped)
    image_to_group, basis_map = patient_groups(all_ids)
    if set(image_to_group) != all_ids:
        raise AssertionError("Patient mapping does not exactly cover All_data images")
    group_images, features = group_features(grouped, signatures, image_to_group)
    group_partition = choose_patient_partitions(features)
    image_partition = {
        identity: group_partition[image_to_group[identity]] for identity in all_ids
    }

    partition_ids = {
        partition: {
            identity
            for identity, assigned in image_partition.items()
            if assigned == partition
        }
        for partition in PARTITIONS
    }
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            if partition_ids[left] & partition_ids[right]:
                raise AssertionError(f"Image overlap between {left} and {right}")
    if set().union(*partition_ids.values()) != all_ids:
        raise AssertionError("Split images do not exactly cover All_data")

    patient_sets = {
        partition: {
            image_to_group[identity] for identity in partition_ids[partition]
        }
        for partition in PARTITIONS
    }
    patient_overlap: dict[str, int] = {}
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            patient_overlap[f"{left}_{right}"] = len(overlap)
            if overlap:
                raise AssertionError(f"Patient overlap between {left} and {right}")

    output_paths = {
        "train": TRAIN_OUTPUT,
        "validation": VALIDATION_OUTPUT,
        "test": TEST_OUTPUT,
    }
    output_rows = {
        partition: [
            row
            for row in rows
            if image_partition[(row["source"], row["id"])] == partition
        ]
        for partition in PARTITIONS
    }
    row_signature = lambda row: tuple(row[field] for field in FIELDS)
    combined_rows = [row for partition in PARTITIONS for row in output_rows[partition]]
    if Counter(map(row_signature, combined_rows)) != Counter(map(row_signature, rows)):
        raise AssertionError("Output rows are not the exact All_data row multiset")

    backups = preserve_existing_outputs()
    for partition, path in output_paths.items():
        atomic_write(path, output_rows[partition])

    with PATIENT_MAP_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["source", "id", "patient_group", "patient_id_basis", "patient_id_available"]
        )
        for identity in sorted(all_ids):
            writer.writerow(
                [
                    identity[0],
                    identity[1],
                    image_to_group[identity],
                    basis_map[identity],
                    identity[0] in PATIENT_AVAILABLE_SOURCES,
                ]
            )

    with ASSIGNMENTS_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "source",
                "id",
                "patient_group",
                "patient_id_basis",
                "partition",
                "image_level_class_signature",
            ]
        )
        for identity in sorted(all_ids):
            writer.writerow(
                [
                    identity[0],
                    identity[1],
                    image_to_group[identity],
                    basis_map[identity],
                    image_partition[identity],
                    " | ".join(signatures[identity]),
                ]
            )

    audit = {
        "input": str(INPUT),
        "input_sha256": file_sha256(INPUT),
        "method": "deterministic patient-grouped greedy multilabel stratification",
        "seed": SPLIT_SEED,
        "requested_fractions": FRACTIONS,
        "target_images": TARGET_IMAGES,
        "patient_group_count": len(group_images),
        "patient_id_available_images": sum(
            identity[0] in PATIENT_AVAILABLE_SOURCES for identity in all_ids
        ),
        "patient_id_unavailable_singleton_images": sum(
            identity[0] not in PATIENT_AVAILABLE_SOURCES for identity in all_ids
        ),
        "patient_id_unavailable_sources": ["VinDR-CXR"],
        "siim_source_local_patient_id_images": sum(
            basis == "SIIM DICOM PatientID; cross-dataset NIH mapping unavailable"
            for basis in basis_map.values()
        ),
        "patient_group_overlap": patient_overlap,
        "exact_input_row_union": True,
        "backups": backups,
        "all_data": split_summary(rows, signatures),
    }
    for partition in PARTITIONS:
        audit[partition] = split_summary(output_rows[partition], signatures)
        audit[f"{partition}_csv"] = str(output_paths[partition])
        audit[f"{partition}_sha256"] = file_sha256(output_paths[partition])
    audit["assignment_csv"] = str(ASSIGNMENTS_OUTPUT)
    audit["assignment_sha256"] = file_sha256(ASSIGNMENTS_OUTPUT)
    audit["patient_mapping_csv"] = str(PATIENT_MAP_OUTPUT)
    audit["patient_mapping_sha256"] = file_sha256(PATIENT_MAP_OUTPUT)
    AUDIT_OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print("ALL_PATIENT_LEVEL_SPLIT_CHECKS_PASS=1")


if __name__ == "__main__":
    main()
