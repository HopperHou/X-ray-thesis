#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


ROOT = Path("/hpc/zhou228/x-ray/Dataset/VinDR-CXR")
INPUT_CSV = ROOT / "annotations_train_WBF.csv"
OUTPUT_CSV = ROOT / "train-rle-box-vindr-cxr.csv"
TEMP_CSV = OUTPUT_CSV.with_suffix(".csv.tmp")
BOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
OUTPUT_COLUMNS = [
    "ID",
    "disease",
    "type_of_abnormalities",
    *BOX_COLUMNS,
    "source",
]


def main() -> None:
    annotations = pd.read_csv(INPUT_CSV, dtype={"image_id": str, "rad_id": str})
    required = ["image_id", "rad_id", "class_name", *BOX_COLUMNS]
    if list(annotations.columns) != required:
        raise RuntimeError(f"Unexpected columns in {INPUT_CSV}: {list(annotations.columns)}")
    if annotations["image_id"].isna().any() or annotations["class_name"].isna().any():
        raise RuntimeError("Missing image_id or class_name in WBF annotations")

    complete_boxes = annotations[BOX_COLUMNS].notna().all(axis=1)
    partial_boxes = annotations[BOX_COLUMNS].notna().any(axis=1) & ~complete_boxes
    invalid_boxes = complete_boxes & (
        (annotations["x_min"] >= annotations["x_max"])
        | (annotations["y_min"] >= annotations["y_max"])
    )
    if partial_boxes.any() or invalid_boxes.any():
        raise RuntimeError(
            f"Invalid WBF coordinates: partial={int(partial_boxes.sum())}, "
            f"invalid={int(invalid_boxes.sum())}"
        )

    output = pd.DataFrame(
        {
            "ID": annotations["image_id"],
            "disease": annotations["class_name"],
            "type_of_abnormalities": annotations["class_name"],
            "x_min": annotations["x_min"],
            "y_min": annotations["y_min"],
            "x_max": annotations["x_max"],
            "y_max": annotations["y_max"],
            "source": "VinDR-CXR",
        },
        columns=OUTPUT_COLUMNS,
    )
    output.to_csv(TEMP_CSV, index=False)

    check = pd.read_csv(TEMP_CSV, dtype={"ID": str})
    if list(check.columns) != OUTPUT_COLUMNS or len(check) != len(annotations):
        TEMP_CSV.unlink(missing_ok=True)
        raise RuntimeError("Temporary output verification failed")
    TEMP_CSV.replace(OUTPUT_CSV)

    print(f"Rows: {len(output)}")
    print(f"Images: {output['ID'].nunique()}")
    print(f"Box rows: {int(complete_boxes.sum())}")
    print(f"No finding rows: {int(output['disease'].eq('No finding').sum())}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
