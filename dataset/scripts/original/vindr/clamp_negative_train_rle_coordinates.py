#!/usr/bin/env python3
"""Clamp negative coordinates in the VinDR training RLE/box CSV to zero."""

import csv
import os
import tempfile
from pathlib import Path


CSV_PATH = Path(__file__).resolve().parent / "train-rle-box-vindr-cxr.csv"
COORDINATE_COLUMNS = ("x_min", "y_min", "x_max", "y_max")


def main():
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)

    changed_rows = 0
    changed_values = 0
    for row in rows:
        row_changed = False
        for column in COORDINATE_COLUMNS:
            value = row[column]
            if value and float(value) < 0:
                row[column] = "0"
                changed_values += 1
                row_changed = True
        changed_rows += int(row_changed)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{CSV_PATH.name}.", suffix=".tmp", dir=CSV_PATH.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, CSV_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f"Changed rows: {changed_rows}")
    print(f"Changed coordinate values: {changed_values}")


if __name__ == "__main__":
    main()
