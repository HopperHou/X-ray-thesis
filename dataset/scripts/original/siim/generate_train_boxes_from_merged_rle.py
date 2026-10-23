#!/usr/bin/env python3
"""Generate SIIM-ACR box annotations from a merged RLE CSV."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
HEIGHT = 1024
WIDTH = 1024
OUTPUT_COLUMNS = [
    "ID",
    "disease",
    "type_of_abnormalities",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "source",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SIIM box annotations from a merged RLE split."
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Merged RLE split to convert (default: train)",
    )
    return parser.parse_args()


def decode_siim_rle(encoded):
    flat = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)
    encoded = encoded.strip()
    if encoded == "-1":
        return flat.reshape((WIDTH, HEIGHT)).T

    numbers = np.fromstring(encoded, sep=" ", dtype=np.int64)
    if len(numbers) % 2:
        raise ValueError(f"RLE has an odd number of tokens: {encoded[:80]}")

    running_start = 0
    previous_length = 0
    for start_offset, length in zip(numbers[0::2], numbers[1::2]):
        running_start += previous_length + int(start_offset)
        start = running_start - 1
        end = start + int(length)
        if start < 0 or end > flat.size:
            raise ValueError(
                f"RLE index outside image: start={running_start}, length={length}"
            )
        flat[start:end] = 1
        previous_length = int(length)
    return flat.reshape((WIDTH, HEIGHT)).T


def mask_to_box(mask):
    y_coordinates, x_coordinates = np.where(mask != 0)
    if not len(x_coordinates):
        return None
    return (
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()),
        int(y_coordinates.max()),
    )


def generate_rows(input_csv):
    output_rows = []
    negative_rows = 0
    positive_rows = 0

    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ImageId", " EncodedPixels"]:
            raise ValueError(f"Unexpected input columns: {reader.fieldnames}")

        for line_number, row in enumerate(reader, 2):
            image_id = row["ImageId"].strip()
            encoded = row[" EncodedPixels"].strip()
            if not image_id:
                raise ValueError(f"Empty ImageId on input line {line_number}")

            if encoded == "-1":
                negative_rows += 1
                output_rows.append(
                    {
                        "ID": image_id,
                        "disease": "no_findings",
                        "type_of_abnormalities": "",
                        "x_min": "",
                        "y_min": "",
                        "x_max": "",
                        "y_max": "",
                        "source": "SIIM-ACR",
                    }
                )
                continue

            mask = decode_siim_rle(encoded)
            box = mask_to_box(mask)
            if box is None:
                raise ValueError(f"Positive RLE is empty on input line {line_number}")
            x_min, y_min, x_max, y_max = box
            if not (0 <= x_min <= x_max < WIDTH and 0 <= y_min <= y_max < HEIGHT):
                raise ValueError(f"Invalid box on input line {line_number}: {box}")
            positive_rows += 1
            output_rows.append(
                {
                    "ID": image_id,
                    "disease": "pneumothorax",
                    "type_of_abnormalities": "",
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "source": "SIIM-ACR",
                }
            )

    return output_rows, positive_rows, negative_rows


def write_output(output_csv, rows):
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_csv.name}.", suffix=".tmp", dir=output_csv.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, output_csv)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    args = parse_args()
    input_csv = ROOT / f"{args.split}-rle-merged.csv"
    output_csv = ROOT / f"{args.split}-rle-box-siim-acr.csv"
    rows, positive_rows, negative_rows = generate_rows(input_csv)
    if positive_rows + negative_rows != len(rows):
        raise RuntimeError("Generated row count is inconsistent")
    input_ids = {row["ID"] for row in rows}
    if not input_ids:
        raise RuntimeError("Generated CSV contains no image IDs")
    write_output(output_csv, rows)
    print(f"Images: {len({row['ID'] for row in rows})}")
    print(f"Positive box rows: {positive_rows}")
    print(f"Negative rows: {negative_rows}")
    print(f"Total rows: {len(rows)}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
