#!/usr/bin/env python3
"""Merge connected SIIM-ACR RLE observations image by image."""

import argparse
import csv
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
HEIGHT = 1024
WIDTH = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge overlapping or 8-neighbor-connected SIIM RLE rows."
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="CSV split to process (default: train)",
    )
    return parser.parse_args()


def decode_siim_rle(encoded):
    mask = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)
    encoded = encoded.strip()
    if not encoded or encoded == "-1":
        return mask.reshape((WIDTH, HEIGHT)).T

    numbers = np.fromstring(encoded, sep=" ", dtype=np.int64)
    if len(numbers) % 2:
        raise ValueError(f"RLE has an odd number of tokens: {encoded[:80]}")

    running_start = 0
    previous_length = 0
    for start_offset, length in zip(numbers[0::2], numbers[1::2]):
        running_start += previous_length + int(start_offset)
        start = running_start - 1
        end = start + int(length)
        if start < 0 or end > mask.size:
            raise ValueError(
                f"RLE index outside image: start={running_start}, length={length}"
            )
        mask[start:end] = 1
        previous_length = int(length)
    return mask.reshape((WIDTH, HEIGHT)).T


def encode_siim_rle(mask):
    pixels = np.asarray(mask, dtype=np.uint8).T.reshape(-1)
    padded = np.concatenate(([0], pixels, [0]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    absolute_starts = transitions[0::2] + 1
    lengths = transitions[1::2] - transitions[0::2]

    tokens = []
    previous_start = 0
    previous_length = 0
    for absolute_start, length in zip(absolute_starts, lengths):
        start_offset = int(absolute_start) - previous_start - previous_length
        if start_offset <= 0 or length <= 0:
            raise ValueError("Generated an invalid SIIM relative RLE run")
        tokens.extend((str(start_offset), str(int(length))))
        previous_start = int(absolute_start)
        previous_length = int(length)
    return " ".join(tokens)


def masks_connected(first, second):
    if np.any(first & second):
        return True
    first_padded = np.pad(first, 1, mode="constant")
    neighbors = np.zeros_like(second)
    for y_offset in range(3):
        for x_offset in range(3):
            if y_offset == 1 and x_offset == 1:
                continue
            neighbors |= first_padded[
                y_offset : y_offset + HEIGHT,
                x_offset : x_offset + WIDTH,
            ]
    return bool(np.any(neighbors & second))


def connected_row_groups(masks):
    adjacency = [set() for _ in masks]
    for first_index in range(len(masks)):
        for second_index in range(first_index + 1, len(masks)):
            if masks_connected(masks[first_index], masks[second_index]):
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)

    groups = []
    visited = set()
    for start_index in range(len(masks)):
        if start_index in visited:
            continue
        group = []
        stack = [start_index]
        visited.add(start_index)
        while stack:
            index = stack.pop()
            group.append(index)
            for neighbor in adjacency[index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(group))
    return groups


def load_rows(input_csv):
    groups = OrderedDict()
    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ImageId", " EncodedPixels"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        for row in reader:
            image_id = row["ImageId"].strip()
            encoded = row[" EncodedPixels"].strip()
            groups.setdefault(image_id, []).append(encoded)
    return groups


def merge_groups(groups):
    output_rows = []
    stats = {
        "negative_images": 0,
        "positive_images": 0,
        "input_positive_rows": 0,
        "output_positive_rows": 0,
        "images_with_changed_row_count": 0,
    }

    for image_id, encoded_rows in groups.items():
        positives = [encoded for encoded in encoded_rows if encoded != "-1"]
        negatives = [encoded for encoded in encoded_rows if encoded == "-1"]
        if positives and negatives:
            raise ValueError(f"Image {image_id} has both positive and negative rows")
        if not positives:
            stats["negative_images"] += 1
            output_rows.append((image_id, "-1"))
            continue

        stats["positive_images"] += 1
        stats["input_positive_rows"] += len(positives)
        masks = [decode_siim_rle(encoded) for encoded in positives]
        merged_rows = []
        for row_group in connected_row_groups(masks):
            combined = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            for row_index in row_group:
                combined |= masks[row_index]
            merged_rows.append(encode_siim_rle(combined))

        if not merged_rows:
            raise ValueError(f"Positive image {image_id} produced no connected components")
        stats["output_positive_rows"] += len(merged_rows)
        stats["images_with_changed_row_count"] += int(len(merged_rows) != len(positives))
        output_rows.extend((image_id, encoded) for encoded in merged_rows)

    return output_rows, stats


def validate(groups, output_rows):
    output_groups = OrderedDict()
    for image_id, encoded in output_rows:
        output_groups.setdefault(image_id, []).append(encoded)
    if list(output_groups) != list(groups):
        raise ValueError("Output ImageId set or order differs from input")

    for image_id, input_rows in groups.items():
        expected = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for encoded in input_rows:
            expected |= decode_siim_rle(encoded)

        actual = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        decoded_output_rows = []
        for encoded in output_groups[image_id]:
            decoded = decode_siim_rle(encoded)
            if encoded != "-1":
                if np.any(actual & decoded):
                    raise ValueError(f"Output components overlap for {image_id}")
                decoded_output_rows.append(decoded)
            actual |= decoded
        if not np.array_equal(expected, actual):
            raise ValueError(f"Merged mask changed pixels for {image_id}")
        for first_index in range(len(decoded_output_rows)):
            for second_index in range(first_index + 1, len(decoded_output_rows)):
                if masks_connected(
                    decoded_output_rows[first_index], decoded_output_rows[second_index]
                ):
                    raise ValueError(f"Connected output rows remain for {image_id}")


def write_output(output_csv, output_rows):
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_csv.name}.", suffix=".tmp", dir=output_csv.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["ImageId", " EncodedPixels"])
            writer.writerows(output_rows)
        os.replace(temp_name, output_csv)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    args = parse_args()
    input_csv = ROOT / f"{args.split}-rle.csv"
    output_csv = ROOT / f"{args.split}-rle-merged.csv"
    groups = load_rows(input_csv)
    output_rows, stats = merge_groups(groups)
    validate(groups, output_rows)
    write_output(output_csv, output_rows)

    print(f"Images: {len(groups)}")
    print(f"Positive images: {stats['positive_images']}")
    print(f"Negative images: {stats['negative_images']}")
    print(f"Input positive rows: {stats['input_positive_rows']}")
    print(f"Output connected-mask rows: {stats['output_positive_rows']}")
    print(f"Images with changed positive row count: {stats['images_with_changed_row_count']}")
    print(f"Output rows: {len(output_rows)}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
