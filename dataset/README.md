# Frozen annotation reference and partition records

This directory fixes the annotation and partition state used by the reported
A0-R, B2 and AXIS experiments. Use these frozen files to obtain the same retained
annotations and split membership. The original processing scripts document the
recoverable preparation history; some later VinDr-CXR decisions remain unrecovered.

## Final data interface

| File | Purpose |
| --- | --- |
| `All_data.csv` | Final curated annotation table: 31,124 rows, 18,438 retained image records, 27,749 boxes and 3,375 task-negative records. |
| `new_train.csv` | Frozen training annotations. |
| `new_validation.csv` | Frozen validation annotations. |
| `new_test.csv` | Frozen internal-test annotations. |
| `patient_level_split_image_assignments.csv` | Source/image-to-analysis-group and partition mapping, including the grouping basis. |
| `dataset_manifest.csv` | One row per retained image record; connects source IDs, anonymous keys, original image paths, training image/label paths, dimensions, box counts and negative status. |
| `records/patient_group_mapping.csv` | Original image-to-group mapping accompanying split generation. |

| Partition | Image records | Box annotations | Task-negative records |
| --- | ---: | ---: | ---: |
| Training | 14,934 | 22,477 | 2,734 |
| Validation | 1,660 | 2,502 | 304 |
| Internal test | 1,844 | 2,770 | 337 |
| Total | 18,438 | 27,749 | 3,375 |

The three split CSV files reproduce the exact row multiset of `All_data.csv`.
Image records are keyed by `(source, id)`; they are not a claim of globally unique
radiographs. The assignments preserve 10,985 analysis groups without overlap
between partitions. VinDr-CXR records use singleton fallback groups where patient
identifiers were unavailable, so this check does not establish patient separation
for those records. The cohort is an internal mixed-source cohort.

`All_data.csv` and the split CSV files use these fields:

```text
id,type_of_abnormalities,x_min,y_min,x_max,y_max,source,width,height
```

Coordinates are pixel-coordinate corners in the original image frame, before
training-time resizing. Each positive row represents one class/box annotation.
Task-negative rows have `type_of_abnormalities=no_finding` and empty coordinates.
`width` and `height` retain original image dimensions. YOLO labels are derived as
normalised centre coordinates and width/height using those dimensions.

The assignment file contains `source`, `id`, `patient_group`, `patient_id_basis`,
`partition` and `image_level_class_signature`. Its group identifiers are analysis
linkage keys, including documented source-specific fallback keys.

## Verify the download

From the repository root, using Python 3.10 or later:

```bash
python dataset/scripts/verify_frozen_dataset.py
```

The verifier uses only the Python standard library and makes no changes. It checks
every file listed in `SHA256SUMS`, matches the five input hashes embedded in the
historical preparation script, validates annotations and box bounds, checks the
exact split row union, confirms analysis-group separation, and reconciles the
assignment, group mapping and training manifest. It also checks the 3,009 verified
enclosing-box groups and their 6,505 input boxes. It does not decode images or run
the historical scripts. A successful run prints JSON with `status: PASS`.

Alternatively, from this directory on a system with `sha256sum`:

```bash
sha256sum -c SHA256SUMS
```

`records/final_interface_audit.json` records the read-only server verification of
the final annotation/split interface. `records/release_provenance.json` records
source paths, byte counts and SHA-256 hashes for copied files. The final
`All_data.csv` hash remains
`a44d2e86b847da005a6b4ede1135606fa79f2aa0d08bd0652a377e2713c75b50`.

## Original preparation scripts and execution boundary

The scripts under `scripts/original/` are byte-for-byte copies from the server:

- `prepare_dataset.py`: validates the frozen CSV inputs, resolves source rasters,
  checks image dimensions, creates image links and writes normalised YOLO labels
  when explicitly run with `--materialize`.
- `verify_setup.py` and `data.yaml`: the original configuration and materialised
  dataset checks for `New_trail_5dataset`.
- `generate_patient_level_splits.py`: original deterministic grouped split
  generation, using seed 42 and the historical source linkage metadata.
- `generate_train_csv.py`: an earlier source-table assembly step, preceding the
  final cross-source integration and later curation changes.
- `vindr/create_annotations_train_wbf.py`: class-agnostic coordinate averaging
  with equal reader weights, overlap threshold 0.60 and large-box filtering.
- `vindr/update_train_rle_box_from_wbf.py` and
  `vindr/clamp_negative_train_rle_coordinates.py`: source-table adaptation and
  negative-coordinate correction.
- `vindr/clean_vindr_same_class_overlaps.py`: verified same-class enclosing-box
  consolidation. Its original group audit is
  `records/vindr_overlap60_merge_audit.csv`; this audit excludes earlier WBF.
- `siim/merge_train_rle_connected_masks.py` and
  `siim/generate_train_boxes_from_merged_rle.py`: connected-mask consolidation
  and mask-to-box conversion.
- `siim/prepare_siim_yolo.py`: the recovered SIIM raster/label preparation code.

These originals retain their historical paths and input dependencies. They are
not a portable end-to-end launcher in this relocated directory. The preparation
script expects the frozen CSV files under the original `Simplified_dataset`
directory, and the original verifier/configuration expects the trial directory.
To materialise data elsewhere, acquire the corresponding source rasters, reproduce
the original layout or explicitly adapt paths in working copies of these scripts,
and preserve the frozen CSV contents and assignments. The final manifest records
the original paths as provenance; those paths are not download links.

## Recoverable history and remaining gaps

`records/patient_level_split_audit.json` is the original split audit.
`records/vindr_record_changes.json` distinguishes verified processing operations
from later record changes established by comparing saved tables. After verified
enclosing-box consolidation, 12 training-branch image identifiers carrying 34
boxes were absent from a later table, and nine further two-to-one enclosing-box
changes reduced the count by another nine boxes. The consensus branch also
contained two case-level changes. Their original decision rules were not
recovered; the released final reference preserves the annotations actually used.
The included scripts therefore do not reconstruct every historical curation
decision from raw annotations.

Images, DICOM files, runtime image caches and trained checkpoints are outside this
metadata directory. Source images must be obtained through the corresponding
dataset access routes and terms. Detailed upstream intensity-conversion histories
were not fully recovered. The recovered SIIM converter matches six inspected
PNG images exactly when recomputed, but the preserved conversion manifest marked
existing files as `kept`; this does not establish the generation history of every
PNG. Frozen metadata alone does not reproduce training without the required
rasters, model initialisations and software environment described in the main
repository.
