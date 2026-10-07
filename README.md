# AXIS: Chest X-ray Object Detection

Source code and official experiment records for the current thesis model: **AXIS E50 / O2O**, built on YOLO26. The final Plan B experiment used paired training with three seeds (42, 43, and 44). Seed 42 at epoch 50 is the predeclared representative checkpoint.

## Repository contents

- `implementation/axis_full50/`: model, quality head, losses, training, evaluation, and statistical analysis code, including the original frozen protocols and SHA256 manifests.
- `refine-logs/AXIS_FULL50_20260930/RESULTS_SNAPSHOT_20261001_1839/`: official E50 reports, per-class results, convergence CSV, evaluation matrices, and pairing and freeze checks.
- `dataset/`: CSV files only. Raw images, DICOM files, YOLO label text files, and dataset archives are excluded.
- `weights/`: trained model checkpoints, stored separately from dataset files and handled with Git LFS.

The repository includes the original AXIS E50 checkpoints for seeds 42, 43, and 44, plus the final `dataset/All_data.csv`. All four files were downloaded from hpc6 and verified against the SHA256 values in the original experiment records. File checksums are listed in [SHA256SUMS](SHA256SUMS).

| Checkpoint | Role |
| --- | --- |
| `weights/AXIS_seed42_E50.pt` | Predeclared representative model |
| `weights/AXIS_seed43_E50.pt` | Second training seed |
| `weights/AXIS_seed44_E50.pt` | Third training seed |

Each checkpoint is 238,430,301 bytes. These are original research checkpoint dictionaries containing a full `model_state` and experiment metadata, rather than standard Ultralytics serialized model objects. Loading requires reconstructing the compatible unfused detector, installing the Q-head with `install_v6a2_head`, and loading `model_state` with strict matching. The archived evaluator in `full_eval.py` constructs that detector from the original A0-R checkpoint.

## Model

AXIS trains the O2O box regression towers and Q-heads. The backbone, neck, O2M branch, and O2O classification towers remain frozen. Inference uses native O2O Top300 without NMS; the quality score correction is fixed to the identity mapping.

The fixed `axis_center_v1` geometry objective is:

```text
L_AXIS = B * L_box_native + L_Q + B * lambda_geo * L_geo_axis
L_geo_axis = mean_fg[(2 - 2^IoU)
                    + 0.5 * (abs(dx) / max(GT_width, stride)
                             + abs(dy) / max(GT_height, stride))]
lambda_geo = 0.6070424318313599
```

Here, `B` is the actual batch size and `mean_fg` averages over native O2O assigned foreground locations. `dx` and `dy` are predicted-to-target box center offsets in pixels.

Class order: Pneumonia, Pneumothorax, Cardiomegaly, Aortic enlargement, Pleural thickening, and Pulmonary fibrosis.

## Official results

On the fixed development set of 1,660 images, the three-seed mean AXIS mAP50–95 is **27.567866%**, compared with **27.355653%** for A0-R: a difference of **+0.212213 percentage points**. The mean AXIS–B2 difference is −0.000028 percentage points, which does not support an independent mAP gain from the axis term. Full metrics and conditional patient-bootstrap intervals are available in the official [report](refine-logs/AXIS_FULL50_20260930/RESULTS_SNAPSHOT_20261001_1839/REPORT.md).

The development data were repeatedly exposed during research. These results do not establish independent external generalization. The internal test set was not used in this evaluation. The discarded O2M+NMS evaluation and the unselected overlap candidate are outside the current model version.

## Environment and reproduction scope

```bash
python -m pip install -r requirements.txt
```

The Ultralytics version is fixed to 8.4.83. Install a PyTorch build appropriate for the target CUDA environment. The other dependencies in `requirements.txt` are not a complete environment lockfile.

The original research code and frozen file hashes are preserved. `full50.py` and `run_pipeline.sh` are the original hpc6 experiment pipeline and depend on server-specific absolute paths, A0-R/M3 checkpoints, data manifests, label caches, and GPU configuration. Cloning this repository alone is insufficient to rerun training. The launcher also manages a server GPU holder and is intended for the original environment. Reproduction elsewhere requires preparing the inputs listed in the protocols and adapting the execution paths explicitly.

To retrieve the checkpoints:

```bash
git lfs install
git lfs pull
```

To verify the downloaded assets on a system with `sha256sum`:

```bash
sha256sum -c SHA256SUMS
```

## Dataset metadata

The original final data index is `/hpc/zhou228/x-ray/Dataset/Simplified_dataset/All_data.csv`. Its SHA256 is `a44d2e86b847da005a6b4ede1135606fa79f2aa0d08bd0652a377e2713c75b50`. The included CSV contains 31,124 rows covering 18,438 unique images: 27,749 positive bounding-box rows and 3,375 `no_finding` rows. CSV metadata does not include the corresponding images.

CSV columns: `id`, `type_of_abnormalities`, `x_min`, `y_min`, `x_max`, `y_max`, `source`, `width`, and `height`.

## Upstream dependency

This code depends on Ultralytics. Its use is subject to the upstream project's license terms.
