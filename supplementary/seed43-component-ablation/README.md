# Appendix A.10: exploratory seed-43 component ablation

Supplementary identifier: **SA10-AXIS-SEED43-E50-20261030**.

This directory preserves the four fixed-E50 component comparison reported in
Table 4.1 and its provenance. B2 and AXIS are historical H100 checkpoints;
O_ONLY and O_ISOTROPIC were trained on an A100. All four were evaluated on the
same A100 with FP32, batch 8 and 10 workers. The source experiment completed on
2026-10-23 UTC. Evidence was collected read-only on 2026-10-30 (New Zealand time),
with CPU-only arithmetic verification and no new training or model inference.

## Evidence inventory

| File or directory | Contents |
| --- | --- |
| `original/E2_SEED43_RESULTS.csv` | Unrounded model metrics and pairwise point differences, in percentages or percentage points as indicated. |
| `original/E2_CLASS_IOU_RESULTS.csv` | All 480 class-by-IoU AP entries (fractions): 2 partitions × 4 configurations × 6 classes × 10 thresholds. |
| `original/CALIBRATION.json` | Original four-batch gradient norms, ratios, frozen scale and production-integration checks. |
| `CALIBRATION_SAMPLE_LIST.csv` | All 720 calibration images in exact batch/order, original server paths, ordering and augmented-input hashes. Order was reconstructed using the original seed-43 index stream and checked against all four recorded ordering hashes. Image pixels/augmentations were not regenerated during collection. |
| `CALIBRATION_DERIVED.json` | Analytically scaled norms; these are derived numbers, not new backward passes. |
| `CHECKPOINTS.json` | Four exact server paths, SHA-256, GPU models, epoch/update counts and initial-state hashes. Checkpoint bytes were rehashed on the server. |
| `training/{configuration}/` | Original status and all 4,150 per-update input-hash records for each of the four configurations. |
| `original/PAIRING_VERIFICATION.json` | Original complete-stream pairing verdict. |
| `original/AP_VERIFICATION.json` | Original eight-matrix export/native comparison. |
| `COLLECTION_VERIFICATION.json` | Fresh server verification: four checkpoint hashes, calibration-order recovery, cached-prediction/trace hashes, identical evaluation inputs/GT, and AP recomputation from stored prediction/TP arrays. All eight maximum AP discrepancies were zero. |
| `evaluation/{partition}/{configuration}/` | Original task-completion provenance and full validator AP matrices. |
| `historical_evaluation/` | Original B2/AXIS validation and internal-test JSON results/AP matrices. |
| `HISTORICAL_VS_A100.csv` | Exact old/new metrics and new-minus-old percentage-point differences. |
| `ENVIRONMENT_AND_PROTOCOL.json` | Recovered software, hardware, numerical settings, fixed-E50 rule, confirmed protocol changes and unresolved causal attribution. |
| `EVALUATION_IMAGE_MANIFEST.csv` | All 3,504 evaluation records in actual order, image names/source, label/input SHA-256, canvas dimensions and GT counts. One common manifest applies to all four models after exact cross-model comparison. |
| `lists/` | Exact original training, validation and test path lists. Paths document provenance; image pixels are not distributed here. |
| `DATA_FILES.json` | Public dataset metadata paths and hashes, split counts and exact path-list hashes. |
| `original/*.py`, `historical/` | Original geometry, integration/calibration/evaluation code and selected historical protocol records. Server paths in these files are provenance and require adaptation outside that environment. |
| `SHA256SUMS.json`, `verify.py` | Release integrity manifest and read-only standard-library verifier. |

The frozen annotations and image/partition linkage are in the repository's
[`dataset/`](../../dataset/README.md), fixed at commit
`288397aaac2283fb2be50982d60c43f1c5bfcbc9`. They include `All_data.csv`,
`new_train.csv`, `new_validation.csv`, `new_test.csv`, `dataset_manifest.csv`
and `patient_level_split_image_assignments.csv`. The counts are 14,934/1,660/1,844
training/validation/test images and 22,477/2,502/2,770 boxes. Dataset grouping
limitations, including singleton fallback groups, are documented there.

## Definition and scope

For detached GT dimensions and stride, let Dx=max(w*,s), Dy=max(h*,s).
O_ISOTROPIC uses the single shared scale sqrt(Dx*Dy), giving
Ciso=(abs(dx)+abs(dy))/(2*sqrt(Dx*Dy)). It retains the native overlap component
O=2-2^IoU and adds a*Ciso, with frozen a=0.9108724655881961. The previous
lambda_geo=0.6070424318313599 is unchanged and is a separate coefficient.

The multiplier is the median of four ratios of L2 norms of gradients of
180*lambda_geo*mean(C), with respect to the 24 O2O regression parameter tensors,
at the frozen M3 state. The four minibatches are the first historical seed-43
training minibatches (180 images each). There were no optimizer updates and no
validation/test data. M3 BN buffers were restored before corresponding forwards;
BF16 AMP forwards used FP32 centre decoding and CPU FP64 norm accumulation.
This matches the median gradient-norm ratio at calibration, not every batch's
gradient vector or subsequent training. The original AXIS integration comparison
covered only the first two batches (relative gradient L2 error zero).

The 4 × 4,150 input records match after excluding only the configuration name.
They bind image order, analysis-group and eligibility order, actual batch size,
and augmented image/target tensors. Initial box and Q states also match. These
checks establish paired inputs; they do not establish identical floating-point
optimization across H100/A100 or identical evolving native assignments.

## Historical results: confirmed settings, unresolved causes

The checkpoints are unchanged. Historical validation used batch 8/workers 8;
historical internal test used batch 2/workers 8. New evaluation used batch
8/workers 10 on an A100 for both partitions, versus the historical H100 runs.
FP32, nominal size 1024, identity O2O scoring, two Top-300 stages, strict score
>0.001 and no NMS were retained.

Native rectangular canvases depend on batch composition. This and numerical
hardware/kernel differences are plausible mechanisms, **not experimentally
isolated causes**. The historical traces do not contain input hashes/canvas
dimensions, and contemporaneous historical cuDNN/backend flags and Python
version were not recovered. Validation already used batch 8. Attribution of
the exact metric changes remains **unresolved**; no single-factor replay was
performed. Historical main results must remain distinct from this common-A100
exploratory comparison.

These are single-seed, fixed-E50 point estimates, without between-seed uncertainty.
The seed was chosen after historical seed-specific outcomes were known.
There is no centre-only arm, so this is not a complete factorial experiment and
does not identify component interaction, stable superiority or equivalence.

## Verification and availability boundary

From this directory run `python verify.py`. It checks release hashes, all four
training streams, calibration sample ordering and ratio arithmetic, 480 exported
AP cells/aggregate metrics, evaluation manifest counts and historical differences.
The fresh server checks are recorded separately in `COLLECTION_VERIFICATION.json`.

This is a public metadata and numerical-evidence supplement. It does **not**
contain radiograph pixels, the four checkpoint weight files, augmented tensors,
or raw prediction NPZ caches. Exact original server checkpoint paths and hashes
are identifiers, not public download URLs. The AP-from-prediction recomputation
therefore remains a recorded server-side verification, not a claim that this
small public supplement alone reproduces model inference. No new licence or
permission for third-party radiographs is asserted.
