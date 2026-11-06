# Exploratory Ablation and Evaluation Provenance

O_ONLY retained the overlap term $O=2-2^u$. O_ISOTROPIC added centre displacement normalised by one shared scale. For input-pixel offsets $\Delta x$, $\Delta y$ and detached GT dimensions $w^*$, $h^*$, the original axis-normalised component is defined in Section 3.2.3.

$$
\begin{aligned}
D_{x,i}&=\max(w_i^*,s_i), & D_{y,i}&=\max(h_i^*,s_i), & q_i&=\sqrt{D_{x,i}D_{y,i}},
\end{aligned}
$$

$$
\begin{aligned}
C_{\mathrm{axis},i}&=\frac{1}{2}\left(\frac{|\Delta x_i|}{D_{x,i}}+\frac{|\Delta y_i|}{D_{y,i}}\right),\\
 C_{\mathrm{iso},i}&=\frac{|\Delta x_i|+|\Delta y_i|}{2q_i}.
\end{aligned}
$$

$$
G_{\mathrm{O_ONLY}}=\operatorname{mean}_{i\in\mathcal F}O_i,
 \qquad
 G_{\mathrm{O_ISOTROPIC}}=\operatorname{mean}_{i\in\mathcal F}(O_i+aC_{\mathrm{iso},i}).
$$

Calibration used the first four historical seed-43 minibatches, each containing 180 images, totalling 720 distinct training images. At frozen M3, the centre-only gradient L2 norms of 180$\lambda_{\mathrm{geo}}$ mean(C) were measured over the 24 O2O regression tensors. Matching compared the original axis-normalised C against Ciso. BN buffers were restored before corresponding forwards; BF16 AMP used FP32 centres and CPU FP64 norm accumulation. No optimiser update or validation/test input was used.

$$
\begin{aligned}
g_{\mathrm{axis},k}&=\left\|\nabla_\theta\left[180\lambda_{\mathrm{geo}}\operatorname{mean}_{\mathcal F_k}C_{\mathrm{axis}}\right]\right\|_2,\\
 g_{\mathrm{iso},k}&=\left\|\nabla_\theta\left[180\lambda_{\mathrm{geo}}\operatorname{mean}_{\mathcal F_k}C_{\mathrm{iso}}\right]\right\|_2,
\end{aligned}
$$

Here $\theta$ comprises the 24 O2O regression parameter tensors and $\mathcal F_k$ contains batch $k$ foregrounds.

The median of four axis/isotropic norm ratios, averaging the middle two, gave a = 0.9108724655881961. This multiplier remained frozen; it does not imply equal gradient vectors or per-batch norms. Production-integration gradient checks covered only the first two batches, with relative L2 error zero. `original/CALIBRATION.json` and `CALIBRATION_SAMPLE_LIST.csv` retain norms and sample order, whose four ordering hashes were recovered by CPU index-only reconstruction.

B2/AXIS were trained on NVIDIA H100 PCIe; the new variants used NVIDIA A100 80GB PCIe. All used seed 43, shared M3 box/Q states, batch 180 with final batch 174, 40 workers and BF16 AMP. Every checkpoint was fixed E50 after 4,150 updates. All four input streams matched throughout training. The A100 environment used Python 3.10.10, PyTorch 2.11.0+cu126, CUDA 12.6 and Ultralytics 8.4.83.

All four checkpoints were evaluated on A100 using FP32, batch eight and ten workers on both partitions, following Section 3.2.6 with input-frame box matching. Ordered images, input hashes, canvases and GT arrays matched across configurations. Cached-record AP matched all eight validator matrices within absolute tolerance $10^{-12}$.

Detailed A100 backend settings are recorded in `ENVIRONMENT_AND_PROTOCOL.json`. Historical Python, cuDNN and backend-flag records were incomplete, preventing exact environment equivalence.

Exact split lists and dataset-manifest hashes accompany SA10; dataset membership and grouping are described in Section 3.1.4.

| Partition | Model | Metric | Historical | A100 | $\Delta$ (pp) |
| --- | --- | --- | --- | --- | --- |
| Validation | B2 | mAP50–95 | 27.58584 | 27.58482 | $-0.00103$ |
|  | AXIS | mAP50–95 | 27.58917 | 27.58969 | $+0.00052$ |
|  | B2 | mAP50 | 53.30607 | 53.30419 | $-0.00188$ |
|  | AXIS | mAP50 | 53.30778 | 53.30589 | $-0.00189$ |
| Internal test | B2 | mAP50–95 | 27.79536 | 27.82431 | $+0.02895$ |
|  | AXIS | mAP50–95 | 27.80762 | 27.83261 | $+0.02499$ |
|  | B2 | mAP50 | 53.05384 | 53.15680 | $+0.10296$ |
|  | AXIS | mAP50 | 53.07168 | 53.17135 | $+0.09967$ |

Historical seed-43 evaluations versus the common-A100 re-evaluation of identical fixed-E50 checkpoints.

Table A.5 compares identical checkpoints under historical and A100 evaluation. Test batch/workers changed from 2/8 to 8/10; validation batch remained eight, with workers increasing from eight to ten. Historical canvas/backend records are incomplete; no single-factor replay was performed. Seed 43 was selected after the historical results were known.

SA10 (`SA10-AXIS-SEED43-E50-20261030`) provides checkpoint hashes, environment and data manifests, calibration records, and pairing/AP checks. Raw results are `original/E2_SEED43_RESULTS.csv` and `original/E2_CLASS_IOU_RESULTS.csv`. Radiograph pixels, weights and prediction caches are excluded; public checks verify exported records rather than independently reproducing inference.

[Seed-43 component-ablation records (GitHub)](https://github.com/HopperHou/X-ray-thesis/tree/81d585e01480ccafe99ac60ebf1806db7b84758c/supplementary/seed43-component-ablation)
