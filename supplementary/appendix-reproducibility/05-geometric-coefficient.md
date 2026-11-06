# Origin of the Fixed Geometric Coefficient

The inherited coefficient was calibrated before M3/E50 using four training images and zero optimiser updates. The ordered manifest was filtered for original 1024 × 1024 images with annotations under size-stratum key `s<=32`, distinct from native stride. Among at most 16 eligible rows, the first four with foregrounds and positive norms were retained. Unaugmented RGB images formed singleton batches; all matched foregrounds contributed.

The fixed A0 detector ran on CPU with Q heads installed and the O2O regression tower trainable. Training-mode forwards allowed regression-tower BN updates. For the earlier squared diagonal-relative objective,

$$
G_{\mathrm{old}}=\operatorname{mean}_{i\in\mathcal F}
\left[(2-2^{u_i})+
\frac{\|\hat{\boldsymbol c}_i-\boldsymbol c_i^*\|_2^2}
{\max((w_i^*)^2+(h_i^*)^2,10^{-12})}\right].
$$

$$
r_k=\frac{\|\nabla_\theta\mathcal R_{\mathrm{native},k}\|_2}
{\|\nabla_\theta G_{\mathrm{old},k}\|_2},
\qquad
\lambda_{\mathrm{geo}}=0.1\,\operatorname{lmed}_{k=1}^{4}(r_k),
$$

Measured ratios were converted to FP32 before median selection and multiplication; torch.median selected the lower middle value. The stored coefficient was 0.6070424318313599. The factor 0.1 defined the nominal calibration scale. This coefficient remained fixed after adopting AXIS centre normalisation and is distinct from the isotropic multiplier in Appendix A.6. Original calibration records are `v6a2_calibrate.py` and `P1_GEO_GRADIENT_SCALE_CPU.json`.
