# Native Regression and Numerical Treatment

For matched index $i$, decoded prediction $\hat b_i$ corresponds to reference box $b_i^*=(x_{1i}^*,y_{1i}^*,x_{2i}^*,y_{2i}^*)$, with width $w_i^*=x_{2i}^*-x_{1i}^*$ and height $h_i^*=y_{2i}^*-y_{1i}^*$. Actual minibatch size $B$ scaled regression and geometry once, including the final 174-image batch. Q used a per-image mean with $\lambda_Q=1$.

Six-class YOLO26x used Ultralytics 8.4.83 ([Ultralytics v8.4.83 source code (GitHub)](https://github.com/ultralytics/ultralytics/tree/v8.4.83)) with end-to-end detection. The detector used prediction strides 8/16/32 and emitted four direct distances (left, top, right and bottom) and six class logits per location. Final fine-tuning and evaluation used its O2O branch. Only the regression tower and Q heads were trainable, and only regression-tower batch-normalisation statistics updated.

TaskAlignedAssigner used initial/secondary top-k 7/1 and $\alpha=0.5$, $\beta=6$, combining detached original class probabilities with detached decoded-box non-negative CIoU. Native candidate/conflict resolution and secondary filtering retained at most one positive location per GT. Candidate selection expanded GT dimensions below the smallest stride to 16 pixels without changing regression targets. Both native regression terms used soft target-score weighting and total-score normalisation floored at one. With $\mathrm{reg_max}=1$, no distribution focal loss applied, and classification BCE was diagnostic only. The CIoU and direct-distance L1 gains remained 7.5 and 1.5, respectively, as specified in the following equations.

Let F be native O2O foreground locations, t the soft class targets, s the native stride, and B the actual batch size. Define regression weights and normalisation over all prediction locations I by

$$
a_i=\sum_{c=1}^{6}t_{ic},
\qquad
T=\max\left(\sum_{i\in\mathcal I}\sum_{c=1}^{6}t_{ic},1\right).
$$

For decoded input-pixel boxes $\hat b$ and $b^*$, and predicted/target distance vectors $\hat d$ and $d^*$ in feature-grid units,

$$
\mathcal L_{\mathrm{CIoU}}
=\frac{1}{T}\sum_{i\in\mathcal F}
a_i\left[1-\mathrm{CIoU}\left(\hat b_i/s_i,b_i^*/s_i\right)\right].
$$

$$
\mathcal L_{\mathrm{dist}}
=\frac{1}{T}\sum_{i\in\mathcal F}
\frac{a_i}{4}\left\|D_i(\hat d_i-d_i^*)\right\|_1,
\qquad
D_i=\operatorname{diag}\left(
\frac{s_i}{W_{\mathrm{in}}},\frac{s_i}{H_{\mathrm{in}}},
\frac{s_i}{W_{\mathrm{in}}},\frac{s_i}{H_{\mathrm{in}}}
\right).
$$

$$
\mathcal R_{\mathrm{native}}
=7.5\mathcal L_{\mathrm{CIoU}}+1.5\mathcal L_{\mathrm{dist}}.
$$

Input width and height are $W_{\mathrm{in}}$ and $H_{\mathrm{in}}$; the direct-distance term applies to reg_max = 1. Native target-score weighting and stabilisation were retained.

AXIS overlap used non-negative areas, union floor $10^{-12}$ and IoU clipping to [0, 1]. Centre terms used FP32 re-decoding; overlap retained the native decoded-pixel path. Predictions remained differentiable, GT dimensions and strides were detached, and empty foregrounds returned graph-connected zero. Positive centre denominators used stride floors without extra epsilon.
