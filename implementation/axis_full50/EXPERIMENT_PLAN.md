# B2 / B3-axis 50-epoch 成对完整训练：冻结实验计划

**状态：FROZEN_PLAN_ONLY。** 本文件只定义最终论文模型训练；未编写新训练代码、未读取新结果、未启动训练。方法已固定为 B3-axis（以下称 AXIS）。本轮不是新公式搜索，也没有「AXIS 必须优于 B2 才能继续」的开发式 GO gate。

**研究问题：** 在完全相同的 50-epoch 训练预算和官方评估路径下，(A) AXIS 完整系统相对同机 A0-R 的表现如何；(B) AXIS 相对无几何辅助项的 B2 有无额外贡献；(C) B2 相对 A0-R 的收益是多少。主指标为六类宏平均 mAP50–95。当前 development 已反复用于方法选择，现有 test 也曾暴露；任何结果都是本研究数据上的内部/描述性证据，不能称为 untouched 外部验证。

## 1. 论文主张与比较责任

| 比较 | 它回答什么 | 无论结果方向如何都必须报告 |
|---|---|---|
| AXIS − A0-R | 完整 proposed model 是否超过原始检测基线 | 三种子逐一差值、均值、患者配对不确定性；不能全部归因于 AXIS 几何项 |
| AXIS − B2 | AXIS 几何项在相同原生 O2O 微调之外的增量 | 可为正、零或负；负结果不得删除 AXIS arm 或改写新模块贡献 |
| B2 − A0-R | 不加新几何项时，原生 O2O 框塔微调贡献多少 | 防止把 B2 的收益转记为几何项或质量头评分 |

不以质量头或 Q scorer 提升 AP 为论文主张：三路质量头继续按旧协议训练，但训练输入对共享/box 特征 detach；正式 scorer 为 identity，推理分数为原分类分数。原 B3、v2/v3/v4、E8 AXIS 和 H1/H2c/R0 等只作已有消融与负结果证据；不在这次完整训练中新增 arm。无 LLM/VLM 等 frontier 组件，不设置与此无关的比较。

## 2. 唯一冻结方法、共同起点与优化目标

所有 B2/AXIS 运行**独立从同一 M3 full-Q E0 checkpoint 重建模型、优化器、数据加载器与随机状态**，绝不从 E8 pilot 或历史 B2/B3 checkpoint 续训。M3 E0 指 B2/AXIS 微调阶段的 epoch 0：质量头此前已在 A0-R 基础上完成 M3 训练。预期 M3 文件：

`/hpc/zhou228/x-ray/Trials/P2_P6_refiner/A0R_O2O_V6A2_MapFirst/runs/m3_full_q/final.pt`

预期 SHA256：`c20481309083b51b68111922c33e68250595cbac2f68018369cb26edf69a2d90`。执行前必须对实际文件重算 SHA，不以文档字符串代替核查。

对每个实际训练 batch，沿用原生 O2O assigner 的 foreground、assigned GT、target scores 和原生框损失。AXIS 唯一增加的训练项是：

\[
L_{geo}^{axis}=\operatorname{mean}_{i\in fg}\left[(2-2^{IoU_i})+\frac12\left(\frac{|dx_i|}{\max(w_{GT,i},stride_i)}+\frac{|dy_i|}{\max(h_{GT,i},stride_i)}\right)\right],
\]

其中 `dx,dy` 是同一增强后像素坐标下预测框与 assigned GT 的中心差；stride 来自该 foreground slot 的原生尺度。空 foreground、FP32 中心计算、GT 合法性、shape 对齐及 overlap 原路径严格沿用已通过的 AXIS E8 P1 实现。不新增 clamp、平滑、IoU 权重、系数或另一种 assignment。

记 `B` 为**当前实际 batch size**，`L_native^O2O` 为旧实现按原 gain/target-score weighting 计算的 CIoU 与归一化直接距离 L1 框项（本模型 `reg_max=1`，不是 DFL 分箱），`L_Q` 为旧连续质量 BCE：

\[
L_{B2}=B\,L_{native}^{O2O}+L_Q,\qquad
L_{AXIS}=B\,L_{native}^{O2O}+L_Q+B\,(0.6070424318313599)\,L_{geo}^{axis}.
\]

`lambda_q=1`。实际 batch 系数只在 caller 乘一次；末批为 174 而不是 180。分类 loss 只按旧协议计算/记录，不把完整 E2E total loss 重新定义为优化项。仅 `one2one_cv2` 框塔与旧 Q heads 可训练；backbone、neck、O2M、`one2one_cv3` 分类塔保持冻结。BN buffer 行为、O2O `x.detach()`、AdamW 参数组与 E8 pilot 完全一致。推理继续用原 one2one 分类 logits、identity scorer、无 NMS、第一次 Top300、第二次 Top300、严格 `score>0.001`；不改网络、assigner、architecture、inference 或任何其他 threshold。

## 3. 固定数据、资源与成对规则

| 项目 | 冻结值 |
|---|---|
| Training | 原 training allowlist 14,934 图、8,896 患者组，原 GT 与 E/P/U；1024×1024；同一 patient split；不读 development/test 做训练选择 |
| Development | 原 1,660 图、990 患者组；仅作训练完成后的官方主评价；训练患者与其交集必须为零 |
| 已暴露 test | 原 1,844 图，仅在全部公式、训练和主评价规则冻结后，对 A0-R/B2/AXIS 同协议作描述性评价；不得选模型或改方法 |
| 每 seed 配对 | B2 与 AXIS 使用同一 seed、患者和图像顺序、相同增强随机流、GT/E-P-U、actual batch size；每一个对应 optimizer step 的输入哈希一致 |
| 训练 | global batch 180，workers 40，bf16 autocast，AdamW；box LR `1e-5`、Q LR `1e-4`、weight decay `5e-4`、constant LR；50 个完整 epoch |
| 增强 | 沿用 AXIS E8 pilot 的单图 HSV、affine、水平翻转；mosaic/mixup/cutmix/copy-paste 均为 0；不因训练时长改变 |
| 步数 | 每 epoch `ceil(14934/180)=83` step，末批 174；每臂 `50×83=4,150` optimizer steps；不增不减 |
| 资源 | hpc6 physical GPU0、原 UUID `GPU-bdf1c899-1408-3f77-05c1-4584e658333d`；GPU holder `computation.py` 必须在所有训练/评价 GPU 工作完成后才恢复并记录状态 |

逐 step hash 至少包含：`seed,arm,epoch,step,actual_batch_size,ordered_patient_ids,ordered_image_ids,eligibility,GT class/box tensor shape+dtype+bytes,augmented_input shape+dtype+bytes`。比较时不把 `arm` 本身纳入共同输入 digest；另存 arm 身份。每 seed 的 B2 与 AXIS 共 **4,150 个对应 step** 必须完全一致。模型分叉后 native assignment、预测及 loss 不要求相同；用它们作配对条件会错误拒绝有效实验。增强及数据顺序应按 seed 预先冻结/重放，不靠两次运行“碰巧相同”。

Plan A 或 B 必须在**第一步训练前**确定；若采用 Plan B，不根据 seed42 的 AP 决定是否启动 seed43/44。任何缺失 seed 或配对失败均如实标为不完整/INVALID，不临时降低要求。OOM、NaN/Inf、错误梯度、来源冲突、不可精确恢复的中断属于执行问题，**不是方法负结果**；不得为了获得结果即兴更改 batch、LR、epoch、lambda 或评价规则。若有完整 model/optimizer/scheduler/AMP/RNG/DataLoader 恢复状态，可从同一已核验 step 恢复；否则受影响的配对 seed 两臂从 M3 E0 重新开始。

## 4. 两个预算版本与推荐

| 版本 | 新训练 | 训练步数 | 基于 E8 实测约 2,460 秒/臂的粗估训练时间* | 证据强度 |
|---|---|---:|---:|---|
| **Plan A** | seed42：B2 + AXIS，各 E1–E50 | `8,300` | 约 8.5 GPU 小时 | 单种子完整训练；能回答本次固定轨迹，不能判断随机种子稳定性 |
| **Plan B（推荐）** | seed42/43/44：各有 B2 + AXIS，各 E1–E50 | `24,900` | 约 25.6 GPU 小时 | 三组成对差值、种子均值/标准差；更适合硕士论文最终实验 |

\*仅把此前 E8 的 2,451–2,470 秒/臂线性放大；未计环境准备、哈希、checkpoint 写入、三臂官方评价、数据拥塞及故障重跑，不是 ETA 保证。

**推荐 Plan B。** 单种子历史 B3−B2 在 E8/E15 出现方向变化；AXIS E8 的正差仅约 `+0.0172 pp`，因此随机种子稳定性是论文主结论的必要背景。若资源只能支持 A，应在运行前选定 A，并把结论严格限定为 seed42；不得先看 A 的开发集 AP 再决定是否补 seed。无论 A/B，AXIS 都是预先固定的最终候选，不因为 B2 胜出而增补新公式。

## 5. checkpoint、评价与统计规则

**唯一 primary checkpoint 为每个 arm/seed 的 E50。** 完整跑满 E1–E50；禁止 best epoch、E15/E20/E50 事后择优、按中途 AP 早停或修改训练。E1–E49 记录每 epoch training loss、native/geo/Q 分量、`N_fg`、finite/梯度摘要和耗时以展示 convergence。预先保存 E10/E20/E30/E40/E50 的等距 checkpoint，可在全部训练结束后对同一 development 重放作**描述性收敛曲线**；这四个早期点绝不参与最终模型选择。若存储不足，保留完整训练日志与 E50，预先在执行协议中统一裁掉两臂相同的早期 checkpoint，不依 AP 方向决定。代表性交付模型预先定为 seed42 E50；总体结果报告所有 seed，不挑最好 seed。

训练全部完成并通过配对后，用**同一版本、同一配置的官方 evaluator**重新评价固定 A0-R 与全部 B2/AXIS E50 checkpoint。A0-R 若 checkpoint SHA、模型结构、数据和评估协议可严格核实，无需重新训练。每次评价保存完整 `AP[6 classes,10 IoU thresholds=.50:.05:.95]`，按相同六类顺序重算：

- 主指标：六类宏平均 `mAP50–95`；原始值存 fraction，展示才转百分数。
- 固定次指标：AP50、AP75、AP80、AP85、AP90、AP95、`H=mean(AP75,AP80,AP85,AP90,AP95)`、逐类 AP50/mAP50–95、完整逐类×IoU AP。
- 比较：每 seed 的 `AXIS−A0-R`、`AXIS−B2`、`B2−A0-R`，并报告 Plan B 的 seed 均值、样本标准差、所有原始值；不能只报最有利的种子或阈值。
- 患者级 paired uncertainty：对同一 development 的 990 个患者组整组有放回抽样，固定 PCG64 seed `20260930` 做 10,000 次 bootstrap；每次纳入患者的全部图像和预测，重新计算官方 AP 后取差值的 2.5/97.5 分位。Plan B 每 seed 各报一次，同时按相同患者重采样计算三 seed **平均差值**的条件区间；它只描述该已暴露 development、已冻结模型下的采样不确定性，不校正模型选择偏差，也不是外部泛化证明。

**解释规则，而非继续/停止 gate：** 无论 `AXIS−B2` 正负，完成有效实验并报告全表。若 AXIS−A0-R 在三 seed 方向一致且条件区间为正，可以写“在该同机、已开发数据上表现稳定优于 A0-R”；若不一致或区间跨零，只写观测数值与不确定性。若 AXIS−B2 未显示正增量，论文将其几何项写成固定 proposed model 中**未获独立性能支持**的构件，不将系统整体收益归因于它。Plan A 不得声称跨种子稳定。任何版本都不得把已暴露 test 或 development 称为 untouched confirmation。

## 6. 执行顺序、有效性与输出合同

| 阶段 | 必须完成的事 | 主要输出 | 失败含义 |
|---|---|---|---|
| P0 来源冻结 | 核对实际 M3/A0 SHA、训练/开发患者、GT/E-P-U、源码/环境、GPU UUID、先前 AXIS 公式与 λ；固定 A 或 B、seed、代码/协议 SHA | `protocol.json`、`source_hashes.json`、`p0_provenance.json` | 来源不符则 INVALID，不训练 |
| P1 CPU correctness | 复用 AXIS E8 的 toy、stride 对齐、FP32/AMP finite、原 B2 路径身份、actual-batch 系数、可训练参数及各层 BN 行为、推理身份；不做新机制探针 | `sanity.json`、`freeze_check.json` | FAIL 则 INVALID，不训练 |
| P2 成对训练 | 每 seed 两臂从 M3 E0 独立 E1–E50；每 step 比较输入 hash；全部 step、epoch、loss finite | `runs/{seed}/{B2,AXIS}/status.json`、`train_metrics.jsonl`、`input_hashes.jsonl`、`epoch_50.pt`、`pairing_check.json` | 任一缺失/不配对/异常为执行无效，不作方法结论 |
| P3 官方评价 | 仅主 checkpoint E50 参与主结果；A0-R/B2/AXIS 同 evaluator、同数据、同 scorer/Top300/score floor；可在全部训练后生成预定轨迹描述 | 每臂 `eval_result.json`、`per_iou_ap.json`、`evaluation_check.json` | 不完整或无法从 AP 矩阵复算则 INVALID |
| P4 统计和论文表 | 三组 primary comparisons、每 seed/均值、患者 bootstrap、收敛图、已暴露 test 的统一描述性评价（若执行则三方同协议） | `comparison_summary.json`、`bootstrap_summary.json`、`per_class.csv`、`convergence.csv`、`REPORT.md` | 全部方向如实报告，不产生新方法 GO gate |
| P5 收尾 | 复核全部 GPU 任务已结束后恢复 `computation.py device=0 mem=50` 并记录 GPU0 UUID/进程；不覆盖旧结果 | `gpu_holder_restore_status.json`、`completion_status.json` | 运维异常单列，不改已有效科学数值 |

机器状态建议仅区分 `PLAN_ONLY`、`RUNNING`、`VALID_COMPLETE`、`INVALID`、`INCOMPLETE`；`VALID_COMPLETE` 与 AXIS 是否胜出无关。旧 `STOP_AXIS_FULL_TRAINING` 是 E8 pilot 的历史判定，不能删除或改为 GO；本计划的 50-epoch 训练是用户重新定义的最终论文实验，不是旧 gate 自动通过。

## 7. 执行前冻结清单

- [ ] 明确选择 Plan A 或 Plan B；默认论文主实验为 B，选择发生在任何新 development AP 可见之前。
- [ ] M3 E0 重新计算实际 SHA；两臂/每 seed 都不是从 E8 checkpoint 续训。
- [ ] B2/AXIS 唯一差别是固定 AXIS 辅助项；`lambda_geo=0.6070424318313599`，模型/Q/scorer/assigner/inference 不变。
- [ ] batch180、workers40、50 epoch、每臂4,150步、末批174；每 seed 4,150个共同输入 hash 完全配对。
- [ ] E50 是唯一主 checkpoint；全部 seed 都报告，seed42 E50 是预定交付权重。
- [ ] A0-R、B2、AXIS 用同一官方 evaluator 完整重评；主指标与三组差值、患者级条件区间预先固定。
- [ ] development/test 已暴露；不再用它们改方法或宣称独立外部验证。
- [ ] 所有 GPU 任务结束后才恢复 holder；本计划阶段不启动任何 GPU 工作。
