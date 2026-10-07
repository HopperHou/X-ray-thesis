# AXIS: Chest X-ray Object Detection

当前硕士论文模型的源码与正式实验记录。当前版本为 **AXIS E50 / O2O**，基于 YOLO26，使用三种子（42、43、44）Plan B 配对训练；seed42 E50 是预先指定的代表模型。

## 内容

- `implementation/axis_full50/`：模型、质量头、损失函数、训练、评价及统计源码，包含原始冻结协议和 SHA256 清单。
- `refine-logs/AXIS_FULL50_20260930/RESULTS_SNAPSHOT_20261001_1839/`：正式 E50 报告、逐类结果、收敛 CSV、评价矩阵及配对/冻结检查记录。
- `dataset/`：仅允许上传 `.csv` 文件；不上传原始影像、DICOM、YOLO 标签文本或数据集压缩包。
- `weights/`：供训练后的模型权重使用，独立于数据集目录。

**当前上传状态：本地没有训练权重或最终数据集 CSV，因此本次源码上传尚不包含这些文件。** 原始文件在 hpc6；连接服务器并下载、核验后才能补充。

## 模型

AXIS 只训练 O2O 框回归塔和 Q-head；backbone、neck、O2M 分支与 O2O 分类塔冻结。推理使用原生 O2O Top300，无 NMS，质量分数校正保持 identity。

几何项采用固定的 `axis_center_v1`：

```text
L_AXIS = B * L_box_native + L_Q + B * lambda_geo * L_geo_axis
L_geo_axis = mean_fg[(2 - 2^IoU)
                    + 0.5 * (abs(dx) / max(GT_width, stride)
                             + abs(dy) / max(GT_height, stride))]
lambda_geo = 0.6070424318313599
```

分类顺序：Pneumonia、Pneumothorax、Cardiomegaly、Aortic enlargement、Pleural thickening、Pulmonary fibrosis。

## 正式结果

在固定的 1,660 张 development 图像上，AXIS 三种子均值 mAP50–95 为 **27.567866%**，A0-R 为 **27.355653%**；差值为 **+0.212213 个百分点**。AXIS 相对 B2 的均值差为 −0.000028 个百分点，尚不支持 axis 项具有独立 mAP 增益。完整数值与条件患者 bootstrap 区间见正式 `REPORT.md`。

这些结果来自已反复使用的 development 数据，不能替代独立外部验证。内部 test 未用于本轮评价。后续废弃的 O2M+NMS 评价和未采用的 overlap 候选不属于当前模型版本。

## 环境与复现范围

```bash
python -m pip install -r requirements.txt
```

本仓库保留原始研究源码，不修改冻结方法或原始文件校验值。`full50.py` 和 `run_pipeline.sh` 是原 hpc6 实验流水线，依赖服务器绝对路径、A0-R/M3 权重、数据清单、标签缓存和 GPU 环境；克隆后不能直接作为通用训练命令运行。启动脚本还包含服务器 GPU holder 管理，仅适用于原环境。完整重跑需先准备协议中列出的输入并规划路径迁移。

最终数据索引原路径为 `/hpc/zhou228/x-ray/Dataset/Simplified_dataset/All_data.csv`，历史核验 SHA256 为 `a44d2e86b847da005a6b4ede1135606fa79f2aa0d08bd0652a377e2713c75b50`。数据集含 18,438 张图像的索引与 27,749 个阳性框；CSV 上传不包含相应影像。

源码依赖 Ultralytics；其使用与许可条件遵循上游项目。
