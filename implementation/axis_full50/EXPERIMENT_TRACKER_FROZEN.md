# B2 / AXIS 50-epoch 完整训练 tracker

**状态：PLAN_ONLY。** 预算默认推荐 Plan B；在任何 GPU 训练前确定 A 或 B。E50 为唯一主 checkpoint。以下是执行顺序，不表示已运行。

| ID | 阶段 | 工作 | Plan A | Plan B | 状态 |
|---|---|---|---|---|---|
| F00 | P0 | M3/A0 来源、SHA、患者划分、环境和预算冻结 | 必须 | 必须 | TODO |
| F01 | P1 | CPU 公式、stride、冻结、原路径、评价 toy 检查 | 必须 | 必须 | TODO |
| F02 | P2 | seed42 B2 从 M3 E0 训练 E1–E50 | 必须 | 必须 | TODO |
| F03 | P2 | seed42 AXIS 从同一 M3 E0 训练 E1–E50，并核对 4,150 step | 必须 | 必须 | TODO |
| F04 | P2 | seed43 B2/AXIS 同样成对训练 | 不运行 | 必须 | TODO |
| F05 | P2 | seed44 B2/AXIS 同样成对训练 | 不运行 | 必须 | TODO |
| F06 | P3 | 同一官方 evaluator 重评 A0-R 与所有 B2/AXIS E50 | 必须 | 必须 | TODO |
| F07 | P4 | 三组差值、全部 seed、逐类/IoU、患者 bootstrap、训练收敛 | 必须 | 必须 | TODO |
| F08 | P4 | 已暴露 test 的三方同协议描述性评价；不参与选择 | 可选且事先冻结 | 可选且事先冻结 | TODO |
| F09 | P5 | 最后一个 GPU 任务结束后恢复 computation.py 并核对状态 | 必须 | 必须 | TODO |
| F10 | 收尾 | 写机器状态与 REPORT；即使 AXIS≤B2 也完整报告 | 必须 | 必须 | TODO |

**主证据文件：** `protocol.json`、`source_hashes.json`、`sanity.json`、每臂/seed `input_hashes.jsonl` 与 `epoch_50.pt`、`pairing_check.json`、`evaluation_check.json`、各 `eval_result.json`、`comparison_summary.json`、`bootstrap_summary.json`、`REPORT.md`、`completion_status.json`、`gpu_holder_restore_status.json`。
