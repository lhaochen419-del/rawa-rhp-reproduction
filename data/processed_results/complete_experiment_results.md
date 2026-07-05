# 完整实验运行结果

## 当前正式状态

本轮实验闭环已完成并通过验收：

- EqEval 预算匹配：通过。
- 核心消融 formal gate：通过。
- 主基线配对统计：通过。
- 严格 latency benchmark：通过。
- RAWA-RHP hard safety：battery violation、clearance violation、reserve shortfall、return failure、emergency abort 全为 0。
- NoRisk-Pure：仅保留为诊断项，不作为核心贡献模块。

## 正式核心消融

正式核心消融目录：

- `outputs/results/formal_core_merged_s1_20_v2/`

关键文件：

- `full_1_20_episodes.csv`
- `full_1_20_summary.csv`
- `formal_ablation_analysis.md`
- `formal_ablation_analysis.json`
- `manifest.md`

正式 full-mode 设置：

- seeds：1-20。
- 场景：ID、correlated gust OOD、narrow passage OOD。
- 风场：calm、moderate、severe。
- 障碍物密度：sparse、cluttered。
- 场景数：360。
- planner：RAWA-RHP + 5 个核心模块 × Native/EqTime/EqEval，共 16 个。
- episode：5760。
- 统计：按 scenario seed 聚类 bootstrap，配对比较 RAWA-RHP 与各消融，Holm 校正。

核心结果：

| 消融 | Native mean diff | EqTime mean diff | EqEval mean diff | 结论 |
|---|---:|---:|---:|---|
| NoPacking | 0.163950 | 0.163950 | 0.144629 | 通过 |
| NoBeam | 0.018054 | 0.018054 | 0.017139 | 通过 |
| NoRisk | 0.033474 | 0.033589 | 0.033138 | 通过 |
| BlindReserve | 0.041053 | 0.041053 | 0.041828 | 通过 |
| NoAdaptiveSearch | 0.046174 | 0.046174 | 0.028184 | 通过 |

全部正式消融均满足：

- mean diff >= 0.01。
- cluster bootstrap 95% CI lower > 0。
- Holm p < 0.05。
- 关键子集方向一致。
- 安全指标不劣化。
- EqTime/EqEval 预算 ratio 在 5% 容差内。

## EqEval 预算检查

最终 EqEval 预算检查基于统一计算量 `unified_eval_count`：

| 变体 | unified eval ratio | 预算状态 |
|---|---:|---|
| RAWA-NoPacking-EqEval | 0.999548 | 通过 |
| RAWA-NoBeam-EqEval | 0.970686 | 通过 |
| RAWA-NoRisk-EqEval | 0.953818 | 通过 |
| RAWA-BlindReserve-EqEval | 0.975441 | 通过 |
| RAWA-NoAdaptiveSearch-EqEval | 0.981772 | 通过 |

最终使用的校准文件：

- `outputs/results/budget_calib_rawa_core_full_s901_910/budget_calibration_eqeval_tuned_v11.json`

相关正式 EqEval 运行：

- `outputs/results/formal_core_eqeval_rerun_s1_20_v1/`
- `outputs/results/formal_core_eqeval_noadaptive_rerun_s1_20_v2/`
- `outputs/results/formal_core_eqeval_rerun_s1_20_v2_combined/`

## 主基线配对统计

主基线统计报告：

- `outputs/results/baseline_paired_comparison.md`
- `outputs/results/baseline_paired_comparison.json`

结果：

| 场景 | 最强基线 | RAWA-RHP mean | 最强基线 mean | mean diff | 95% CI | Holm p | 结论 |
|---|---|---:|---:|---:|---|---:|---|
| blind100 | FairALNSPlanner | 0.843491 | 0.698748 | 0.144743 | [0.130698, 0.159109] | 0.000150 | 通过 |
| stress100 gust 3.0 | FairALNSPlanner | 0.800568 | 0.598629 | 0.201939 | [0.181900, 0.223108] | 0.000150 | 通过 |

RAWA-RHP 相对 FairRiskAwareGreedy 和 ReserveOnlyPlanner 的提升也均显著为正，且安全指标不劣化。

## 严格 Latency Benchmark

严格 latency benchmark 目录：

- `outputs/results/strict_latency_benchmark_rawa_s1_20_v3/`

关键文件：

- `latency_benchmark_episodes.csv`
- `latency_benchmark_summary.json`
- `latency_benchmark.md`

设置：

- planner：RAWA-RHP。
- seeds：1-20。
- profile：ID、correlated gust OOD、narrow passage OOD。
- 风场：calm、moderate、severe。
- 障碍物密度：sparse、cluttered。
- 单 worker。
- 固定核心。
- 预热后测量。
- 使用与正式 full-mode 一致的连通地图预检查和确定性 retry。

结果：

| 指标 | 数值 |
|---|---:|
| episodes | 360 |
| replan samples | 9962 |
| p50 | 0.331365 s |
| p95 | 1.754860 s |
| p99 | 1.759013 s |
| max | 1.791652 s |
| deadline miss rate | 0.000000 |
| battery violation | 0 |
| clearance violation | 0 |
| reserve shortfall | 0 |
| return failure | 0 |
| emergency abort | 0 |

Latency gate 通过：p95 <= 1.8 s，p99 <= 2.2 s。

## 验证命令

编译：

```bash
/home/zyf/miniconda3/envs/uav-vla/bin/python -m compileall src scripts tests
```

合同测试：

```bash
PYTHONUSERBASE=/home/zyf/.cache/uav-vla-userbase /home/zyf/miniconda3/envs/uav-vla/bin/python -m pytest tests/test_optimization_contracts.py -q
```

最终验证结果：

- compileall：通过。
- tests/test_optimization_contracts.py：16 passed。
