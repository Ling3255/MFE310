# 拒识方法选择结论与证据

## 核心结论

在当前模型和冻结协议下，**Mahalanobis 的 OOS 检出能力最强，但不存在适用于所有指标和业务目标的绝对最优方法。成本变化主要改变阈值工作点，而不是改变分数选择。**

## 已有证据

- `is_acceptance_95` 下，Mahalanobis 的 OOS F1 为 0.8453，OOS recall 为 0.8890，AUROC 为 0.9765；均高于 MSP 和 Entropy。
- Entropy 在部分接纳指标上仍有优势，且 Mahalanobis 相对 Entropy 的 OOS F1 优势仅 0.30 pp，区间跨 0，不能据此断言其必然更好。
- 噪声下 Mahalanobis 的 OOS recall 最高，但正确接纳率不一定最好。中度噪声时 MSP、Entropy、Mahalanobis 的正确接纳率分别为 65.83%、64.33%、60.38%。
- Entropy 在 `is_acceptance_95` 下的 test IS 接纳率 95% 区间为 94.0%–96.3%，阈值工作点相对稳定。
- 温度缩放显著改善 ECE，却降低 MSP/Entropy 的 OOS 排序能力，说明校准与拒识是不同目标。

## 业务成本下的选择

C5 在 validation 上按 `(1−π)·FR + λ·π·FA` 选择候选，再报告 test 成本。λπ 越大，所需工作点越保守，IS 接纳率从 max-F1 逐步降至约 90%。三种分数的原选择均为 Mahalanobis，与 test 上最优选择的差距最多约 0.7 pp。

因此建议：

1. 将 Mahalanobis 作为当前主要 OOS 分数；
2. 根据业务成本调整阈值，而不是只根据 F1 选择工作点；
3. 同时报告 IS 误拒与 OOS 误接纳，避免只追求 OOS recall；
4. 在具有互补性的样本上，允许评估 `combo_max` 和 `combo_mean`，但其 ECDF 必须在 validation 固定，不能在 test 重拟合。

## 未完成与限制

- 缺少三个 seed 的 CLS 和 FP32 logits，未执行协方差估计器消融和完整温度曲线扫描。
- 所有分析均属于探索性补充，不能声称独立确认。
- 三个 seed 的结果、阈值 bootstrap 和成本选择均保留原始文件，不声称消除了全部训练或历史 test 暴露的不确定性。

## 主要产物

- `c0_verification.json`：输入、阈值与工作点核验；
- `score_correlations.csv`、`rejection_overlap.csv`、`complementarity_decision.json`：方法分歧与互补性；
- `covariance_diagnostics.json`：Ledoit–Wolf 收缩记录及消融限制；
- `threshold_stability_summary.json`：阈值与接纳率稳定性；
- `calibration_summary.json`、`calibration_rank_flips.csv`：校准与拒识分离；
- `method_choice_by_cost.csv`、`rejector_candidates.json`、`method_choice_heatmap.png`：成本驱动选择和交接候选；
- `disagreement_cases.csv`：30 条人工解释案例。
