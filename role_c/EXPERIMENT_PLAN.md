# 角色 C：拒识器设计与选择——实验与分析计划

版本：2026-10-30，修订 3。状态：**待执行**。本计划只使用已保存的预测、分数和本地已有的 CLS 特征，不训练、不重新抽取特征、不改动任何已有实验目录。

## 0. 分工边界

团队分工中角色 C 的职责是：实现和比较 MSP、Entropy、Mahalanobis；负责分布拟合、验证集阈值选择和校准对照；解释方法选择的依据。要回答的核心问题是：**系统凭什么拒绝请求，为什么选这个方法？**

| 归属 | 内容 |
|---|---|
| C（本计划） | 三种分数的机制和分数分布；Mahalanobis 的分布拟合（协方差估计）；阈值策略及其稳定性；温度缩放校准对照；按成本给出的选择依据；交给 D 的冻结拒识器 |
| D（不在本计划内） | clean / light / moderate 噪声测试；0/10/25/50% 参考分布；五个接纳率工作点；噪声下的不确定性、错误案例和报告整合 |
| 接口 | C 提供拒识实现和冻结的候选拒识器（`role_c/outputs/rejector_candidates.json`：分数定义、方向、每个模型/seed/策略的阈值）；D 用它做受控评估。C 引用 D 的噪声结果时注明来源，不重复计算 |

修订 3 相对修订 2：按上述分工删去噪声漂移和噪声下的分离度分析（属于 D）；恢复分布拟合消融，因为原始 BERT 的 `{train,validation,test}_cls.npy` 已保存在本地（被 `.gitignore` 排除，未提交）；新增校准对照分析和个人展示演示。

## 1. 已有基础（不重复）

- 拒识实现：`BERT_Rejection_Study_20261019/distance_model.py`（类内残差 + Ledoit–Wolf 共享协方差）、`run_rejection_study.py`、单元测试 `test_distance_model.py`；
- 原始 BERT（seed 42/43/44）三种分数在 validation（3,000 IS + 100 OOS）和 test（4,500 IS + 1,000 OOS）上的逐样本预测、冻结阈值和阈值曲线；
- 温度缩放：`CLINC150_Team_20261019/outputs/multiseed/clean_calibration_mean_sd.csv`、`temperature_paired_deltas.csv`、`outputs/oos/seed*/calibration_bins_*.csv`；
- 拼写增强 BERT（`typo_mix`）和 `clean_continue` 的三种分数逐样本预测（报告 Table 7 的来源）。

已有证据给出的初步事实（C0 核验后再写入报告）：

| 事实 | 数值（3 seed 均值） | 来源 |
|---|---|---|
| clean test AUROC：Mahalanobis / Entropy / MSP | 0.977 / 0.954 / 0.945 | `BERT_Rejection_Study_20261019/outputs/summary_mean_sd.csv` |
| F1 最优阈值下 Mahalanobis 比 MSP 高 | +6.81 pp [5.27, 8.30] | 报告 V.C、`paired_intervals.csv` |
| F1 最优阈值下 Mahalanobis 比 Entropy 高 | +0.30 pp [−0.88, 1.51]，未确立优势 | 同上 |
| 原始 softmax 严重欠自信 | 平均置信度 0.59，准确率 0.95，ECE 35.6% | `clean_calibration_mean_sd.csv` |
| 温度缩放后 | ECE 0.85%，但 MSP AUROC −2.58 pp，Entropy −2.04 pp | `temperature_paired_deltas.csv` |

## 2. 实验约束

1. 阈值、协方差估计器、组合规则和方法选择只用 validation 决定；test 只用于报告。
2. 凡是“按 test 选出的最优”只作描述性参照（oracle），明确标注。
3. 3 个 seed：报告逐 seed 值、均值 ± 样本标准差和方向一致性；需要区间时沿用报告附录 B 的配对 bootstrap 约定（按 intent 和 OOS 分层，2,000 次，跨 seed 共享索引，固定模型和阈值）。不做显著性声明。
4. 早期 test 已被查看，全部为**探索性补充分析**。
5. 新代码与产物只放在 `role_c/`；输出行保留 `row_id`、seed、模型、分数、阈值和输入文件哈希。

## 3. C0：接管核验（必做）

1. 运行 `python tools/verify_snapshot.py`、`python tools/verify_results.py`、`python -m unittest BERT_Rejection_Study_20261019/test_distance_model.py -v`。
2. 记录全部输入文件（含本地 `*_cls.npy`）的 SHA-256；若某个 seed 缺少特征文件，记录下来，C2 只在特征齐全的 seed 上做。
3. 用 validation 预测按原规则重算 `max_oos_f1` 与 `is_acceptance_95` 阈值，必须与 `frozen_thresholds.json` 一致；用本地特征和 `distance_model.py` 重拟合 Ledoit–Wolf 检测器，重算的 `mahalanobis_score` 必须与已保存预测一致（相对误差 ≤ 1e-6）。不一致就停下报告。

产出：`role_c/outputs/c0_verification.json`、`input_hashes.json`。

## 4. C1：系统凭什么拒绝——机制与互补性（必做）

**机制**：MSP 和 Entropy 只看分类器对 150 个意图的概率分布，分别看最大概率和分散程度；Mahalanobis 看 [CLS] 表示到最近意图中心的距离，用类内共享协方差做尺度校正。分析内容：

- 三种分数在 clean validation/test 上按“IS 分类正确 / IS 分类错误 / OOS”三组的分布图，3 个 seed 叠加；
- 每种分数的 AUROC、FPR@95%TPR（IS 为正类）；另外计算“IS 分类错误 vs IS 分类正确”的 AUROC，衡量分数能否顺带拦住分错类的请求。

**互补性**（相同 `is_acceptance_95` 工作点，clean test）：

- Spearman 分数相关（IS、OOS 分开）。MSP 与 Entropy 同源，预期很高，重点放在 softmax 类分数与 Mahalanobis 之间；
- 拒绝集合的 2×2 表和 Jaccard，分 IS 正确、IS 错误、OOS 三类；
- 判断规则（预先写定）：仅一方拦截的 OOS 占全部 OOS ≥ 5%，且 3 个 seed 方向一致，才做第 9 节的组合拒识器。

产出：`score_distributions.png`、`separation_clean.csv`、`score_correlations.csv`、`rejection_overlap.csv`。

## 5. C2：分布拟合——协方差估计消融（必做，本地 CPU）

**问题**：Mahalanobis 的优势来自“用特征距离”本身，还是来自 Ledoit–Wolf 共享协方差？

**设置**：原始 BERT，seed 42/43/44；使用相同的 train CLS、相同的 150 个类均值、相同的类内残差。只更换协方差估计器：

| 变体 | 协方差 |
|---|---|
| LW（现方案） | Ledoit–Wolf 收缩的共享协方差 |
| OAS | Oracle Approximating Shrinkage |
| Empirical | 样本协方差（伪逆） |
| Diagonal | 只保留方差 |
| Identity | 单位阵，即到最近类中心的欧氏距离平方 |

每个变体使用同一阈值规则（`max_oos_f1` 与 `is_acceptance_95`，只在 validation 上选），报告 validation 和 clean test 的 AUROC、FPR@95%TPR、OOS recall、IS 接纳率；另外记录收缩系数和协方差条件数。

**预先写定的选择规则**：以 validation AUROC 选择估计器；差异 < 0.002 时选更简单的（Identity < Diagonal < LW/OAS < Empirical）。test 结果只用于报告，不用于改变选择。

产出：`covariance_ablation.csv`、`covariance_ablation.png`。

## 6. C3：阈值选择及其稳定性（必做）

**两种策略**：`max_oos_f1`（依赖 100 条 OOS validation）和 `is_acceptance_95`（只依赖 3,000 条 IS validation）。

**bootstrap**：每个分数、每个 seed 做 2,000 次：IS validation 按 intent 分层重抽（每个 intent 20 条），OOS validation 整体重抽，每次按原规则重新选阈值。报告：

- 阈值的分布（用分数在 IS validation 上的分位数表示，便于跨分数比较）；
- 把这些阈值用于 clean test 后，IS 接纳率和 OOS recall 的 2.5–97.5% 区间；
- 预期 `max_oos_f1` 的波动明显更大，这会削弱报告中基于 F1 最优阈值的比较（Mahalanobis vs Entropy +0.30 pp）。

**OOS 比例提示**：validation 的 OOS 比例约 3.2%，test 约 18.2%。`max_oos_f1` 依赖 OOS 比例，`is_acceptance_95` 不依赖。

产出：`threshold_bootstrap_distribution.csv`、`threshold_stability_summary.json`、`threshold_stability.png`。

不得用这些结果修改任何阈值。

## 7. C4：校准对照（必做，基本只用已有结果）

**问题**：温度缩放把 ECE 从 35.6% 降到 0.85%，为什么 MSP 拒识反而变差？校准好的置信度是否就是好的拒识分数？

1. 核验已有校准结果：ECE、NLL、可靠性分箱和温度值，3 个 seed；
2. 解释：温度缩放用同一个 T 缩放所有样本的 logits，不改变每个样本的预测类别。但对 MSP/Entropy 来说，它会改变不同样本之间置信度的相对排序，所以能改变 AUROC。已有结果显示 MSP AUROC 下降 2.58 pp，Entropy 下降 2.04 pp；Mahalanobis 使用特征，不受温度影响。
3. 补充分析（只用已保存的 logits 或概率，若已保存）：把 MSP AUROC 画成 T 的函数，标出验证集 NLL 最优的 T，以及拒识 AUROC 最好的 T 落在哪里。如果没有保存 logits，这一步跳过，只用已有的 delta 结果。

**结论形式**：说明“校准”和“OOS 分离”是两个不同目标，因此不能按 ECE 选择拒识方法；温度缩放适合需要可信概率的场景，但不能作为拒识改进。

产出：`calibration_vs_separation.csv`、`calibration_vs_separation.png`（左：可靠性图 raw vs TS；右：AUROC 随 T 的变化，或已有 delta 的条形图）。

## 8. C5：为什么选这个方法——成本驱动选择（必做）

**成本（按比率，预先固定）**：

\[
Cost(\lambda,\pi) = (1-\pi)\cdot FR + \lambda\cdot\pi\cdot FA
\]

- λ ∈ {1, 2, 5, 10}，π ∈ {0.05, 0.182, 0.30}，主分析 π = 0.182；
- 次要成本：再加上 \(\lambda\cdot(1-\pi)\cdot WA\)，即把“接受但分错类”也视为错误执行。

**候选**：3 种分数（加上 C2 选出的协方差变体，若它不是 LW）× {`max_oos_f1`，validation IS 接纳率 90/92.5/95/97.5/99%}。对齐规则复用 Operating Points 研究中的规则（C0 先确认能复现已有 Mahalanobis 阈值）。

**协议**：对每个 (λ, π)，在 clean validation 上选成本最低的候选，再报告它在 clean test 上的成本、遗憾值（相对 test oracle）和 2,000 次 validation bootstrap 的选中频率。原始 BERT 为主分析；拼写增强 BERT 用同一协议重做一遍，确保与报告 Table 7 的模型一致。

**交接给 D**：每个 (λ, π) 选中的候选写入 `rejector_candidates.json`。噪声条件下的表现由 D 评估，C 在报告中引用 D 的结果。

产出：`cost_candidates.csv`、`method_choice_by_cost.csv`、`method_choice_heatmap.png`、`rejector_candidates.json`。

## 9. 可选：组合拒识器（只有满足 C1 判断规则才做）

先用 clean IS validation 上的经验 CDF 把各分数映射到 [0, 1]，再比较 `max(F_ent, F_maha)` 和 `mean(F_ent, F_maha)`。阈值按 C5 的规则只在 validation 上确定，在 clean test 上评估一次，作为 C5 的一个候选。不做加权搜索。

## 10. 个人展示演示

`role_c/demo_rejector.py`：输入若干句子，加载 seed 42 原始 BERT 和已拟合的检测器，输出预测意图、三种分数、各自阈值和接受/拒绝决定。只做推理，不拟合、不调阈值，输出写在 `role_c/demo_outputs/`。演示句建议：一句正常请求、一句 OOS、一句三种分数意见不一致的句子（从 C1 的分歧样本中按规则选取）。

## 11. 实现与验收

- 单一脚本 `role_c/run_role_c_analysis.py`，子命令为 `c0`–`c5`、`all`；输出到 `role_c/outputs/`；只使用 NumPy、pandas、scikit-learn、matplotlib（项目已有），不加入新依赖。
- `role_c/tests/`：分数方向；阈值边界（`>=` 拒绝、并列）；对齐规则复现已有阈值；各协方差变体在小样本上与手算一致；成本手算例；bootstrap 分层计数与固定种子可复现。
- 所有数值登记到 `role_c/outputs/evidence_map.json`；结论写入 `role_c/REJECTION_METHOD_SELECTION.md`，不写“Mahalanobis 普遍最优”。

## 12. 优先级（截止 2026-11-01）

| 优先级 | 内容 | 回答的问题 |
|---|---|---|
| 1 | C0、C1、C4 | 凭什么拒绝；为什么不按校准选方法 |
| 2 | C2、C3 | 分布拟合为什么这样做；阈值可不可靠 |
| 3 | C5 | 在什么代价下选哪个方法 |
| 4 | 第 9 节、演示脚本 | 时间允许时再做；演示在展示前完成 |

报告中 C 负责 IV.C 和 V.C，以及温度缩放那句话。新增内容约 150–200 词加一张两联图（左：协方差消融或分数分布；右：成本选择热力图），所有数值登记在 `evidence_map.json`。

## 13. 交付清单

```text
role_c/
  EXPERIMENT_PLAN.md  REJECTION_METHOD_SELECTION.md
  run_role_c_analysis.py  demo_rejector.py  tests/
  outputs/
    c0_verification.json  input_hashes.json
    score_distributions.png  separation_clean.csv  score_correlations.csv  rejection_overlap.csv
    covariance_ablation.csv  covariance_ablation.png
    threshold_bootstrap_distribution.csv  threshold_stability_summary.json  threshold_stability.png
    calibration_vs_separation.csv  calibration_vs_separation.png
    cost_candidates.csv  method_choice_by_cost.csv  method_choice_heatmap.png  rejector_candidates.json
    evidence_map.json
```

## 14. 补充（2026-10-30）：缺少本地资产时的替代方案

组员无法提供三个 seed 的 `*_cls.npy`、`*_logits_fp32.npy` 和模型权重（C0 记录 `c2_c4_possible: false`；组仓库没有 Release，也没有提交这些文件）。不重新训练模型：新模型与报告不是同一批，所有数字都会对不上。替代方案如下，代码自动判断走哪条路径，命令不变。

| 步骤 | 原计划 | 替代方案 | 报告中的说法 |
|---|---|---|---|
| C2 | 五种协方差估计器的消融 | 只记录已拟合 LW 的收缩系数（`development_summary.json`，三个 seed 约 0.04），写入 `covariance_diagnostics.json`；不写 `covariance_selection.json`，C5 继续用 LW | 作为局限：协方差估计器未做消融 |
| C4 | FP32 logits 上的温度扫描 | 用 `CLINC150_Team_20261019/outputs/oos/seed*/` 中已保存的逐行 raw / 温度缩放 MSP、熵分数，做 T = 1 与拟合 T 的两点对比。增加三项：等 IS 接纳率（validation 95%）下的 FR / FA 变化；IS–OOS 样本对被重新排序的比例（`calibration_rank_flips.csv`，AUROC 变化 = 变好比例 − 变差比例）；与 FP32 分数的逐行一致性检查（同一批 checkpoint，FP16 前向，Spearman 须 > 0.9999） | 只给出两个温度点，没有 T 曲线 |
| 演示 | `demo_rejector.py` 实时推理 | `demo_replay.py`：在已保存的 validation / test 预测中查找句子，显示预测意图、三种分数、阈值和决定 | 视频中说明是“回放已保存输出，不是实时推理” |

**第 9 节组合拒识器**：C1 的预设规则已满足（三个 seed 中只被一种分数拦截的 OOS 比例为 5.3%–12.7%，`complementarity_decision.json`），因此 C5 自动加入 `combo_max`、`combo_mean` 两个候选。ECDF 参照为同一模型、同一 seed 的 clean IS validation 分数；C5 的 bootstrap 中 ECDF 固定不重拟合（只重选阈值），这一点在报告中注明。分离度结果在 `candidate_separation*.csv`。

**C1 案例表更正**：原类别名 `oos_caught_only_by_entropy` 等只排除了 Mahalanobis 或熵中的一方，没有排除 MSP，名字有误导。现改为 `oos_entropy_rejects_mahalanobis_accepts` 等（行不变，只改名字）；重跑 C1 会保留已写的 `interpretation`，但 test 行 4762、5335 的解释需要改，因为 MSP 也拒绝了这两句。
