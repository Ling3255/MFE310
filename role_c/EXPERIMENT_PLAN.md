# 角色 C：拒识器设计、校准与选择——增量实验计划

版本：2026-10-30，修订 2。状态：**待执行**。本计划只分析已保存的预测和分数，不训练、不重新抽取特征、不改动任何已有实验目录。

修订 2 相对修订 1 的变化（均在执行任何角色 C 分析之前确定）：

1. 主分析对象改为报告 Table 7 使用的拼写增强 BERT（`typo_mix`），原始 BERT 与 `clean_continue` 作为对照，使角色 C 的结论能直接接到报告 V.C 节；
2. C1 的集合比较固定在相同验证接纳率下进行，避免拒绝率差异混入 Jaccard；
3. C2 拆成“抽样波动”和“噪声漂移”两部分，`max_oos_f1` 与 `is_acceptance_95` 分开描述；
4. C3 的成本改为按比率并显式给出 OOS 先验 π，MSP/Entropy 的五个工作点按已有对齐规则由 validation 重新计算，方法选择只在 validation 上进行；
5. 新增 C4：阈值无关的分离度（AUROC、FPR@95%TPR），区分“排序失效”和“阈值漂移”；
6. 组合拒识器先做尺度统一；协方差消融因 CLS 特征未提交而不做；
7. 按 2026-11-01 截止日期给出必做/可选优先级和报告篇幅预算。

## 1. 当前基础与问题

已有成果（不重复）：TF-IDF+LR 基线；三个 seed（42、43、44）的原始 BERT；拼写增强 `typo_mix` 与等步数 `clean_continue`；MSP、Entropy、Mahalanobis（类内残差 + Ledoit–Wolf 共享协方差）；`max_oos_f1` 与 `is_acceptance_95` 两种阈值策略；Mahalanobis 的五个接纳率工作点和参考分布混合；clean / light / moderate 拼写噪声测试。

报告目前的拒识部分（V.C、Table 7）只在一个工作点上给出三种分数的描述性均值，结论是“存在权衡”。角色 C 把这个结论变成可检验、可用于决策的分析：

1. **C1 互补性**：三种分数是否在相同样本上犯错？
2. **C2 阈值稳定性**：验证集选出的阈值，抽样波动有多大？噪声导致的接纳率偏离是否远超这个波动？
3. **C3 成本驱动选择**：给定误接收 OOS 相对误拒的代价 λ 和 OOS 比例 π，应该选哪种分数和哪个工作点？只用 clean validation 做出的选择，在噪声下还是否合理？
4. **C4 失效机理**：噪声下误拒上升，是分数排序变差（AUROC 下降），还是排序基本保持而阈值不再对应 95% 接纳率？

## 2. 输入（只读）

| 角色 | 模型 | 文件 |
|---|---|---|
| 主分析 | 拼写增强 BERT（Table 7） | `BERT_Typo_Augmentation_20261019/outputs/typo_mix/seed{42,43,44}/{validation,clean,typo_light,typo_medium}_predictions.csv` |
| 对照 1 | 等步数 clean continuation | `BERT_Typo_Augmentation_20261019/outputs/clean_continue/seed{42,43,44}/` 同名文件 |
| 对照 2 | 原始 BERT | `BERT_Rejection_Study_20261019/outputs/seed*/{validation,test}_predictions.csv`，`.../typo_stress_test/outputs/seed*/typo_{light,medium}_predictions.csv` |
| 冻结阈值 | 各模型 | 同目录 `frozen_thresholds.json` |
| 领域映射 | CLINC150 | `CLINC150_Team_20261019/data/provenance/domains.json` |

每个预测文件含 `msp_score`、`entropy_score`、`mahalanobis_score`（均为越大越像 OOS，`score >= threshold` 拒绝）、`predicted_label_id`、`label_id`、`is_oos`、原文与噪声文本。Validation 为 3,000 IS + 100 OOS（OOS 比例约 3.2%），test 为 4,500 IS + 1,000 OOS（约 18.2%）。噪声只施加在 test 上，没有带噪声的 validation。

## 3. 实验约束

1. 不训练、不重抽特征，不修改原始数据、模型、冻结配置、旧预测或旧结果。
2. 任何阈值、工作点、组合规则和方法选择只用 validation 决定；test（含噪声 test）只用于报告。
3. 报告中凡是“按 test 选出的最优”只作为描述性参照（oracle），明确标注，不作为建议。
4. 只有 3 个 seed：报告逐 seed 值、均值 ± 样本标准差和方向一致性；条件区间沿用报告附录 B 的配对 bootstrap 约定（按 intent 和 OOS 分层，2,000 次，跨 seed/条件共享索引，固定模型、阈值和噪声）。不做显著性声明，不做多重比较校正，明确写出。
5. 早期 test 已被查看，本计划整体为**探索性补充分析**。
6. 新代码与产物只放在 `role_c/`；每个输出行保留 `row_id`、seed、模型、条件、分数、阈值和来源文件哈希。

## 4. C0：接管核验（必做，约 0.5 小时）

1. 运行 `python tools/verify_snapshot.py`、`python tools/verify_results.py` 和 `BERT_Rejection_Study_20261019/test_distance_model.py`，记录结果。
2. 记录全部输入文件的 SHA-256。
3. **阈值复算一致性**：用 validation 预测按原规则重算每个模型、分数、策略的 `max_oos_f1` 与 `is_acceptance_95` 阈值，必须与 `frozen_thresholds.json` 完全一致；按 test 预测复算的 Table 7 数值必须与 `summary_mean_sd.csv` 一致（容差 1e-9）。任何不一致都要停下报告，不能继续。
4. 复现已有五工作点规则：Mahalanobis 在 Operating Points 研究中的 aligned 阈值，凡与本计划输入是同一模型和参考分布的，重算结果必须一致；同时把该规则写成函数，供 C3 用于 MSP/Entropy。

产出：`role_c/outputs/c0_verification.json`、`role_c/outputs/input_hashes.json`、`role_c/outputs/table7_recomputed.csv`。

## 5. C1：检测器分歧与错误互补性（必做，精简版）

**工作点**：主分析固定在各分数自己的 `is_acceptance_95` 阈值上（validation IS 接纳率相同），补充 90% 和 99% 两个对齐工作点。

**分析**（主模型 × clean / light / moderate × 3 seed）：

- Spearman 分数相关，分 IS 和 OOS 两组计算。MSP 与 Entropy 都来自 softmax，相关性预期很高，报告重点放在 softmax 类分数与 Mahalanobis 之间；
- 在相同工作点下比较拒绝集合：Jaccard，以及“只有 A 拒绝 / 只有 B 拒绝 / 都拒绝 / 都接受”的 2×2 表，分 IS 正确、IS 错误、OOS 三类；
- 互补上限：对每个 OOS 样本，任一分数拒绝即算拦截，计算 OOS recall 上限，以及对应的 IS 误拒代价（描述性，不是可部署规则）；
- 分歧样本的 domain、句长和是否被扰动（`changed`）分布。

**判断规则（预先写定）**：在 moderate 条件下，如果 softmax 类分数与 Mahalanobis 在 OOS 上“只有一方拦截”的样本占 OOS 至少 5%，并且在 3 个 seed 上方向一致，才进入第 9 节的组合拒识器；否则组合实验不做，并在报告中说明原因。

产出：`score_correlations.csv`、`rejection_overlap.csv`、`complementary_error_counts.csv`、`disagreement_cases.csv`。

## 6. C2：阈值稳定性——抽样波动与噪声漂移（必做，核心）

### 6.1 抽样波动

对每个模型、分数、seed 做 2,000 次 bootstrap：IS validation 按 intent 分层重抽（每个 intent 20 条），OOS validation 整体重抽（100 条），每次按原规则重新选阈值。

- `is_acceptance_95` 只依赖 IS validation，重抽 OOS 对它没有影响。报告它的阈值分布，以及这些阈值在 clean test 上对应的 IS 接纳率区间；
- `max_oos_f1` 依赖仅 100 条 OOS validation，单独报告它的阈值分布和 test 接纳率区间；预期它会明显更不稳定。

### 6.2 噪声漂移

用冻结阈值计算 light / moderate test 的实际 IS 接纳率，与 6.1 中 clean 的 bootstrap 区间比较。报告“漂移倍数”：|实际接纳率 − 95%| 除以 bootstrap 区间半宽。报告 Table 7 中 IS FR 已达 19.6%（light）和 34.8%（moderate，Mahalanobis），本节要回答的是：这种偏离能否用验证抽样波动解释。

### 6.3 产出

`threshold_bootstrap_distribution.csv`、`threshold_stability_summary.json`、`threshold_drift_by_condition.csv`、`threshold_stability.png`（x 轴：条件；y 轴：test IS 接纳率；灰带：clean bootstrap 区间；点：各 seed 实际值；三种分数分面）。

不得用这些结果修改任何阈值。

## 7. C3：成本驱动的方法与工作点选择（必做，核心）

### 7.1 成本定义（按比率，预先固定）

\[
Cost(\lambda,\pi) = (1-\pi)\cdot FR + \lambda\cdot\pi\cdot FA
\]

FR 为 IS 误拒率（分母为全部 IS），FA 为 OOS 误接收率（分母为全部 OOS）。

- λ ∈ {1, 2, 5, 10}：误接收一条 OOS 相对误拒一条正常请求的代价；
- π ∈ {0.05, 0.182, 0.30}：部署时的 OOS 比例。0.182 等于 test 比例，作为主分析值；另外两个值作敏感性分析；
- 次要成本 `Cost_WA`：再加上 \(\lambda\cdot(1-\pi)\cdot WA\)，即把“接受但分错类”的 IS 也视为错误执行。主分析和次要分析都报告。

### 7.2 候选工作点

三种分数 × 五个对齐工作点（validation IS 接纳率 90%、92.5%、95%、97.5%、99%，规则同 C0.4），再加 `max_oos_f1`，每个分数 6 个候选。MSP/Entropy 的对齐阈值由角色 C 按同一规则从 validation 计算，这是新增内容，不是沿用已有结果。

### 7.3 选择协议

1. **验证选择**：对每个 (λ, π)，在 clean validation 上计算 18 个候选的成本，选出最小者。validation 的 OOS 比例是 3.2%，所以成本按比率计算，与 validation 的样本构成无关。
2. **测试评估**：把选中的候选放到 clean / light / moderate test 上，报告成本和各项错误率。
3. **遗憾值**：验证选择的 test 成本减去 test oracle 的最小成本，oracle 只作描述性参照。
4. **稳健性**：对第 1 步做 2,000 次 validation bootstrap，记录每个候选被选中的频率。

### 7.4 产出

`cost_candidates.csv`（全部候选在 val/test 各条件下的 FR、WA、FA 与成本）、`method_choice_by_cost.csv`、`selection_regret.csv`、`method_choice_heatmap.png`（行：λ；列：条件；格内：验证选中的分数/工作点及遗憾值；π=0.182）。

### 7.5 回答的问题

- 更担心误接收 OOS（λ 大）时，选哪个分数和工作点？更担心误拒正常请求时呢？
- clean validation 上选出的方案，到 moderate 噪声时遗憾值有多大？
- 是否存在对所有 (λ, π, 条件) 都不劣的方法？预期不存在，需要用数据确认。

## 8. C4：分离度与阈值的分解（必做，约 1 小时）

对主模型与两个对照，计算三种分数在 clean / light / moderate test 上的 AUROC 和 FPR@95%TPR（以 IS 为正类，阈值无关）。

- 如果 AUROC 基本不变但冻结阈值下 FR 大幅上升，说明主要问题在阈值漂移，C2 是根因；
- 如果 AUROC 明显下降，说明问题在分数本身，改阈值也解决不了。

另外用配对 bootstrap（附录 B 约定）给出 Table 7 中两组关键差值的条件区间：moderate 条件下 Entropy − Mahalanobis 的 IS CA 和 OOS recall。

产出：`separation_by_condition.csv`、`table7_paired_intervals.csv`。

## 9. 可选：组合拒识器（只有满足 C1 判断规则才做）

1. 尺度统一：每个分数用其在 clean IS validation 上的经验 CDF 映射到 [0, 1]，因为 Mahalanobis 在几千的量级，Entropy 在 0–5 之间，原始分数不能直接比较；
2. 候选规则：`max(F_ent, F_maha)`、`mean(F_ent, F_maha)`；
3. 阈值按 C3 的同一套对齐和选择规则只在 validation 上确定，冻结后在 test 上评估一次，作为 C3 的第 4 种“分数”加入成本比较。

不做加权搜索，不做两阶段规则，以免在 100 条 OOS validation 上过拟合。

## 10. 不做的内容

- 协方差估计消融：需要 train/val/test 的 CLS 特征，仓库中只提交了 metadata，没有提交特征，不为此重新抽取；
- 带噪声 validation 上的阈值重选：需要对 validation 重新做前向推理，超出“只用已保存预测”的范围。在报告局限中说明，作为后续工作。

## 11. 案例分析

从 C1–C3 中选 8–12 条代表性案例，覆盖：仅 Mahalanobis 拦截的 OOS、仅 Entropy 拦截的 OOS、两者都漏掉的 OOS、噪声导致的误拒、接受但分错类。每条记录原文、噪声文本、真实标签、预测、三种分数的决定和一句人工解释。案例按预先写定的规则抽取（每类按 `row_id` 取 seed 42 中的前若干条，或用固定随机种子抽样），不挑选。

产出：`casebook.csv`。

## 12. 实现与验收

- 单一脚本 `role_c/run_role_c_analysis.py`，子命令为 `c0`、`c1`、`c2`、`c3`、`c4`、`all`；输出写到 `role_c/outputs/`；使用 NumPy、pandas、matplotlib，不加入新依赖。
- `role_c/tests/`：分数方向；阈值边界（`>=` 拒绝、并列分数）；对齐规则复现已有 Mahalanobis 阈值；成本计算的手算例；bootstrap 的分层计数和固定种子可复现；Entropy 与 MSP 的定义。
- 所有表和图由脚本生成；`role_c/outputs/evidence_map.json` 记录报告中每个角色 C 数值对应的文件、行和计算方式。
- `role_c/REJECTION_METHOD_SELECTION.md`：结果摘要和建议。建议必须同时结合 λ、π、目标接纳率和噪声条件，不写“Mahalanobis 普遍最优”。

## 13. 优先级与报告篇幅（截止 2026-11-01）

| 优先级 | 内容 | 用途 |
|---|---|---|
| 1 | C0、C2、C4 | 解释 Table 7 中噪声下误拒为什么上升 |
| 2 | C3 | 把“存在权衡”变成按成本给出的选择建议 |
| 3 | C1 | 判断是否需要组合拒识器 |
| 4 | 第 9 节、案例 | 时间允许时再做 |

报告目前正文约 2,250 词，上限 2,750 词（2,500 + 10%），先以实际字数统计为准。角色 C 计划在 V.C 节增加约 150–200 词和一张两联图（左：C2 漂移；右：C3 选择热力图）；必要时压缩 V.D 的文字作为置换。所有新增数值都要在 `evidence_map.json` 中登记。

## 14. 交付清单

```text
role_c/
  EXPERIMENT_PLAN.md
  REJECTION_METHOD_SELECTION.md
  run_role_c_analysis.py
  tests/
  outputs/
    c0_verification.json  input_hashes.json  table7_recomputed.csv
    score_correlations.csv  rejection_overlap.csv  complementary_error_counts.csv  disagreement_cases.csv
    threshold_bootstrap_distribution.csv  threshold_stability_summary.json  threshold_drift_by_condition.csv  threshold_stability.png
    cost_candidates.csv  method_choice_by_cost.csv  selection_regret.csv  method_choice_heatmap.png
    separation_by_condition.csv  table7_paired_intervals.csv
    casebook.csv  evidence_map.json
```
