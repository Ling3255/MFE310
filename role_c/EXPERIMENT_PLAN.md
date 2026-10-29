# 角色 C：拒识器设计、校准与选择——增量实验计划

版本：2026-10-25，修订 1。状态：**待执行**。本计划在已有 MSP、Entropy、Mahalanobis 实验基础上开展增量分析，不重复训练分类模型，不重复已有拒识实验。

## 1. 当前基础

共同仓库：`G:/MFE310/report_workspace/MFE310TC-CLINC150`

已有成果：

- TF-IDF + Logistic Regression 基线；
- 三个 seed（42、43、44）的 BERT 分类模型；
- 拼写增强与 clean continuation 对照；
- MSP、Entropy、Mahalanobis 三种拒识方法；
- 基于类内残差和 Ledoit–Wolf 的共享协方差拟合；
- `max_oos_f1` 与 `is_acceptance_95` 两种阈值策略；
- 五种 IS 接纳率工作点、参考分布混合及 clean/light/moderate 拼写噪声评估。

角色 C 当前任务不是重新实现上述内容，而是回答三个增量问题：

1. 不同拒识方法是否在错误样本上具有互补性？
2. validation 选出的阈值是否稳定？
3. 在不同业务错误成本下，应当如何选择拒识方法？

## 2. 实验约束

1. 不重新训练分类模型，不修改原始数据、模型、冻结配置、旧预测或旧结果。
2. OOS 分数统一为“越大越像 OOS”，`score >= threshold` 表示拒绝。
3. 不使用 test 选择检测器、组合规则、阈值、随机种子或报告指标。
4. 不把 Mahalanobis 用于没有 BERT CLS 特征的 TF-IDF 模型。
5. 新分析必须能由保存的预测和分数重新计算，不编造实验数值。
6. 新代码与产物放入 `role_c/`，不覆盖旧实验目录。
7. 保留 `sample_id`、seed、score、threshold 和预测来源，确保结果可追溯。
8. 由于早期 test 已被查看并用于探索，本计划属于**探索性补充分析**，不声称独立确认。

## 3. 阶段 C0：现有结果接管与核验

### 3.1 执行

```bash
cd G:/MFE310/report_workspace/MFE310TC-CLINC150
python tools/verify_snapshot.py
python tools/verify_results.py
python -m unittest BERT_Rejection_Study_20261019/test_distance_model.py -v
```

### 3.2 核对内容

- 三种分数的方向和拒识规则；
- train/val/test 的用途；
- 类内残差协方差拟合；
- 阈值选择策略；
- AUROC、FPR95、OOS recall、IS acceptance、OOS false acceptance 的定义；
- 预测文件、结果表和图能否独立复算。

### 3.3 产出

- `role_c/baseline_verification.json`
- `role_c/existing_method_summary.csv`

本阶段是接管与审计，不算新实验。

## 4. 实验 C1：检测器分歧与错误互补性

### 4.1 目的

判断 Entropy 和 Mahalanobis 是否在不同样本上具有互补性，而不是仅比较平均指标。

### 4.2 输入

```text
BERT_Rejection_Study_20261019/outputs/seed42/test_predictions.csv
BERT_Rejection_Study_20261019/outputs/seed43/test_predictions.csv
BERT_Rejection_Study_20261019/outputs/seed44/test_predictions.csv
```

### 4.3 分析内容

- MSP、Entropy、Mahalanobis 的 Spearman 分数相关性；
- 拒识集合 Jaccard 相似度；
- 每种方法独有的误拒和误接纳数量；
- 分歧样本的 Intent、Domain、句长和噪声类型分布；
- 三种方法共同失败的样本；
- 分歧样本中正确接纳、错误接纳和误拒的构成。

### 4.4 产出

- `role_c/score_correlations.csv`
- `role_c/rejection_disagreements.csv`
- `role_c/complementary_error_counts.csv`
- `role_c/disagreement_cases.csv`

### 4.5 回答的问题

Mahalanobis 与 Entropy 是给出高度相似的判断，还是在不同错误上具有互补价值？是否需要进一步研究组合拒识规则？

## 5. 实验 C2：阈值稳定性与校准稳健性

### 5.1 目的

检验 validation 选出的阈值是否稳定，以及 95% IS acceptance 是否只是特定验证样本上的工作点。

### 5.2 方法

对 IS validation 和 OOS validation 联合重采样，每次按原规则重新选择阈值，不改变模型或 test：

1. 按 Intent 分层重抽 IS validation；
2. 整体重抽 OOS validation；
3. 分别按 `max_oos_f1` 和 `is_acceptance_95` 重新选阈值；
4. 保存每次阈值、IS acceptance 和 OOS recall；
5. 比较验证分布与已有冻结阈值在 test 上的实际表现。

### 5.3 产出

- `role_c/threshold_bootstrap_distribution.csv`
- `role_c/threshold_stability_summary.json`
- `role_c/threshold_stability.png`

### 5.4 回答的问题

95% IS acceptance 是否稳定？阈值波动能否解释 test 上接纳率偏离目标的现象？

不得利用这些结果重新调整 test 阈值。

## 6. 实验 C3：按业务错误成本选择拒识方法

### 6.1 目的

把“哪个方法指标更高”转化为“在什么业务目标下应该选择哪个方法”。

### 6.2 成本定义

\[
Cost = IS\_FalseRejection + \lambda \times OOS\_FalseAcceptance
\]

预先固定：

- `λ = 1`：误拒与误接纳成本相当；
- `λ = 2`：OOS 被错误执行的代价较高；
- `λ = 5`：OOS 被错误执行的代价很高；
- `λ = 10`：OOS 被错误执行的代价极高。

### 6.3 分析内容

沿用已有的五种 IS 接纳率工作点（90%、92.5%、95%、97.5%、99%），比较 MSP、Entropy、Mahalanobis 在不同噪声和 λ 下的 Cost，并检查方法选择是否随业务目标改变。

### 6.4 产出

- `role_c/cost_sensitive_frontiers.csv`
- `role_c/method_choice_by_cost.csv`
- `role_c/method_choice_heatmap.png`

### 6.5 回答的问题

- 更担心错误执行 OOS 时，应该选择谁？
- 更担心误拒正常请求时，应该选择谁？
- 是否存在不依赖单一指标的“全局最优方法”？

## 7. 案例分析与报告

从三个实验中选择 12–20 条代表案例，覆盖：

- 三种方法各自独有错误；
- 三者共同错误；
- 分类正确但被误拒；
- 分类错误但被接纳；
- OOS 被误接纳；
- 拼写噪声引发的拒识变化。

每条记录：原文、噪声文本、真实标签、各方法预测、拒识决定、错误类型及人工解释。

产出：

- `role_c/casebook.csv`
- `role_c/REJECTION_METHOD_SELECTION.md`
- `role_c/evidence_map.json`
- `role_c/run_role_c_analysis.py`

报告不得直接宣布 Mahalanobis 普遍最优，而应结合：

- 不同指标和错误类型；
- 阈值稳定性；
- 业务错误成本；
- 目标 IS 接纳率；
- 方法各自的失败案例。

## 8. 可选扩展：组合拒识器

仅当 C1 证明 Entropy 与 Mahalanobis 具有明显互补性时考虑：

- `max(Entropy, Mahalanobis)`；
- 加权分数；
- Entropy 初筛后 Mahalanobis 复核的两阶段规则。

组合规则必须只在 validation 上选择并冻结，再在已有 test 上做一次描述性评估，不得根据 test 反复调整。

## 9. 可选扩展：协方差估计消融

只有在取得 BERT checkpoint 或 CLS 特征后执行：

- Ledoit–Wolf；
- OAS；
- 对角协方差；
- 不使用共享协方差的最近类中心距离。

保持相同 CLS、类均值、train 拟合规则、validation 阈值规则和 test 预测，用于判断收益来自距离本身还是协方差估计方式。

此项不是当前必做内容，不为获取它重新训练分类器。

## 10. 实现与验收要求

1. 使用项目现有 Python 风格及 NumPy、pandas、matplotlib，不增加不必要依赖。
2. 为 Entropy、阈值边界、配对索引、分数方向和成本计算添加针对性测试，放在 `role_c/tests/`。
3. 所有表格和图必须由脚本生成，不手工编造结果。
4. 运行单元测试和完整分析，确认结果可复现。
5. 最终提交修改文件、运行命令、测试结果、主要发现和未验证限制。
6. 不把 Mahalanobis 写成普遍最优；最终建议必须结合阈值稳定性、业务成本和目标接纳率。

## 11. 最终交付清单

```text
role_c/
  baseline_verification.json
  existing_method_summary.csv
  score_correlations.csv
  rejection_disagreements.csv
  complementary_error_counts.csv
  disagreement_cases.csv
  threshold_bootstrap_distribution.csv
  threshold_stability_summary.json
  threshold_stability.png
  cost_sensitive_frontiers.csv
  method_choice_by_cost.csv
  method_choice_heatmap.png
  casebook.csv
  REJECTION_METHOD_SELECTION.md
  evidence_map.json
  run_role_c_analysis.py
  tests/
```

核心工作为 **C1 分歧分析、C2 阈值稳定性、C3 成本驱动选择**。A/B 的分类基线和拼写增强不需要重复运行。
