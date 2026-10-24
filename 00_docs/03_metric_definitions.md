# 指标定义与模型选择规则

版本：2026-10-24，修订 2。适用于 M04、M07、M08、M09 及对应测试；当前是实现规范，尚无实验结果。

## 1. 分数与决策

内部 OOS 分数记为 s，越大越像 OOS：MSP 对应 `s=1-max(intent_probability)`；Mahalanobis 对应最近类的平方距离。

拒绝规则为 `s >= tau`；接纳规则为 `s < tau`。原始分数始终按这一方向保存。评估 IS 接纳的 ROC 时转换为 `a=-s`，仅改变评价方向，不修改检测器和原始数据。

## 2. OOS 指标各自的正类

| 报告名 | 机器字段 | 正类 | 使用分数/阈值 | 含义 |
|---|---|---|---|---|
| AUROC_OOS | `auroc_oos` | OOS | s 越大越正 | OOS 与 IS 的排序区分能力 |
| FPR@95TPR (IS positive) | `fpr95_is_positive` | IS | a=-s，ROC 工作点 | 接纳至少 95% IS 时的 OOS 误接纳比例 |
| FPR@95TPR (OOS positive) | `fpr95_oos_positive` | OOS | s，ROC 工作点 | 检出至少 95% OOS 时的 IS 误拒比例（原课题计划的定义） |
| FPR95 实际 IS 接纳率 | `fpr95_actual_is_acceptance` | IS | 同一 ROC 工作点 | ties 或离散样本使目标未必恰好等于 0.95 |
| OOS_FA@val95 | `oos_fa_at_val95` | 直接按样本计数 | 冻结的 val tau | test 上接纳的 OOS / 全部 test OOS |
| Test IS acceptance@val95 | `is_acceptance_at_val95` | 直接按样本计数 | 同一冻结 tau | test 上接纳的 IS / 全部 test IS |
| OOS recall@val95 | `oos_recall_at_val95` | OOS | 同一冻结 tau | test 上拒绝的 OOS / 全部 test OOS |

在 IS 为正类的 ROC 上：TPR=IS acceptance，FPR=OOS false acceptance。因此 FPR95 与“检出 95% OOS 时的 IS 误拒率”是两个不同方向的工作点。2026-10-24 导师审查后两者都输出：`fpr95_is_positive`（文献常用方向）与 `fpr95_oos_positive`（原课题计划方向）。报告以哪一个为主指标须在 E6 之前与导师确认并写入冻结记录，另一个作为补充；不得在看到 test 结果后再选择。

AUROC 使用 OOS 正类与 OOS 分数；FPR95 同时反转标签与分数。不要只反转标签，也不要用 AUROC 的正类约定直接计算 FPR95。

FPR95 参考常用 OOD 接纳方向，例如 ODIN：<https://arxiv.org/abs/1706.02690>。不同文献/工具可能采用不同约定，本文始终显式注明正类。

## 3. FPR95 的执行规则

1. 合并 IS test 和 OOS test 的 s 分数，标签设为 IS=1、OOS=0，接纳分数 a=-s。
2. 用全部唯一分数计算 ROC，不丢弃中间工作点；接受集合按 `a >= t` 定义，ties 整组纳入。
3. 在按阈值递减排列的 ROC 中，选择第一个 TPR>=0.95 的点，读取其 FPR；不插值。
4. 保存实际 TPR 和该描述性工作点，不能把这一 test 工作点写回运行阈值文件。
5. 如果要对应内部 `s < tau` 的严格接纳规则，可取 tau 为该 s 边界向正无穷方向的相邻可表示数，确保边界样本纳入；优先直接按 ROC 定义计数，避免不一致的 <=/<。

验证阈值选择规则：在 IS val 接纳率 ≥ 0.95 的候选中最大化 OOS val 召回；召回相同时取 IS 接纳率最高者，再相同时取最小 tau（2026-10-24 修订，原规则直接取最小 tau，会无谓降低 IS 接纳率）。

FPR95 使用测试分数描述曲线的标准工作点，允许计算但不能用于选择模型或运行阈值。E5 的阈值只在 val 选取，两者必须分别导出、分别命名。

## 4. 手算验收例子

以下是人工构造的测试夹具，不是实验结果。

给 20 条 IS 设置 s 分数 `0.01,0.02,...,0.20`，5 条 OOS 分数设为 `0.05,0.18,0.30,0.40,0.50`。达到 95% IS 接纳时，需要接纳前 19 条 IS（s<=0.19）；其中接纳了 2 条 OOS，所以 `FPR95=2/5=0.40`。

若在另一个验证集选出的冻结阈值为 tau=0.175，则这份人工 test 中只接纳 17 条 IS 和 1 条 OOS，得到 `IS acceptance@val95=0.85`、`OOS_FA@val95=0.20`。同一夹具下 `fpr95_oos_positive=0.80`：检出全部 5 条 OOS 需要 s≥0.05，同时误拒 16/20 条 IS。这说明验证时满足 95% 接纳并不保证 test 也达到 95%；两个 OOS 误接纳指标不必相等。

测试还应覆盖：所有分数相同、目标边界有 ties、分数方向反转、样本重排不改变结果，以及 IS/OOS 某一类缺失时报错。

## 5. 分类与领域错误

IS 意图 macro-F1 在拒绝前计算，固定 150 类完整标签集合；accuracy 为正确意图数 / 全部 IS。

跨领域错误率为 `count(domain(pred)!=domain(gold))/N_IS`。

同领域错误率为 `count(pred!=gold and domain(pred)==domain(gold))/N_IS`。

因此 `accuracy + cross_domain_error + within_domain_error = 1`。只在错误样本中计算的“构成比例”另行标记，不能混用分母。

## 6. 选择指标与验证使用

B0 的 C、BERT checkpoint 和领域损失 λ 统一按 IS val macro-F1 选择。B0 完全并列时选较小 C；checkpoint 完全并列时选较早 epoch；λ 按主计划的三种子均值与 0.2 个百分点容差规则选择。Accuracy 仅作为次要指标。

同一 val 反复参与 checkpoint 和 λ 等选择，选中的数值可能偏乐观。val 表和曲线用于说明选择过程，不作为无偏性能估计。test 不参与这些选择，验证重复使用本身不会泄漏 test 信息。

## 7. 单位和结果保存

机器文件统一保存 0–1 比例；报告可显示百分数。Dom−CE 的比例差乘 100 后才是百分点。保存 `positive_class`、`score_direction`、`threshold_source`、`target_tpr` 和实际工作点，避免只记录名为 FPR95 的一个数字。
