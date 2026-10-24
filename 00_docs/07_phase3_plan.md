# 第三阶段实验计划：已有文献支持的 OOS 检测与分类改进（E12–E15）

版本：2026-10-25，修订 1（草案）。作者：导师审查。状态：**计划草案，导师确认后冻结**。代码按本计划第 9 节另行实现，实现合并之后才成为预注册版本。

---

## 1. 背景：前两个阶段确立了什么

| 结论 | 证据 |
|---|---|
| 领域辅助（λ=0.1）**不能**可靠地提升意图分类 | 第二阶段 E10，在 MASSIVE 上用 10 个种子检验：Δ = −0.30 pp，95% CI [−0.94, +0.28]，判定为「不确定」。第一阶段 CLINC 上的 +0.17 pp，在 10 个种子下缩小到 +0.02 pp |
| BERT 明显优于 TF-IDF | 第一阶段：macro-F1 约高 5 pp |
| **Mahalanobis 明显优于 MSP** | 两个阶段都成立：CLINC 上 FPR@95%IS-TPR 约 0.11，MSP 约 0.17，所有种子一致 |
| 训练可能不充分 | 第一、二阶段分别有 9/15、13/20（CLINC）、15/20（MASSIVE）次运行在最后一个 epoch 才达到最佳 |

所以第三阶段**不再研究领域辅助**，转而检验**文献中被反复证实有效的方法**，看它们在本项目的协议下能否复现：
- **OOS 检测**：Energy、KNN 等打分方式（Liu et al., NeurIPS 2020；Sun et al., ICML 2022），以及 Outlier Exposure 或「OOS 作为额外一类」的训练方式（Hendrycks et al., ICLR 2019；Larson et al., EMNLP 2019）；
- **集成**：多种子集成（Lakshminarayanan et al., NeurIPS 2017）；
- **训练预算**：更长的训练。

## 2. 关于 test 数据的约束（重要）

- **CLINC150 的 test 已经用过两次**（E6、E9），第三阶段**完全不再使用**，包括 `test` 和 `oos_test`。
- **外部确认数据集**：BANKING77-OOS（Zhang et al., 2021，arXiv:2106.04564），来源为 <https://github.com/jianguoz/Few-Shot-Intent-Detection>，固定提交 `fa9553312b1dba54b1be0aae2d30b13171f898e5`，目录 `Datasets/BANKING77-OOS`。
  - in-scope 部分：50 个银行意图，train 5,905 条，valid 1,506 条，**test 2,000 条**。
  - **ID-OOS**：属于银行领域、但不在支持范围内的另外 27 个意图，train 2,062 条，valid 530 条，**test 1,080 条**。
  - **OOD-OOS**：valid 200 条，test 1,000 条。
- **已核实的数据重叠**（导师审查时逐条比对，见附录 A）：
  - BANKING77-OOS 的 **OOD-OOS test 与 CLINC 的 `oos_test` 完全相同**（1000/1000）；OOD-OOS valid 就是 CLINC 的 `oos_val` 加上 `oos_train`。
  - 因此 **OOD-OOS 不是新数据**，只能作为次级结果报告，并且必须披露它是「第三次使用 CLINC oos_test」。
  - **只有 in-scope test 和 ID-OOS test 是从未用过的数据**，第三阶段的确认性检验只建立在这两部分上。
- **ID-OOS train（2,062 条）任何时候都不得用于训练**。它和 ID-OOS test 属于同一批意图，用它训练就等于看到了 test 意图。
- 许可：BANKING77 原始数据为 CC BY 4.0（PolyAI）。该仓库本身没有声明许可，所以**只下载使用，不把原始数据提交进本仓库**；只提交下载脚本、哈希和处理后的统计结果。

## 3. 研究问题与假设

| 编号 | 假设 | 主要指标 | 角色 |
|---|---|---|---|
| **H4** | 在第 4 节开发阶段选出的**最佳免训练打分方法**（Energy、MaxLogit 或 KNN 之一），其 OOS 检测优于当前最佳的 Mahalanobis | BANKING77 **ID-OOS** test 上的 FPR@95%IS-TPR（越低越好） | 确认性 |
| **H5** | 在开发阶段选出的**最佳 OOS 训练方法**（Outlier Exposure 或 K+1 类之一），配合其最佳检测器，OOS 检测优于 CE + Mahalanobis；同时意图分类**不差于** CE 超过 0.5 pp | 同上；macro-F1 做非劣效检验 | 确认性 |
| **H6** | 5 个种子的集成，在意图分类上优于单个模型 | BANKING77 in-scope test 上的 macro-F1 | 确认性 |
| H4–H6 的次级指标 | AUROC_OOS；OOD-OOS（即 CLINC oos_test，已披露重复使用）上的 FPR95；集成对 OOS 检测的影响 | — | 描述性 |

**多重比较**：H4、H5、H6 三项主检验采用 Bonferroni 校正，每项 α = 0.05/3 ≈ 0.0167，区间取 **98.33%**。次级指标不做校正，只做描述。

**最小有意义差值（SESOI）**：FPR95 降低 **1.0 pp**（绝对值）；macro-F1 提升 **0.3 pp**；H5 的非劣效界限为 macro-F1 下降不超过 **0.5 pp**。

## 4. 开发阶段（E12–E13）：只用 CLINC 的 train 和 val，不看任何 test

### E12 免训练打分方法（使用现有模型，不重训）
- **模型**：第一、二阶段已有的 CLINC CE 模型，种子 0–9。
- **需要补做的推理**：对 `oos_val`（100 条）提取特征。种子 0–2 在 S05 中已有，种子 3–9 需要补做。这些都属于 val，允许使用。
- **候选方法**（统一规定分数越大越像 OOS）：
  - MSP：参照基线；
  - **MaxLogit**：−max logit；
  - **Energy**：−T·logsumexp(logit/T)，T 固定为 1，不调参；
  - Mahalanobis：当前最佳，作为参照；
  - **KNN**：对 L2 归一化的 CLS 特征，取到第 k 近邻的 train 特征的余弦距离，k ∈ {1, 10, 50}，k 在 val 上选。
- **选择规则（预注册）**：在 CLINC val 上，比较 10 个种子的平均 AUROC_OOS，选最高的候选进入 H4。之所以用 AUROC 而不用 FPR95，是因为 oos_val 只有 100 条，FPR95 的粒度太粗。如果最高的就是 Mahalanobis 本身，**H4 就不进行**，并如实记录「开发阶段没有找到更好的免训练方法」。

### E13 OOS 训练方法与训练预算（重新训练，种子 0–2）
| 候选 | 做法 | 使用的离群样本 |
|---|---|---|
| **K+1** | 把 OOS 当作第 151 类，用其概率作为 OOS 分数 | CLINC `oos_train`（100 条） |
| **OE** | 交叉熵损失 + β · 在离群样本上与均匀分布的交叉熵，β 固定为 0.5 | 同上 |
| **长训练** | CE，epoch 从 5 增加到 10 | 无 |

- 其余超参数与 `01_base.yaml` 相同。
- **选择规则（预注册）**：在 CLINC val 上比较 3 个种子的平均 AUROC_OOS，选最高的一个进入 H5；同时要求它的 val macro-F1 不低于 CE 超过 0.5 pp。如果两个都不满足，**H5 就不进行**。
- 「长训练」不直接进入 H5。如果它在 val 上使 macro-F1 提高 0.3 pp 以上，那么第 5 节所有 BANKING77 训练统一改用 10 个 epoch；这个决定必须在任何 BANKING77 训练开始之前，写进本计划第 11 节。

### E13 的局限
CLINC 的 oos_val 只有 100 条，开发阶段的排序可能不稳定。这是有意接受的折中，好处是不需要动用任何 test。

## 5. 确认阶段（E14–E15）：BANKING77-OOS

### 数据处理（预注册）
- **in-scope**：使用 train / valid / test 三个原始划分（50 个意图）；valid 作为 val。
- **验证用 OOS**：ID-OOS valid（530 条）加上 OOD-OOS valid（200 条），**只用于选阈值**（选阈值的方法与 S05 相同）。FPR95 和 AUROC 本身不需要阈值。
- **test**：in-scope test 2,000 条；ID-OOS test 1,080 条（主要）；OOD-OOS test 1,000 条（次级，已披露）。
- **领域头**：BANKING77 只有一个领域，所以**关闭领域头**（λ=0，num_domains 设为 1），需要修改代码，见第 9 节。
- **OE 和 K+1 的离群样本来源**：不能用 CLINC 的 `oos_train`，因为它出现在 OOD-OOS valid 里，会污染验证集。改用 **CLINC 的 in-scope train 文本，但剔除 banking 和 credit_cards 两个领域**（共 12,000 条），并且去掉与 BANKING77 任何划分完全相同的文本。这属于标准的辅助离群数据做法。

### E14 训练
| 组别 | 内容 | 训练次数 |
|---|---|---|
| CE | bert-base，种子 0–9 | 10 |
| H5 的候选 | 在 E13 中选出的 OE 或 K+1，种子 0–9 | 10（如果 H5 不进行则为 0） |

### E15 test 评估（每项都只运行一次，需要导师批准）
- **H4**：同一个 CE 模型，比较「选出的打分方法」与「Mahalanobis」，按种子配对。
- **H5**：同一种子下，比较「候选训练方法 + 它在 val 上 AUROC 最高的检测器」与「CE + Mahalanobis」。
- **H6**：用 CE 的种子 0–4 和种子 5–9 组成两个不相交的集成，概率取平均；分别与各自 5 个成员的单模型平均值比较。

## 6. 统计分析（预注册，沿用第二阶段的 M12）

- **H4、H5**：按种子配对，配对单位是同一种子下的两个系统。
  - 种子层面：计算 Δ_s，做精确 Wilcoxon 检验（双侧）。
  - 两层 bootstrap（2000 次）：重抽种子，in-scope test 按意图分层重抽，ID-OOS test 整体有放回重抽。
  - **判定为「支持」**：98.33% 区间整体落在有利的一侧，并且 Wilcoxon p < 0.0167。
  - H5 的非劣效部分：macro-F1 差值的 98.33% 区间下界 > −0.5 pp。
- **H6**：集成只有 2 个，无法做种子层面的检验。改为对每个集成分别做「集成 − 成员平均」的分层 bootstrap（2000 次，只重抽 test 样本）。**两个集成的 98.33% 区间都整体大于 0 才算支持**。
- 「不确定」不得解释为等价；所有结果都要报告，包括没有进入确认阶段的候选。

## 7. 流程与门控

| 步骤 | 内容 | 门控 |
|---|---|---|
| 0 | 导师确认本计划 → 实现第 9 节的代码 → 合并（即为预注册） | 代码 PR 需经导师审查 |
| 1 | E12：补做 CLINC 种子 3–9 的 oos_val 推理，计算各打分方法在 val 上的结果 | 只用 val |
| 2 | E13：训练 K+1、OE、长训练（各 3 个种子），在 val 上比较 | 只用 val |
| 3 | 按第 4 节的规则确定进入 H4 和 H5 的方法，以及 epoch 数，**写进第 11 节并提交** | 导师确认后才能开始第 4 步 |
| 4 | 下载 BANKING77-OOS（固定提交）并校验哈希 → E14 训练 | 不得使用 ID-OOS train |
| 5 | 在 val 上做健全性检查，然后推送 | 导师审查，**在 GitHub 上批准** |
| 6 | E15：H4、H5、H6 各运行一次 test 评估和统计 | 每项只能运行一次 |

## 8. 算力估计（参考：4090 上 CLINC 每个 epoch 约 30 秒）

| 内容 | 估计 |
|---|---|
| E12 推理与打分 | <15 分钟 |
| E13：9 次训练，其中长训练 3 次，每次 10 个 epoch | 约 50 分钟 |
| E14：20 次 BANKING77 训练，train 集约为 CLINC 的 40% | 约 30–40 分钟（如果改用 10 个 epoch，约 1–1.2 小时） |
| E15 推理与统计 | <20 分钟 |
| 磁盘 | 每次运行约 2.6 GB，E13 + E14 合计约 75 GB |

## 9. 需要实现的代码（计划确认后另开 PR）

| 编号 | 文件 | 作用 |
|---|---|---|
| M13 | `03_src/clinc_domain_aux/ood_scores.py` | MaxLogit、Energy、KNN 打分，并带单元测试，与现有的 MSP、Mahalanobis 接口一致 |
| M06 修改 | `training.py`、`domain_aux.py` | 支持 K+1 类和 OE 损失（离群样本单独一个 loader，β 可配置）；支持 `num_domains=1` 时关闭领域头。CLINC 的默认行为和配置哈希保持不变 |
| S13 | `04_steps/13_score_dev.py` | E12 和 E13 在 val 上的打分、比较和选择记录 |
| S14 | `04_steps/14_prepare_banking77_oos.py` | 下载 BANKING77-OOS 并校验哈希，构建离群辅助集，剔除重复文本，**提交前检查确保 ID-OOS train 没有被用到** |
| S15 | `04_steps/15_evaluate_phase3.py` | 批准门控，H4、H5、H6 各评估一次 |
| J05、J06 | `05_jobs/05_dev_phase3.sh`、`06_train_banking77.sh` | 批量训练 |
| T05 | `06_tests/test_05_phase3.py` | 打分方法、OE 损失、数据泄漏检查、门控 |

## 10. 风险

| 风险 | 应对 |
|---|---|
| CLINC 的 oos_val 太小，开发阶段选错方法 | 已接受。所有候选在 val 上的结果都要报告 |
| BANKING77 的 OOD-OOS 被误当作新数据 | 计划里已明确只能作为次级结果，S15 会在输出中自动附上披露说明 |
| 学生用 ID-OOS train 做 OE 或调参 | S14 不导出这部分数据；S15 会检查训练运行的数据哈希 |
| E13 什么都没选出 | 如实报告，只进行 H6 |
| OE 或 K+1 损害意图分类 | H5 自带非劣效检验 |

## 11. 修订与开发阶段决定记录

| 日期 | 内容 | 是否在相应 test 评估之前 |
|---|---|---|
| 2026-10-25 | 初稿 | 是 |
| （待填） | E12 选出的打分方法：____ | 是 |
| （待填） | E13 选出的训练方法：____；BANKING77 的 epoch 数：____ | 是 |

## 附录 A：BANKING77-OOS 与 CLINC150 的重叠核查（2026-10-25）

比较方法：小写化、合并空白后做精确匹配；数据来自固定提交 `fa95533`。

| BANKING77-OOS 文件 | 条数 | 与 CLINC 重叠 |
|---|---|---|
| ood-oos/test | 1000 | **与 `oos_test` 重合 1000 条（完全相同）** |
| ood-oos/valid | 200 | 与 `oos_val` 重合 100 条，与 `oos_train` 重合 100 条 |
| id-oos/test | 1080 | 与 CLINC train 重合 2 条 |
| test（in-scope） | 2000 | 与 CLINC test 重合 1 条 |
| valid / train（in-scope） | 1506 / 5905 | 0 |
| id-oos/train | 2062 | 与 CLINC train 重合 1 条 |

`seq.in` 的 SHA-256 前 16 位：train `5ba97261ce844b33`，valid `5dcde9139ca21345`，test `dc38952ebf63fcb1`，id-oos/train `cd0e24f07649f4b0`，id-oos/valid `859a89edce0e726f`，id-oos/test `faeeb6fa5c97b249`，ood-oos/valid `d1e14f587ab3a090`，ood-oos/test `5602a8ce5239fe5f`。
