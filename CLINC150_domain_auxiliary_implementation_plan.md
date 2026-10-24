# CLINC150 领域辅助学习实验实施计划

版本：2026-10-24，修订 3（导师 PR #2：阈值平局、双方向 FPR95 与冻结核验）。用途：依据导师提供的领域辅助学习方案，指导课程项目实施、实验记录与报告写作。

本文保留导师方案的研究问题、三个比较系统和 E0–E8 实验结构。新增内容是实现约定、验收标准、统计口径、计算预算及结果归档规范。本文是待执行计划，不代表实验已完成，也不预设改进一定有效。

## 1. 项目定位与实验边界

研究问题：在英文单句意图识别中，为 BERT 增加领域预测辅助任务，是否提高细粒度意图分类的 macro-F1、减少跨领域混淆？这种训练变化对不支持请求的 OOS 检测有何影响？

| 研究问题 | 主要证据 | 实验 |
|---|---|---|
| RQ1：BERT 是否优于词法基线？ | BERT-CE 与 B0 的意图 macro-F1、准确率 | E2、E3、E6 |
| RQ2：领域辅助监督是否改善意图分类？ | BERT-Dom 减 BERT-CE 的 macro-F1 差值、跨种子变异和配对置信区间 | E4、E6 |
| RQ3：改进发生在什么错误上？ | 跨领域/同领域错误率、领域混淆矩阵、修复与新增错误 | E7 |
| RQ4：领域辅助监督如何影响 OOS？ | 相同检测方法下的 AUROC、FPR@95TPR、固定阈值下的拒绝表现 | E5、E6、E8 |

唯一主干预为增加领域分类损失：`L = L_intent + λ × L_domain`。主实验不加入拼写增强、对比学习或新的数据采样策略。

与队友的区别：队友改变训练文本，研究拼写增强；本项目改变监督任务，研究领域层次结构。可以共用经核验的数据加载、词法基线、BERT 基线和检测代码，但领域模型、参数选择、领域错误分析和结论应有明确个人贡献记录。共享内容如实署名，报告提交形式按导师最终安排执行。

领域标签由意图标签确定，不是额外采集的人工信息。研究假设是：将已有层级关系显式加入训练，可能提供有用的归纳偏置。它也可能过度强调领域差异、损害同领域细粒度区分，因此负结果同样有分析价值。本项目不声称提出全新的学习算法，也不将公开英文助手数据上的性能等同于制造业部署能力。

## 2. 数据、版本与使用权限

### 2.1 已有本地数据

主数据文件：`G:/MFE310/dataset/data/data_full.json`。

领域元数据：`G:/MFE310/dataset/data/domains.json`。

已核对仓库提交：`828f8093932c8fe6ca7936c3d2e52903b1c523de`。实施时仍须记录数据文件及领域映射文件的 SHA-256，不能仅记录目录名。

来源为官方 CLINC OOS-Eval 仓库，参考 Larson 等（2019），数据仓库包含 CC BY 3.0 许可。模型权重的版本与许可应另外记录，不能假定与数据许可相同。

数据为众包英文单句、单意图文本。十个领域为 banking、credit_cards、kitchen_and_dining、home、auto_and_commute、travel、utility、work、small_talk 和 meta，每领域 15 个意图。

| 部分 | 范围内 IS | OOS | 本项目用途 |
|---|---:|---:|---|
| train / oos_train | 15,000 | 100 | 仅 15,000 IS 用于模型和 Mahalanobis 拟合；oos_train 不使用 |
| val / oos_val | 3,000 | 100 | 模型、λ 和 checkpoint 选择；拒绝阈值按统一规则确定 |
| test / oos_test | 4,500 | 1,000 | 协议冻结后评估 |
| 合计 | 22,500 | 1,200 | 共 23,700 条 |

只使用上述语料和官方领域映射，不混入 small、imbalanced、oos_plus、Wikipedia 或额外文本。预训练 BERT 是模型资源，不属于新增任务语料，但必须记录其来源。

### 2.2 测试集历史与结论边界

依据本次对设计过程的澄清，本项目关于领域辅助模型的设计决策没有参考任何 test 结果；模型、checkpoint、λ、检测器和阈值的选择均使用 train/val。该 benchmark 的 test 曾被小组其他成员查看，并用于其拼写增强项目的迭代，因此需要分别记录本项目的设计过程与小组此前的测试访问。

本项目将结果表述为“冻结协议下的一次性评估”，不将其整体自动定性为探索性，也不声称测试集在整个小组范围内完全未被接触，或具有完全独立确认的地位。这一表述以实际实验记录为依据；如果后续确有根据 test 结果改变领域模型或分析选择的情况，须另行标注受影响的结果。

建议报告措辞："Design decisions for the domain-auxiliary model did not use any test results. All model and hyperparameter selection used training and validation data. Although other group members had previously inspected this benchmark's test set, the present study reports a one-time evaluation under a frozen protocol and does not claim fully independent confirmation."

规则如下：

1. 开发阶段可对各 split 做预先约定的数据质量统计，不能用 test 的预测、性能或个例错误选择模型。
2. 参数、特征、λ、检测分数、阈值策略和报告指标均在 E5 冻结。
3. E6 导出最终预测后，允许基于这些预测重算表格、置信区间和错误分析；不据此回头选模型。
4. 如发现真实实现错误，记录 bug、受影响结果、修改提交和重跑原因，保留旧记录。
5. 原始切分不移动。重复样本排除结果仅作为敏感性分析，不代替官方结果。

## 3. 工作目录与代码接口

**全部新写的项目代码统一放在 `G:/MFE310/mfe310code/`。** 数据仓库继续位于 `G:/MFE310/dataset/`，通过配置读取。后续不在根目录、数据仓库或其他临时目录分散编写正式实验代码。以下代码清单是待实现规划；文件夹和结构说明先建立，代码按阶段实现。

计划编写 **26 个有功能编号的代码文件：21 个 Python 实现文件、3 个 Python 测试文件、2 个 Linux Shell 脚本**。另外有 1 个 Python 包初始化文件 `__init__.py`，不计为独立功能；全部实现后物理代码文件合计 27 个。15 次 BERT 训练复用同一个训练入口，不需要写 15 份训练代码。

```text
mfe310code/
  README.md
  00_docs/                  # 编号文档、冻结记录、贡献记录
    00_code_structure.md    # 完整文件编号、职责、输入输出和实现顺序
    01_freeze_record.md
    02_contributions.md
    03_metric_definitions.md
    04_ai_use_declaration.md
    05_change_log.md
    evidence_map.json       # 实验后生成的证据索引
  01_configs/
    01_base.yaml
    02_baseline.yaml
    03_search.yaml
    04_frozen.yaml
  02_environment/
    01_requirements.in
    02_requirements-lock.txt
  03_src/
    clinc_domain_aux/       # 可导入的 Python 包，模块名称不以数字开头
      __init__.py
      common.py            # M01 配置、种子、哈希、日志
      data.py              # M02 加载、映射、切分
      audit.py             # M03 领域统计、审计
      baseline.py          # M04 TF-IDF + LR
      domain_aux.py        # M05 共享 BERT、意图头、领域头
      training.py          # M06 统一训练循环
      detectors.py         # M07 MSP、Mahalanobis、阈值
      evaluation.py        # M08 预测导出、指标
      statistics.py        # M09 配对 bootstrap
      domain_analysis.py   # M10 混淆、案例、图表
      inference.py         # M11 单句推理演示
  04_steps/                # 按执行顺序编号的命令入口
    00_check_environment.py
    01_prepare_and_audit.py
    02_train_baseline.py
    03_train_bert.py
    04_select_lambda.py
    05_fit_detectors_and_freeze.py
    06_evaluate_test.py
    07_bootstrap.py
    08_analyze_domains.py
    09_demo_inference.py
  05_jobs/
    01_train_all.sh
    02_evaluate_frozen.sh
  06_tests/
    test_01_data.py
    test_02_metrics.py
    test_03_pairing.py
  07_metadata/
    data_manifest.json
    intent_domain_map.csv
    label_to_id.json
    domain_to_id.json
    exposure_log.md
  08_runs/
    ce_seed0/
    dom_lam0p1_seed0/
    ...
  09_results/
    01_validation/
    02_test/
    03_statistics/
    04_figures/
    05_tables/
    06_cases/
  10_exports/              # 报告图表、演示、冻结 ZIP 等最终交付
```

**编号规则：** `00`–`10` 标识目录类别；`04_steps/00`–`09` 表示执行顺序；M01–M11 为可导入模块的功能编号；T01–T03 为测试编号；J01–J02 为 Linux 作业编号。Python 模块使用合法名称，功能编号在清单及模块 docstring 中标明。编号不对应随机种子，实验 ID 另用 `ce_seed0`、`dom_lam0p1_seed0` 等命名。

完整的 26 个代码文件职责、输入输出和 E0–E8 对应表见 [代码结构与编写清单](mfe310code/00_docs/00_code_structure.md)。该清单是后续新增代码的目录约定。

路径用配置或环境变量指定。提交仓库不包含私人绝对路径、账号、令牌或无权再分发的权重。检查点若不随 ZIP 提交，提供获允许的获取方式和哈希。

每个 `08_runs/` 训练运行目录至少保存 `config.yaml`、`environment.json`、`train_log.jsonl`、`checkpoint_selection.json`、最佳 checkpoint、验证集预测及训练样本顺序摘要。每条样本使用稳定 ID，例如 `train:00000`、`val:00000`，避免通过随机排序后的行号关联结果。

## 4. 预先固定的模型与选择规则

### 4.1 三个系统

| 系统 | 输入与结构 | 训练目标 | 测试时输出 |
|---|---|---|---|
| B0 | 词 1–2 gram TF-IDF + 多类逻辑回归 | 意图分类 | 150 类概率；MSP 拒绝分数 |
| BERT-CE | bert-base-uncased + 150 类意图头 | 意图 CE | 150 类 logits、最终层 CLS |
| BERT-Dom | 同一 BERT + 150 类意图头 + 10 类领域头 | 意图 CE + λ × 领域 CE | 主结果只用意图头；领域头保留作补充分析 |

模型共同表征使用 `last_hidden_state[:, 0, :]`，不要将 `pooler_output` 与原始 CLS 混用。训练时可将一个共同的 dropout 输出同时送入两个线性头，避免领域分支额外 dropout 消耗随机数影响主分支的可比性。提取检测特征时使用 `eval()` 模式下未经过分类头的 CLS。

### 4.2 训练默认值

| 参数 | 初始约定 |
|---|---|
| 编码器 | bert-base-uncased，固定模型与 tokenizer revision |
| 随机种子 | 0、1、2，训练与分析种子分开记录 |
| 学习率 | 2e-5 |
| 优化器 | AdamW，weight decay 0.01；bias 与 LayerNorm 不衰减 |
| 调度 | linear decay，warmup 为总步数的 10% |
| 有效 batch size | 32 |
| 最大输入长度 | 64，包含特殊 token；记录截断比例 |
| 训练上限 | 5 epochs |
| 检查频率 | 每个 epoch 完成后评估 IS val macro-F1 |
| 早停 | 连续 2 次评估无提升则停止；严格大于旧值才算提升 |
| 最佳 checkpoint | val macro-F1 最大；完全相同取较早 epoch |
| 梯度裁剪 | max norm 1.0 |
| λ | 0 为 CE；Dom 搜索 0.1、0.3、0.5、1.0 |
| 最终推理 | 所有 BERT checkpoint 使用同一种精度，建议 FP32 |

这些是可实施默认值，新增的优化器细节应先通过 smoke test 并记录，再用于全部正式运行。显存不足时可统一使用 microbatch 16、梯度累积 2；最后一个不足的累积窗口按真实样本数归一化。两组必须使用同一有效 batch、调度和精度。

“相同早停规则”不等于“实际 epoch 数相同”。主实验使用相同上限与选择规则，并报告每次实际步数。若声称严格等更新预算，应额外预先定义固定 epoch 的敏感性比较；不能在报告中把自动早停结果写成等步数实验。

### 4.3 配对初始化

对每个种子，先构建一次 BERT 编码器和意图分类头，并保存共同初始权重。该种子的 CE 和所有 λ 实验均从这份初始化独立开始，不从已经训练完的 CE checkpoint 再续训。

可以使用统一双头代码：CE 将 λ 设为 0，并关闭领域头的优化；Dom 对领域损失反向传播。这有助于避免两个实现路径不一致。先用单 batch 验证 λ=0 路径与意图 CE 单头实现的 logits、loss 和主干梯度一致。

构建模型后重新设定训练 RNG 状态；sampler 使用独立 generator。固定每个 epoch 的训练样本 ID 顺序，保存其哈希。新增领域头的随机初始化不能改变意图头权重或样本顺序。尽可能启用确定性计算，并记录仍存在的 GPU 非确定性。

## 5. E0 数据与领域映射核对

输入：原始 JSON、domains.json、仓库版本。

1. 读取六个分区，确认每条记录是 `[text, intent]`，文本非空、标签有效。
2. 对每个 IS 意图断言 train/val/test 数量为 100/20/30，并确认六个分区的总数。
3. 将领域文件转换成唯一的 `intent -> domain` 映射，断言每个意图只出现一次，10 × 15 完整覆盖。
4. 意图和领域名称按固定排序生成 ID；保存映射，不在每次训练中临时重新编号。
5. 生成逐样本 ID、intent_id、domain_id。OOS 无已知领域标签，使用空值或 -1，不能赋为某个已知领域。
6. 保存文件哈希、完整 Git commit、原始 counts、许可来源和生成时间。

产出：`07_metadata/data_manifest.json`、三份标签映射、数据断言日志。

验收：映射可逆；各 split 标签集合一致；OOS 不进入 150 类训练 loader；所有计数通过。失败时停止训练并检查原因，不修改数据来凑计数。

## 6. E1 领域分析与数据质量审计

1. 输出 split × domain × intent 计数表，以及词数、BERT token 数分布。
2. 记录 max_len=64 的截断条数。训练集明显被截断时，可在开发阶段统一调整长度；test 截断统计只记录，不能据它改配置。
3. 检查精确重复、规范化重复、跨 split 重复与标签冲突。规范化只用于审计，不能静默改变正式模型输入。
4. 复查队友报告的“3 条 train–val、2 条 train–test 重叠、4 条标签冲突”；这些是待核验的既有说法。明确计数单位是唯一文本还是样本对，不要求通过修改数据复现数值。
5. 使用 train 文本建立每领域词集合。词法提取采用固定规则，例如小写后的连续英文字母片段。计算 `J(A,B)=|A∩B|/|A∪B|`，输出 10 × 10 热力图；它是词汇重叠描述，不是语义相似度的直接度量。
6. 仅依据标签定义和 train/val 建立 10–20 个可能混淆的意图对，分同领域和跨领域，记录选取理由。
7. 简述数据的英文、众包、单意图及文化背景限制；检查输出示例是否含不宜公开的内容。

产出：`audit_counts.csv`、`duplicates.csv`、`length_stats.csv`、词汇热力图和 `candidate_confusions.csv`。

验收：数据问题有记录和处理政策；原始 split 未变；图表可由脚本重新生成。

## 7. E2 TF-IDF 基线

1. 默认采用词 unigram/bigram、lowercase=True、min_df=1、L2 归一化；记录 tokenizer、max_df、sublinear_tf 等完整实际配置。
2. vectorizer 只在 15,000 条 train IS 上 fit，val/test 只 transform。
3. 训练多类逻辑回归，固定求解器及软件版本，设置足够迭代次数并检查是否收敛。
4. 搜索 `C ∈ {0.1,1,10,100}`，统一使用 IS val macro-F1 选择；完全相同时取较小 C。BERT checkpoint 和 λ 也按 IS val macro-F1 选择。Accuracy 仅作次要报告指标，不参与选 C。
5. 保存验证 logits/概率、预测、意图映射及选择依据。
6. B0 的 OOS 仅使用 MSP，不对稀疏 TF-IDF 强行复用 BERT 的 CLS Mahalanobis 流程。

B0 默认报告一次固定运行；若求解器具有随机性则固定 random_state。重复同一确定性模型三遍不提供独立不确定性证据。三种子要求主要针对 BERT。

验收：未 fit 验证或测试文本；收敛无警告或已解释；可从保存的 pipeline 复现预测。

## 8. E3 BERT-CE

1. tokenizer 对原文本分词；不删除停用词或标点，以保留意图线索。
2. DataLoader 返回 input_ids、attention_mask、intent_id、domain_id、sample_id。
3. 先完成两个小测试：一个 batch 的前向/反向；小训练子集上检查模型能降低损失。测试运行单独标为 smoke，不进入正式统计。
4. 从 seed 0 的共同初始化开始训练，记录逐 epoch loss、val macro-F1、accuracy、两种错误率、时间和显存。
5. 保存最佳 checkpoint；按统一规则完成 seed 1、2。
6. 最佳模型以 eval 模式导出 train 和 val 特征，用于后续检测器拟合。

复用队友 baseline 的条件：模型 revision、tokenizer、初始权重、意图头、种子、数据顺序、batch、优化器、训练上限和 checkpoint 选择规则均可核验一致。仅仅“同样用了 BERT”不满足配对要求；不匹配时将其视为历史参考，重训正式 CE 对照。

验收：三个运行均有配置、日志、checkpoint 选择记录与数据版本；没有 test 指标参与选择。

## 9. E4 BERT-Dom 与 λ 选择

### 9.1 核心实现

```text
h = BERT(input_ids, attention_mask).last_hidden_state[:, 0, :]
h_train = shared_dropout(h)
intent_logits = intent_head(h_train)    # [batch, 150]
domain_logits = domain_head(h_train)    # [batch, 10]
loss_intent = CE(intent_logits, intent_id)
loss_domain = CE(domain_logits, domain_id)
loss = loss_intent + lambda * loss_domain
```

损失分别记录，不用总 loss 直接比较不同 λ。领域 CE 只对 IS 标签计算。正式推理不使用领域头筛选候选意图，否则会同时改变训练与推理策略，超出本次单一改进。

### 9.2 实验矩阵与选择

| 方法 | λ | 种子数 | 正式训练次数 |
|---|---|---:|---:|
| BERT-CE | 0 | 3 | 3 |
| BERT-Dom | 0.1、0.3、0.5、1.0 | 各 3 | 12 |
| 合计 | 5 个 λ 水平 | 3 | 15 |

每个 λ、每个种子从对应的共同初始权重开始。对不同 λ 使用同一训练和 checkpoint 选择流程。

令 `m(λ)` 为该 λ 的三个最佳 checkpoint 在 val 上的 macro-F1 均值。先找到正 λ 中的最高均值 `m_best`，再从满足 `m(λ) >= m_best - 0.002` 的正 λ 中取最小值。这里 0.002 是比例尺度，相当于 0.2 个百分点。最终整个项目只选一个 λ，不按种子分别选。

λ=0 是对照并绘入敏感性曲线，不属于正 λ 的选择集合。如果所有正 λ 都弱于 CE，也照实报告已选 Dom 的负结果，不改规则隐藏它。

同一 val 被用于 checkpoint、λ 和其他开发选择，选中的 val 数字可能偏乐观。验证结果只用于描述选择过程和参数敏感性，不作为无偏泛化性能估计，也不将其上的提升作为最终有效性证据。冻结后的 test 评估不参与这些选择；重复使用 val 本身不等于 test 泄漏。

产出：12 份训练记录、`lambda_summary.csv`、带三种子离散情况的 λ 曲线、`selected_lambda.json`。

验收：每个种子内共享初始化哈希；各 epoch 样本顺序匹配；没有改变主预测规则；λ 选择可由结果表自动重算。

## 10. E5 OOS 检测与配置冻结

### 10.1 两种分数

统一约定分数越大越像 OOS，`score >= threshold` 表示拒绝。

MSP：`score = 1 - max(softmax(intent_logits))`，仅使用 150 类意图头。

Mahalanobis 按每个训练好的 BERT 模型独立拟合：

1. 在 eval 模式提取该模型 15,000 条干净 train IS 的 CLS。
2. 对每个意图 c 计算类均值 `μ_c`。
3. 对每条 train 表征计算类内残差 `r_i = h_i - μ_yi`。
4. 对这些残差使用 Ledoit–Wolf 估计共享类内协方差 Σ。不能直接对未去除各类均值的全部 CLS 估计协方差，否则混入类间差异。
5. 计算 `d_c(x) = (h(x)-μ_c)^T Σ^-1 (h(x)-μ_c)`，OOS 分数为 `min_c d_c(x)`。
6. 保存均值、协方差/精度矩阵、拟合样本哈希及数值精度。检测统计建议使用 float64，避免混淆数值误差和模型差异。

CE 与 Dom 使用相同算法和数据来源，但各自拟合检测参数；因此 OOS 比较是完整流程的比较，不能单独归因于某个协方差或阈值变化。

### 10.2 阈值选择

对每个模型、种子、检测器使用同一程序：

1. 计算 3,000 条 IS val 与 100 条 OOS val 的分数。
2. 候选阈值包含验证分数的唯一取值及必要边界，明确处理相等分数。
3. 保留实际 IS 接纳率至少 0.95 的阈值，再最大化 OOS val recall。
4. 若 OOS recall 并列，优先取 IS 接纳率最高的阈值；仍并列时取最小可行阈值，确保程序可重复且不为相同 OOS 收益额外拒绝 IS。
5. 保存 IS acceptance、OOS recall、阈值和 tie-breaking 结果。

对于分数单调的拒绝器，降低阈值会同时增加 IS 拒绝和 OOS 召回。阈值选择只是在验证集上满足约束并处理并列，不要把它夸大为复杂的 OOS 参数优化。100 条 OOS val 的召回以 1 个百分点为单位变化，讨论其小样本限制。

若做阈值稳定性补充，应对 IS val 和 OOS val 一起重采样并重新选阈值；只重采样 OOS 往往看不到由 IS 分位数造成的不确定性。

### 10.3 冻结内容与许可

保存 `01_configs/04_frozen.yaml` 和 `00_docs/01_freeze_record.md`，包含数据/映射哈希、全部模型和 checkpoint、所选 λ、种子、检测器参数、阈值、指标定义、bootstrap 方法、个例抽样规则、测试历史、可选分析是否启用。

按导师原方案，在实现阶段提交 `frozen-v1` 版本给导师确认后再运行 E6。这是未来的执行检查点，不妨碍现在完成计划文件。

## 11. E6 最终评估与统计

### 11.1 预测导出

冻结后评估 B0 一次、BERT-CE 三个 checkpoint、所选 λ 的 BERT-Dom 三个 checkpoint。其他 λ 不进入 test 比较。

每条预测保存：`sample_id, split, gold_intent, gold_domain, pred_intent, pred_domain_from_intent, msp_score, mahalanobis_score, accepted_msp, accepted_mahalanobis, seed, model, checkpoint_hash`。OOS gold_domain 留空。原文可通过 sample_id 回查，避免在多个日志重复保存。

BERT 的一次特征/logits 导出即可支持两种检测分数，避免为每个指标反复推理。

### 11.2 指标定义

令 N 为全部 IS 样本数，g 为固定意图到领域的映射：

| 指标 | 明确定义 |
|---|---|
| IS macro-F1（主要指标） | 拒绝前，对全部 150 个意图的 F1 等权平均，固定完整标签集合，zero_division=0 |
| Accuracy | 正确意图预测数 / N |
| 跨领域错误率 | `count(g(pred) != g(gold)) / N` |
| 同领域错误率 | `count(pred != gold and g(pred) == g(gold)) / N` |
| 领域内意图 macro-F1 | 每领域的 15 个意图在完整 IS 测试集上的逐类 F1 均值；保留跨领域误报对 precision 的影响 |
| 领域分类 F1 | 把预测意图映射为领域后，对 10 类计算 F1；不能与上一项混称 |
| IS 误拒率 | 被拒绝的 IS 数 / N，不论原意图预测是否正确 |
| OOS recall | 被拒绝的 OOS 数 / 全部 OOS 数 |
| OOS 误接纳率 | 被接纳的 OOS 数 / 全部 OOS 数，等于 1−OOS recall |

自动检查 `accuracy + 跨领域错误率 + 同领域错误率 = 1`。除以全部错误数的“错误构成比例”作为另一张图，明确分母，不与上述错误率混用。

AUROC 使用 OOS 为正类，输入第 10 节的 OOS 分数 `s`（越大越像 OOS），报告名为 `AUROC_OOS`。

**FPR@95TPR 的两个方向均输出，主报告使用 IS 正类。** 主指标为：TPR 是 IS 接纳率，FPR 是 OOS 误接纳率；给 ROC 的接纳分数为 `a=-s`（越大越像 IS）。定义为：在 IS 接纳率达到至少 95% 的工作点，OOS 被误接纳的比例，越低越好。报告列名用 `FPR@95TPR (IS positive)`，机器字段用 `fpr95_is_positive`。

补充指标使用 OOS 正类和原 OOS 分数 `s`：在 OOS 检出率至少 95% 的工作点，IS 被误拒的比例，机器字段用 `fpr95_oos_positive`，并保存实际 OOS recall。两个方向必须分别命名，不得在看到 test 结果后更换主指标。

实现时对全部唯一分数阈值计算 ROC（不丢弃中间点），选择 TPR 首次达到至少 0.95 的工作点，不插值；分数相同的样本作为整体处理，记录实际达到的 TPR。这些值是测试分数曲线上的标准 benchmark 描述指标，不将测试阈值用于部署或回头更新模型。

另外单列 `OOS_FA@val95`：使用 E5 在 val 上按 IS 接纳率至少 95% 冻结的阈值，计算 test OOS 误接纳率，并同时报告 test 的实际 IS 接纳率。它与 IS 正类 FPR95 的业务方向一致，但阈值来源不同，数值不必相等。前者反映固定策略的实际表现，后者比较测试曲线上的标准工作点。两者不得共用同一个字段名。

具体正类、符号、边界和验算例子见 `00_docs/03_metric_definitions.md`。2026-10-24 在 E6 前确定 IS 正类为主指标，OOS 正类作为补充。

### 11.3 统计方案

1. BERT 报告三个种子的均值和样本标准差（ddof=1），保留每种子的原始数值。B0 不生成虚假的三种子标准差。
2. 对每个种子计算 `Δ_s = F1(Dom_s) - F1(CE_s)`，主差值为三个 Δ_s 的均值。
3. 遵循导师方案，进行 1,000 次配对 bootstrap；分析 RNG 单独固定为 20261024。
4. 每次在每个真实意图内有放回抽取 30 条 IS test 样本，组成一份分层重采样测试集。CE、Dom 和全部种子共享同一组抽样索引。
5. 每次重新计算三个配对差值，再取均值；用 2.5% 和 97.5% 分位数构成区间。
6. 该区间条件于三个已拟合模型和已选 λ，主要反映测试样本抽样变异，不包含重训、选 λ、历史测试暴露的全部不确定性。种子波动另用标准差报告，不把样本数和种子数简单相乘当作独立观察。
7. 若主差值区间跨 0，写“证据不足以确认提升”，不能写“两方法等价”；跨领域错误减少也不自动证明总体分类改善。

若拓展 OOS 差值区间，按 IS/OOS 分层重采样，保持两模型配对；作为次要探索性结果，不以大量指标中偶然显著的一个替换主结论。

### 11.4 敏感性分析

基于保存预测，排除精确 train–test 重叠的测试行后重算主要指标，记录剩余样本数。保留官方全量结果为主要 benchmark 表，单列敏感性结果。它只检查重叠行的影响，不声称消除训练暴露或历史测试调参。

验收：各 checkpoint 的测试样本集合完全一致；标签顺序正确；拒绝阈值未改；所有图表由冻结预测生成。

## 12. E7 领域错误分析

1. 从意图预测映射得到 CE、Dom 的 10 × 10 领域混淆矩阵；输出原始计数、按真实领域归一化比例以及 Dom−CE 差值。
2. 比较每领域的 15 类意图 macro-F1，并另报 10 类领域 F1，以区分“认对领域”与“认对意图”。
3. 堆叠图使用正确、同领域错误、跨领域错误三个互斥类别，三者之和为 100%。另有错误构成图时注明它只对错误样本归一化。
4. top-15 意图混淆对以预先确定的 CE 三种子平均错误数排序，Dom 使用同一列表，不分别挑各自有利的意图对。
5. 个例展示固定使用 seed 0，并明确不是代表所有种子。分别从 CE 错/Dom 对、CE 对/Dom 错的样本池按固定分析种子抽取 5 条；不足时全部报告。每条附真实/预测意图、领域、MSP、句子及解释。
6. E1 的人工混淆清单与测试后观察分开呈现。测试后的例子用于解释，不用于新模型修改。
7. 若跨领域错误下降、同领域错误上升，讨论辅助任务是否将有限表示能力偏向粗粒度区分。若只有个别领域收益，报告范围，不写普遍提升。

产出：领域混淆图、领域意图 F1 表、错误构成图、top-15 表及 10 条修复/退化案例。

## 13. E8 OOS 补充与可选扩展

主流程已经包含明确的 OOS 泛化评估，能够提供课程要求的至少一个稳健性/泛化测试。以下按时间选择，不将其全部变成必做任务。

### 13.1 优先补充：OOS 误接纳去向

分别对 MSP 和 Mahalanobis，统计误接纳 OOS 被预测到各领域的数量、占全部 OOS 的比例，以及占已误接纳 OOS 的比例。没有真实 OOS 领域标签，不将预测去向解释成真实领域分布。

### 13.2 可选：领域头分数

若在 E5 之前已确定执行，保存 Dom 的领域 logits，并以 `1-max(domain probability)` 计算补充 AUROC。只有 Dom 有训练过的领域头，不能拿 CE 的随机领域头做对照。该结果不能替代意图分类主结果，也不能在 test 上选新的拒绝策略。

### 13.3 可选：表示可视化

预先选择 2–3 个领域、固定样本 ID 和随机种子，绘制 CLS 特征。两个模型分别拟合的 t-SNE/UMAP 坐标不具备直接数值可比性；图只作探索，不根据视觉簇距宣称统计改进。

### 13.4 留一领域暂列未来工作

它不是本项目必需项。若以后执行，需移除该领域全部训练样本和 15 个意图，将意图头改为 135 类、领域头改为 9 类，并预先定义未见领域验证/评估及阈值选择流程。不能保留一个从未获得正样本的第十领域头后直接解释其概率，也不能用留出领域调阈值后再宣称对该领域完全未知。

## 14. 资源、时间与执行优先级

导师原稿按六周安排。此前作业说明列出的截止日期为 2026-11-01；若该日期仍有效，不能按六周安排执行。采用下面按依赖顺序推进的阶段计划，并在首次 GPU smoke run 后根据真实耗时安排每日工作。

| 阶段 | 执行内容 | 退出条件 |
|---|---|---|
| A | 环境、E0、E1、指标单元测试 | 数据和映射完整，指标口径通过手算样例 |
| B | E2、E3 seed 0、Dom smoke run | 基线跑通、配对初始化与训练顺序通过检查 |
| C | E3/E4 的正式 15 个运行 | 参数搜索完成，λ 规则自动给出结果 |
| D | E5 检测器和冻结说明 | 阈值、配置与证据固定，完成导师确认 |
| E | E6、E7 | 保存预测、主表、置信区间和错误分析 |
| F | 写作、代码复现、个人演示 | 仓库/ZIP/报告一致，展示有贡献证据 |

预算先用一个代表性 epoch 实测：若一个运行需 t 小时，15 次完整运行的串行训练约为 15t，再加审计、检测、统计、失败重跑和写作时间。不要在无实测时承诺所有运行能在固定小时内完成。

算力不足时优先保留三种子 CE 与 Dom、公平对照、OOS、错误分析；先取消 E8 可视化等可选内容。若仍需缩小 λ 网格，应在查看 test 前请导师确认，并清楚记录实际搜索范围，不能在结果出来后只保留表现好的种子。

## 15. 待实现命令接口

下面是建议 CLI 契约，需要代码完成后逐条验证。所有命令均从 `mfe310code/` 根目录运行。Linux 上的数据路径通过 `DATA_ROOT` 指定，模块通过 `PYTHONPATH=03_src` 导入，避免把本机盘符写死。Windows 本地运行时设置对应环境变量；正式 GPU 作业由 `05_jobs/` 下脚本管理。

```bash
export PYTHONPATH=03_src
python 04_steps/00_check_environment.py
python 04_steps/01_prepare_and_audit.py --data-root "$DATA_ROOT" --output 07_metadata
python 04_steps/02_train_baseline.py --config 01_configs/02_baseline.yaml
python 04_steps/03_train_bert.py --config 01_configs/01_base.yaml --lambda-domain 0 --seed 0
python 04_steps/03_train_bert.py --config 01_configs/01_base.yaml --lambda-domain 0.1 --seed 0
bash 05_jobs/01_train_all.sh
python 04_steps/04_select_lambda.py --config 01_configs/03_search.yaml
python 04_steps/05_fit_detectors_and_freeze.py --config 01_configs/03_search.yaml
bash 05_jobs/02_evaluate_frozen.sh
python 04_steps/07_bootstrap.py --config 01_configs/04_frozen.yaml
python 04_steps/08_analyze_domains.py --config 01_configs/04_frozen.yaml
python 04_steps/09_demo_inference.py --config 01_configs/04_frozen.yaml --text "what is my account balance"
```

`05_jobs/01_train_all.sh` 枚举 5 个 λ × 3 个种子，并按显存限制串行或受控排队；已完成且哈希匹配的运行跳过，防止 smoke 命令之后重复训练。`05_jobs/02_evaluate_frozen.sh` 调用 `04_steps/06_evaluate_test.py`，仅加载 B0、三个 CE、三个选定 Dom，检查冻结记录和文件哈希后导出最终预测。正式脚本须将导师确认状态作为显式检查条件，不因文件名含 frozen 就自动假定已经确认。

主深度实验按课程要求运行在获批准的远程 Linux GPU 上。若导师明确批准本地 GPU 例外，保留该批准依据；不能仅因队友报告写了例外就假定自动适用于本项目。记录 Python、PyTorch、CUDA、驱动、GPU、操作系统、依赖锁文件、命令、作业脚本、时间及显存。

## 16. 实施验收与结果解释

### 16.1 必须通过的程序检查

- 数据计数、领域映射唯一性、OOS 排除检查。
- 训练 fit 仅接触 train，验证选择不读取 test 结果。
- 同一种子 CE/Dom 初始化和样本顺序核对；λ=0 的 loss/梯度正确。
- 领域标签映射与意图预测映射使用同一份文件。
- 用小型手算样例检查 macro-F1、两类错误率及拒绝分数方向。
- Mahalanobis 使用类内残差拟合协方差；数值有限，检测器未拟合 val/test。
- 阈值在验证集满足约束，测试阶段只加载，不重新优化。
- bootstrap 共享配对索引，报告单位明确区分比例与百分点。
- 每个报告数字可回溯到配置、checkpoint、预测文件及生成脚本。

### 16.2 预设结论写法

| 观察 | 可以支持的解释 |
|---|---|
| macro-F1 提升且跨领域错误下降 | 领域辅助任务在当前数据/协议下可能改善意图识别，并与跨领域错误减少一致 |
| 跨领域错误下降但 macro-F1 不变/下降 | 存在粗粒度与细粒度区分的权衡，不能宣称整体改进 |
| 分类改善但 OOS 下降 | 封闭集分类和开放请求拒绝有不同需求，领域监督不保证两者兼得 |
| 各种子方向不一致或区间跨 0 | 当前证据有限，需要更多运行或独立验证 |
| 全部无明显改善 | 如实呈现假设未获支持，分析层级标签冗余、任务权重和数据规模等可能原因 |

这些解释是待结果检验的框架，不是事后必须套用的结论。披露小组此前的测试访问与本项目设计未使用 test 的事实，以及共享代码、小种子数和验证选择偏乐观的限制。

## 17. 报告、展示与提交材料

按原作业说明，最终英文报告为 2,500 词 ±10%，参考文献不计入；当前实施计划不受该字数约束。建议正文约 2,500 词：问题和目标 250、数据与治理 300、方法和理论 450、受控设计 400、结果和误差分析 700、伦理/限制/结论 400。具体格式遵循最终报告模板。

主报告优先保留六组证据：双头结构图；数据/切分表；λ 验证敏感性图；主结果与差值区间；领域混淆差值/错误构成图；代表性修复与退化案例。更多参数、日志和补充分析放在允许的附录或仓库，不挤占核心论证。

每位成员展示不超过 5 分钟、8 页幻灯片；说明自己的实现和分析，展示 IS 与 OOS 推理例子，解释一个成功与一个失败案例。共用事实可以一致，个人叙述与贡献证据应明确。

提交材料：英文 PDF 报告、可访问代码仓库、固定 ZIP、依赖配置、英文注释/docstring、命令/日志/结果文件、个人展示、AI 使用声明。声明 AI 实际用途与核验方式，不将计划中的实验写成已运行，不捏造结果或引用。

`00_docs/evidence_map.json` 建议按每个图表记录 `claim_id, source_prediction, config_hash, checkpoint_hash, script, output_path`。贡献表记录共同模块作者、个人模型/分析代码及对应提交，不仅列出文件名。最终图表、报告材料及冻结 ZIP 统一存入 `10_exports/`，打包时排除 ZIP 自身、缓存和不允许分发的文件。

## 18. 来源与实施时应核对的参考资料

本计划依据导师提供的 `CLINC150_domain_auxiliary_plan.md` 细化；其中 E0–E8、BERT-Dom 目标、λ 网格、三种子和主指标保持原方案。新增的参数默认值、统计细节和验收流程应随冻结配置一同留档。

修订 2 已统一 B0/BERT 的选择指标为 IS val macro-F1，改用 IS 正类的 FPR@95TPR，补充验证集重复选择的解释，并根据本项目设计过程的澄清更新测试访问披露。

- 导师原稿：`F:/wechat_download/xwechat_files/wxid_n4nk3gnstj8b12_6e19/msg/file/2026-10/CLINC150_domain_auxiliary_plan.md`（仅作本地来源记录，公开报告无需包含私人路径）。
- 数据与许可：<https://github.com/clinc/oos-eval>，本地固定提交见第 2 节。
- Larson 等：*An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*，<https://aclanthology.org/D19-1131/>。
- BERT：*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*，<https://aclanthology.org/N19-1423/>。
- FPR95 的 IS 正类方向参考 ODIN：*Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks*，<https://arxiv.org/abs/1706.02690>；报告中仍显式标出正类，避免不同实现的约定混淆。
- Mahalanobis 检测与 Ledoit–Wolf 收缩：实施前核对所采用方法的原始论文及库文档，在报告中准确引用实际实现。
- 课程作业说明与评分细则：本地 `coursework/` 中提供的原始 PDF；以导师确认的更新安排为准。
