# 角色 C 对共同报告的修改稿

对象：`MFE310TC_Group_Report.pdf`（10 页，正文约 3,430 词）。

## 修改原则

- **只改角色 C 的文字。** 范围是 IV.C（拒识分数与阈值）、V.C 中关于分数比较和温度缩放的句子。
- **不动别人的部分。** 不改 V.C 中引用 Table 7 的句子（属于角色 D），也不改 Table 3–8、摘要、结论、局限和附录。
- **正文词数基本不变。** 正文净增约 13 词，约 0.4%。新增的一张图 Fig. 3 的图注约 34 词；如果组里把图注计入字数，就再加上这部分。
- **新数字都可追溯。** 每个新数字都对应 `role_c/outputs/` 里的一个文件，见各处修改的"证据"一栏。
- **符号沿用原报告。** FR 指 IS false rejection，FA 指 OOS false acceptance，与 Table 4 的定义一致。

## 修改 1：IV.C 第一段（熵的定义）

**原文（45 词）**

> The alternative scores in Table 3 use either class probabilities or BERT features. Predictive entropy follows Shannon [5]: [公式] Here pₖ(x) is the probability of intent k; ln denotes the natural logarithm. Concentrated probabilities give lower entropy, whereas dispersed probabilities give higher entropy and favour rejection.

**修改后（22 词）**

> Predictive entropy (Table 3) follows Shannon [5]: [公式] where pₖ(x) is the probability of intent k; dispersed probabilities raise entropy and favour rejection.

**理由**：第一句和 Table 3 的 "Information" 一列重复；"ln 是自然对数"属于常识；高、低熵两种情况说一种就够了。省下的词用于修改 3 和修改 6。公式本身不变。

## 修改 2：IV.C 第二段开头（Mahalanobis）

**原文（21 词）**

> Mahalanobis distance [6] accounts for feature scales and correlations. Following its Transformer OOS application [7], we use the minimum squared distance:

**修改后（17 词）**

> Mahalanobis distance [6] accounts for feature scales and correlations; following [7], we use the minimum squared distance:

**理由**：引用 [7] 的标题里已经写明是 Transformer 上的 OOS 检测，正文不必重复。段落其余部分（h(x)、μ、Σ 和 Ledoit–Wolf 的说明）不变。

## 修改 3：IV.C 第三段第一句之后（选择规则）

**原文（7 词）**

> Initial score comparisons maximise validation OOS F1.

**修改后（21 词）**

> Initial score comparisons maximise validation OOS F1. Cost comparisons choose score and operating point on validation by minimising (1−π)FR + λπFA.

**理由**：原报告只说三种分数"有取舍"，没有给出怎么选。这里补上选择规则，λ 和 π 的取值放在 Fig. 3 的图注里。段落后面关于增强模型阈值的句子不变。

**证据**：`role_c/EXPERIMENT_PLAN.md` 第 8 节；`role_c/outputs/method_choice_by_cost.csv`。

## 修改 4：V.C 第一句（去掉两个 "interval"）

**原文（28 词）**

> For fixed original BERT, Mahalanobis achieves 74.58% OOS F1 under validation-F1 threshold selection: +6.81 pp over MSP, interval [5.27, 8.30], but +0.30 pp over entropy, interval [−0.88, 1.51].

**修改后（26 词）**

> For fixed original BERT, Mahalanobis achieves 74.58% OOS F1 under validation-F1 threshold selection: +6.81 pp over MSP [5.27, 8.30], but +0.30 pp over entropy [−0.88, 1.51].

**理由**：省两个词，数字不变。

## 修改 5：V.C 第二句（为什么不能说优于熵）

**原文（6 词）**

> Superiority over entropy is not established.

**修改后（13 词）**

> Superiority over entropy is not established; this threshold rests on 100 OOS requests.

**理由**：说明区间跨 0 的一个原因。F1 最优阈值只由 100 条 OOS 验证样本决定。bootstrap 重抽验证集后，熵在 test 上的 OOS recall 会在 50.6%–73.5% 之间变化；95% 接纳率策略只在 80.0%–86.2% 之间。这几个区间不写进正文，以控制词数，但在答辩或视频中可以讲。

**证据**：`role_c/outputs/threshold_stability_summary.json`（original、entropy，`test_oos_recall_p2.5` 和 `test_oos_recall_p97.5`）。

## 不改：V.C 中引用 Table 7 的三句

"With augmented BERT (Table 7) … These means express a tradeoff." 保持原样。这部分属于角色 D。

## 修改 6：V.C "Mahalanobis supports…" 一句（怎么选）

**原文（12 词）**

> Mahalanobis supports reference-distribution analysis, rather than being universally best or computationally faster.

**修改后（27 词）**

> Under explicit costs (Fig. 3b), validation selects Mahalanobis when FR and FA are charged; cost only moves its threshold. Charging wrong acceptance favours MSP for rare OOS.

**理由**：把原报告中的"取舍"落到具体的选择规则上。原句说 Mahalanobis 并非处处最好，新句说清楚它在什么条件下不是最优，原句的意思仍然保留。

**证据**：`role_c/outputs/method_choice_by_cost.csv`：

- 只算 FR 和 FA 时（`cost = fr_plus_fa`），两个模型的 24 个格子都选 Mahalanobis。工作点从 max-F1 降到 90% 接纳率，相对 test oracle 的 regret 最大为 0.70 pp。
- 同时计入错误接纳时（`cost = with_wrong_acceptance`），π = 0.05 且 λ ≥ 5 的 4 个格子选 MSP：原始 BERT 两格，拼写增强 BERT 两格。原始 BERT λ = 5 那一格的 bootstrap 选中频率只有 0.42，不稳定。
- 背后的原因见 `role_c/outputs/separation_clean_summary.csv`：区分 IS 错分与正确分类的 AUROC，MSP 为 0.907，Mahalanobis 为 0.892，熵为 0.878（original，test）。

**可选的加长版（+6 词）**：句末改为 "…favours MSP for rare OOS, as MSP best flags misclassified requests."

## 修改 7：V.C 温度缩放一句

**原文（20 词）**

> Temperature scaling [9] reduces clean calibration error from 35.59% to 0.85%, while MSP OOS F1 falls from 67.77% to 58.87%.

**修改后（26 词）**

> Temperature scaling [9] cuts clean calibration error from 35.59% to 0.85% but, at equal 95% acceptance, raises MSP OOS false acceptance by 16.3 pp (Fig. 3a).

**理由**：原句中 F1 的下降是在两个不同的 max-F1 阈值下比较的，阈值变化和排序变化混在了一起。新句把两边 IS 接纳率都固定在 95%，只剩下排序的影响。Fig. 3a 的右图显示，下降来自排序变差：3.0% 的 IS–OOS 句子对变差，只有 0.4% 变好。

**证据**：`role_c/outputs/calibration_summary.json`：

- `scaled_minus_raw_mean.val95_test_oos_false_acceptance_msp = 0.163`；熵为 0.080；误拒率几乎不变（−0.3 / −0.4 pp）。
- `calibration_rank_flips.csv`：MSP 有 3.0% 的句子对变差、0.4% 变好；熵为 2.7% 和 0.7%。
- 原句中的 ECE 数字（35.59% → 0.85%）沿用原报告。
- 这部分分数来自同一批 checkpoint 的历史 FP16 推理，与现有 FP32 分数逐行一致（Spearman 0.999997，`consistency_with_fp32 = PASS`）。

## 新增 Fig. 3

**位置**：放在 Fig. 2 之后（第 V 节末尾），编号为 Fig. 3。这样 Fig. 1、Fig. 2 不需要改号。V.C 先于 Fig. 2 引用了 Fig. 3；如果组里要求严格按引用顺序编号，就和负责排版的同学商量是否改号。

**图片**（两张上下排列，各占通栏宽度）：

- (a) `role_c/outputs/calibration_vs_separation.png`
- (b) `role_c/outputs/method_choice_heatmap.png`

**图注（34 词）**

> Fig. 3 Rejector choice. (a) Temperature scaling calibrates confidence (left) but lowers MSP and entropy AUROC (right); historical FP16 scores of the same checkpoints. (b) Validation-chosen rejector per cost; regret against the clean-test oracle.

如果版面紧，可以只放 (b)，并删去修改 7 末尾的 "(Fig. 3a)"。

## 词数汇总

| 修改 | 原文 | 修改后 | 变化 |
|---|---:|---:|---:|
| 1 IV.C 熵 | 45 | 22 | −23 |
| 2 IV.C Mahalanobis | 21 | 17 | −4 |
| 3 IV.C 选择规则 | 7 | 21 | +14 |
| 4 V.C 第一句 | 28 | 26 | −2 |
| 5 V.C 优于熵 | 6 | 13 | +7 |
| 6 V.C 怎么选 | 12 | 27 | +15 |
| 7 V.C 温度缩放 | 20 | 26 | +6 |
| **正文合计** | 139 | 152 | **+13** |
| Fig. 3 图注 | 0 | 34 | +34 |

按空格分词统计，公式不计。Word 的计数方式可能略有不同，改完以 Word 的统计为准。

## 留给全组决定（角色 C 不改）

以下内容在别人负责的部分，只提建议：

- **第 VI 节局限**：可以加半句 "the covariance estimator was not ablated and temperature was compared at two points only"（约 15 词）。原因是 CLS 特征和 FP32 logits 不在快照里，见 `role_c/outputs/covariance_diagnostics.json`。
- **附录 B**：可以注明 Fig. 3a 用的是历史 FP16 分数。图注里已有这一点，所以不加也可以。
- **组合拒识器**：结果是阴性的（AUROC：combo_max 0.972，combo_mean 0.968，Mahalanobis 单独 0.976；见 `candidate_separation_summary.csv`）。正文不写，以控制词数，在个人视频里讲。

---
---

# 个人视频讲稿（Yitong Ling，≤ 5 分钟，第二版）

本版按作业要求的六项内容重写，替换上一版的 8 段结构。

## 与评分标准的对应

| 段 | 作业要求 | 主要对应的评分项 | 时长 | 英文词数 |
|---|---|---|---|---|
| 1 | 个人信息与项目介绍 | Structure, clarity and delivery（5 分） | 30 秒 | 约 65 |
| 2 | 数据流程与模型 | Technical explanation（5 分） | 40 秒 | 约 90 |
| 3 | 实验结果：定量结果、鲁棒性、代表性错误 | Technical explanation（5 分） | 50 秒 | 约 115 |
| 4 | 推理演示 | Technical explanation, demonstration（5 分） | 35 秒（含运行命令的停顿） | 约 65 |
| 5 | 个人贡献 | Individual contribution and ownership（10 分） | 65 秒 | 约 145 |
| 6 | 个人分析与反思 | Personal critical insight and reflection（10 分） | 75 秒 | 约 175 |

正文合计约 655 词，按每分钟 140 词约 4 分 40 秒，留出约 20 秒给演示时运行命令。标有"（可删）"的句子默认不读。

## 录制前必须处理

1. **小组编号：** 报告封面上没有。把第 1 段的 `[group number]` 换成真实编号。
2. **演示必须是模型推理（作业第 4 项）：** `demo_replay.py` 只是回放已保存的输出，可能被认为不满足 "how the model produces predictions" 这一要求。请尽量向组员要一个模型文件 `CLINC150_Team_20261019/outputs/bert/seed42/best_model/model.safetensors`（约 440 MB）和 `BERT_Rejection_Study_20261019/outputs/seed42/distance_model.npz`，然后改用 `demo_rejector.py` 做实时推理。拿不到时才用回放，并按第 4 段的备用说法讲清楚。
3. **使用 AI 工具的说明：** 第 5 段有一句说明角色 C 的代码是在 AI 编程助手协助下完成的。请按课程的学术诚信规定决定怎么写：如果要求披露就保留；无论如何，不要说成代码全部由自己独立手写。
4. **数字来源：** 讲稿里的数字都来自报告（Table 5A、5B、6、8）和 `role_c/outputs/`，录制时不要改。

## 1. 个人信息与项目介绍（约 30 秒）

**画面：** 一张信息卡，列出姓名、学号、组号、NLP 问题、应用场景和数据集；然后是报告标题页。

> Hello, I'm Yitong Ling, student ID 2362615, from Group [group number]. Our NLP problem is intent classification with out-of-scope rejection under spelling errors. The intended application is a customer-service assistant: it must route a supported request to the right intent and refuse requests it cannot handle, even when users mistype. Our dataset is CLINC150: 150 intents across ten domains, plus 1,000 out-of-scope test requests.

## 2. 数据流程与模型（约 40 秒）

**画面：** 一张流程图：CLINC150 → 加拼写噪声（light / moderate）→ 分类器（TF-IDF + LR，或 BERT）→ 拒识器（打分，与阈值比较）→ 接受或拒绝。

> Our pipeline keeps the original split: 15,000 training, 3,000 validation and 4,500 test in-scope requests. Spelling noise deletes or swaps internal letters in longer words, at a light and a moderate level. The reference baseline is TF-IDF with logistic regression; the principal model is fine-tuned bert-base-uncased over three seeds. Out-of-scope is not trained as a class: a separate rejector scores each request and refuses it above a threshold. Our controlled improvement is spelling augmentation, compared with clean continuation from the same checkpoints, with the same samples and the same 938 updates.

## 3. 实验结果（约 50 秒）

**画面：** 先放 Table 5A 的柱状图（可以直接用项目简介视频里的那张），再放 Fig. 1，最后放 Table 8 中的两行。

> Three results matter most. First, BERT beats the baseline on clean text, 95.0 against 90.6 percent accuracy, but under moderate noise it falls to 65.3, below the baseline's 67.6. Second, the robustness evidence: against matched clean continuation, augmentation adds 6.0 and 11.5 points of accuracy under light and moderate noise, and 4.7 and 6.9 points of correct acceptance. It does not, however, meet our strict clean-retention criterion. Third, representative errors. "Set a 3 minue tiemr" is still predicted as timer but was rejected; augmentation turns it into a correct acceptance. But "can I make a trnasfer btween my accounts" becomes a wrong acceptance as freeze account: fewer rejections also let some errors through.

## 4. 推理演示（约 35 秒）

**画面：** 终端。有模型文件时运行：

```bash
PY=BERT_Rejection_Study_20261019/.venv/Scripts/python.exe
$PY role_c/demo_rejector.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

没有模型文件时运行：

```bash
$PY role_c/demo_replay.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

**实时推理版：**

> Here the model takes raw text, BERT predicts an intent, and the rejector scores the same request three ways. For "how do I make my Android phone more secure", BERT predicts sync device. MSP and entropy stay under their thresholds and would accept it, but its Mahalanobis distance passes the threshold, so it is rejected. For the stamps request it is the other way round.

**回放版（没有模型文件时用）：**

> Our trained weights were not in the shared snapshot, so this replays the model's saved outputs for real test requests. For "how do I make my Android phone more secure", BERT predicted sync device. MSP and entropy accept it, but its Mahalanobis distance, 3,885, passes the threshold of 3,661, so it is rejected. The stamps request shows the reverse.

两句的实际数字在实时推理时会显示在屏幕上，与回放的数字可能有极小差异，讲的时候以屏幕为准。

## 5. 个人贡献（约 65 秒）

**画面：** 依次显示 GitHub 上 PR #9 的提交列表、`role_c/` 目录树、`c0_verification.json` 的 PASS 行、`threshold_stability.png`、`method_choice_heatmap.png`、`rejector_candidates.json`。每一句话都对应屏幕上的一个文件。

> My role was role C: rejector design and selection, and the evidence is in our repository under role_c, pull request 9. First, I took over the inherited rejection pipeline and verified it: 183 checks re-derive every frozen threshold and the five operating points, and all pass. Second, I built the role C analysis with an AI coding assistant: I specified each step, reviewed and ran the code, and checked every output, from a six-step runner and 18 unit tests to the demo. Third, I ran the experiments: score complementarity, a 2,000-draw threshold bootstrap, a calibration comparison and a cost-based selection over 24 settings, and I handed the chosen thresholds to role D. Fourth, decisions: I fixed the selection rules before looking at test results, and when the feature files were missing I chose not to retrain, since new models would no longer match the report.（可删：I also drafted the report's rejection sections and Figure 3.）

## 6. 个人分析与反思（约 75 秒）

**画面：** `calibration_vs_separation.png`，然后一张文字幻灯片，分"Limitations""Responsible use""Next steps"三栏。

> How do I read these results? Mahalanobis separates out-of-scope requests best, with an AUROC of 0.976, but its F1 advantage over entropy, 0.3 points, is smaller than the noise in the threshold itself: with only 100 out-of-scope validation requests, entropy's test recall ranges from 51 to 74 percent across resamples. So choosing the threshold matters as much as choosing the score. Calibration is not rejection either: temperature scaling cuts calibration error to 0.85 percent, yet at equal acceptance it lets 16 points more out-of-scope requests through MSP. And the best score depends on cost: when executing a wrong intent is the main risk, validation prefers MSP, because it best flags the model's own mistakes. The main limitations are synthetic noise only, three seeds, and a test set we had already explored. For responsible use, a rejected request should go to clarification or a human, and sensitive actions need explicit confirmation. Next, I would set thresholds from explicit error costs and recalibrate them on noisy validation data, because false rejection rises sharply under noise, then confirm on natural typos.

## 视觉素材清单

| 段 | 素材 | 位置 |
|---|---|---|
| 1 | 信息卡；项目简介视频的标题画面 | 需要新做，或用 `role_c_video` 重新导出 |
| 2 | 流程图 | 需要新做（一张幻灯片即可） |
| 3 | 准确率柱状图、Fig. 1、Table 8 | 项目简介视频 21.5–35.5 秒；报告 |
| 4 | 终端录屏 | 现场录制 |
| 5 | PR #9 页面、目录树、各输出文件 | GitHub；`role_c/outputs/` |
| 6 | 校准对比图；反思幻灯片 | `role_c/outputs/calibration_vs_separation.png`；需要新做 |
