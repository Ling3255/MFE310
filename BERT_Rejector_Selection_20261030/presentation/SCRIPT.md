# 个人视频讲稿（Yitong Ling，≤ 5 分钟，第四版）

第四版的主线分两问：**系统凭什么拒绝请求**（三种分数、分布拟合、验证集阈值），**为什么选这个方法**（分离能力、噪声表现、校准对照、阈值稳定性，最后按错误成本选择）。每一问结束时用一句话给出结论。第 1 段与前几版相同。

## 结构

| 段 | 作业要求 | 内容 | 时长 | 英文词数 |
|---|---|---|---|---|
| 1 | 个人信息与项目 | 问题、场景、数据集 | 30 秒 | 约 60 |
| 2 | 数据流程与模型 | 划分、噪声、基准、BERT、受控改进及其结果；拒识是分类之后单独的一步 | 30 秒 | 约 75 |
| 3 | 技术解释（一） | **系统凭什么拒绝**：三种分数、Mahalanobis 拟合、阈值为什么只用验证集 | 60 秒 | 约 140 |
| 4 | 实验结果 | **比较**：分离能力 → 噪声下的表现与代表性错误 → 校准对照 → 阈值稳定性 | 60 秒 | 约 145 |
| 5 | 技术解释（二） | **为什么选这个方法**：错误成本、选择结果、选择规则 | 40 秒 | 约 100 |
| 6 | 推理演示 | 两句话，两种分数各漏一句 | 25 秒 | 约 55 |
| 7 | 个人贡献 | 我负责的部分 | 15 秒 | 约 35 |
| 8 | 分析与反思 | 原因、局限、负责任使用、下一步 | 40 秒 | 约 85 |

正文约 695 词：第 1 段配 30 秒视频，其余约 635 词按每分钟 145 词约 4 分 23 秒，全片约 4 分 53 秒。演示时边运行边讲，不要停下等输出；超时就先删第 8 段括号里的可删句。

## 每个内容只出现一次

| 内容 | 所在段 |
|---|---|
| 数据划分、噪声规则、准确率与拼写增强的提升 | 2 |
| 三种分数的算法、Mahalanobis 的拟合、训练 / 验证 / 测试各自的用途、两种阈值策略 | 3 |
| AUROC、F1 差值、噪声下的召回与正确接纳、温度缩放（报告的 F1 下降与补充的同接纳率比较）、bootstrap | 4 |
| 代价公式、24 个设置的选择结果、MSP 的条件性例外、选择规则（组合分数的阴性结果只写在 `docs/FINDINGS.md`，视频不讲） | 5 |
| Android 与 stamps 两句 | 6 |
| 负责范围 | 7 |
| 为什么会这样、局限、负责任使用、下一步 | 8 |

## 录制前必须处理

1. **第 7 段说的负责范围要有证据。** AlexXu23 仓库只有两个提交，作者都是 AlexXu23，看不出每个文件是谁写的。按内容，拒识代码在 `CLINC150_Team_20261019/src/evaluate_oos.py`（及 `test_oos_math.py`、`verify_oos_results.py`）和 `BERT_Rejection_Study_20261019/` 的 8 个脚本里。只说你确实写的部分，并准备好原始文件和时间、聊天或提交记录。参考分布混合、工作点对齐的四个文件夹（`BERT_Rejection_Ratio`、`_Operating_Points`、`_Alignment`、`BERT_Mixed_Rejection`）不属于拒识器本身。
2. **演示尽量做成实时推理**：拿到 seed 42 的 `model.safetensors` 和 `distance_model.npz` 后用 `demo_rejector.py`，否则用 `demo_replay.py` 和回放版台词。
3. **数字**来自共同报告（Table 5A、5B、7）和本文件夹的 `outputs/`，录制时不要改。
4. **哪些是报告外的补充分析**：阈值 bootstrap（51–74% / 80–86%）、按代价选择（24 个设置、0.7 pp regret、MSP 的条件性改选、组合分数阴性结果）、同接纳率下的温度缩放比较（16 pp）都不在当前报告正文中，讲稿里已用 "follow-up" 标明。如果组里采纳了 `docs/REPORT_EDITS.md` 的修改，这些就会进入报告。
5. **模型对应**：第 4 段的 AUROC、F1 来自原始 BERT；噪声下的召回、正确接纳和误拒来自拼写增强 BERT（报告 Table 7），台词已写明。

## 1. 个人信息与项目介绍（约 30 秒）

**画面：** 姓名、学号、NLP 问题、应用场景和数据集的信息卡，然后是报告标题页。

> Hello, I'm Yitong Ling, student ID 2362615. Our NLP problem is intent classification with out-of-scope rejection under spelling errors. The intended application is a customer-service assistant: it must route a supported request to the right intent and refuse requests it cannot handle, even when users mistype. Our dataset is CLINC150: 150 intents across ten domains, plus 1,000 out-of-scope test requests.

## 2. 数据流程与模型（约 30 秒）

**画面：** 流程图：CLINC150 → 拼写噪声（light / moderate）→ 分类器（TF-IDF + LR 或 BERT）→ 拒识器 → 接受或拒绝。讲到最后一句时，拒识器那一格高亮。

> We keep CLINC150's original split: 15,000 training, 3,000 validation and 4,500 test in-scope requests. Spelling noise deletes or swaps letters inside longer words, at two levels. The baseline is TF-IDF with logistic regression; the principal model is fine-tuned BERT over three seeds. Under moderate noise BERT falls below the baseline, 65.3 against 67.6 percent accuracy, and our controlled improvement, spelling augmentation, recovers 6.0 and 11.5 points. My part is the step after classification: the rejector.

## 3. 系统凭什么拒绝请求（约 60 秒）

**画面：** 三栏示意图，每栏一种分数及其计算方式；然后放 `outputs/score_distributions.png`；最后一张数据用途图，三栏：训练集（15,000 条）→ 拟合分类器和 Mahalanobis 分布；验证集（3,000 条 IS + 100 条 OOS）→ 选阈值和方法；测试集 → 只用于报告结果。

> Each request gets a score: a larger score means more likely out of scope, and the system rejects the request once its score reaches a threshold. MSP is one minus the highest class probability; entropy measures how spread the 150 probabilities are. Both read only the classifier's output. Mahalanobis reads BERT's 768-dimensional sentence vector instead: it uses the 150 intent centres and one shared covariance, shrunk by Ledoit–Wolf, and scores the squared distance to the nearest centre. The classifier and this distribution are fitted on training data only; thresholds and method choices use validation only, and the test set takes no part in these choices. Validation has only 100 out-of-scope requests, so I compare two threshold policies: maximising out-of-scope F1, which depends on those 100, and accepting 95 percent of in-scope requests, which mainly constrains the acceptance of normal requests. So the system rejects a request when its output is uncertain, or when its representation lies far from every known intent.

## 4. 比较：分离能力、噪声、校准与稳定性（约 60 秒）

**画面：** 依次放：AUROC 与 F1 区间的小表；报告 Table 7 的 Moderate 行；`outputs/calibration_vs_separation.png`；`outputs/threshold_stability.png`。

> On clean test data, Mahalanobis separates best: AUROC 0.976, against 0.954 for entropy and 0.945 for MSP. With F1 thresholds it beats MSP by 6.8 F1 points, interval 5.3 to 8.3, but entropy by only 0.3, an interval crossing zero. On the spelling-augmented model under moderate noise, it still catches the most out-of-scope requests, 91 against 86 percent for entropy, but accepts fewer valid ones correctly, 60 against 64, and false rejection reaches 29 to 35 percent. Temperature scaling cuts calibration error from 35.6 to 0.85 percent, yet MSP's F1 falls from 67.8 to 58.9; my follow-up at equal acceptance shows 16 points more out-of-scope requests getting through. A follow-up bootstrap of validation moves entropy's recall between 51 and 74 percent under F1 thresholds, but only 80 to 86 under the acceptance rule. So no single metric decides; the threshold policy matters as much as the score.

## 5. 为什么选这个方法（约 40 秒）

**画面：** 代价公式，然后 `outputs/method_choice_heatmap.png`，最后一张写着选择规则的文字卡。

> So, in a follow-up analysis, I choose by the cost of false rejection versus false acceptance, over twelve settings per model, on validation only. In all 24 settings, validation chooses Mahalanobis; cost only moves its threshold, from the F1 point down to 90 percent acceptance, with at most 0.7 points of test regret. Only when accepting a misclassified request is also charged, out-of-scope requests are rare and the cost weight is high does validation switch to MSP, which best flags the classifier's own mistakes. My rule: Mahalanobis with an acceptance-based, cost-chosen threshold; MSP only under those conditions.

## 6. 推理演示（约 25 秒）

**画面：** 终端。有模型文件时运行实时推理：

```powershell
& .\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe BERT_Rejector_Selection_20261030/demo_rejector.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

没有模型文件时运行回放：

```powershell
& .\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe BERT_Rejector_Selection_20261030/demo_replay.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

**实时推理版：**

> BERT reads raw text and predicts an intent; the rejector scores it three ways. "How do I make my Android phone more secure" is predicted as sync device; MSP and entropy would accept it, but Mahalanobis rejects it. "Order me a book of stamps" is the reverse: the probability scores reject it, Mahalanobis accepts it.

**回放版（没有模型文件时）：**

> Our weights were not in the shared snapshot, so this replays saved model outputs. "How do I make my Android phone more secure" was predicted as sync device; MSP and entropy accept it, but its Mahalanobis distance, 3,885, passes the threshold of 3,661. "Order me a book of stamps" is the reverse.

## 7. 个人贡献（约 15 秒）

**画面：** `BERT_Rejector_Selection_20261030/` 的文件列表和 `outputs/inputs_verification.json` 的 PASS 行。

> I was responsible for this rejection part: the scoring and Mahalanobis code, the validation thresholds, and the follow-up selection analysis, built with an AI coding assistant, which I ran and checked end to end.

## 8. 分析与反思（约 40 秒）

**画面：** 一张文字幻灯片，分"Why""Limitations""Next"三栏。

> Why do the scores disagree? A confidently predicted wrong intent fools the probability scores, while familiar wording fools the distance. Temperature scaling fails because it reorders requests, not just rescales them. The limits are synthetic noise, three seeds and a test set we had already inspected. Responsibly, a rejected request should go to clarification or a human. Future work should evaluate the cost-based method and threshold selection on naturally occurring spelling errors, with more seeds and a fresh, untouched evaluation split.（可删：and check whether a cost-chosen threshold still holds when the out-of-scope rate shifts.）

## 视觉素材清单

| 段 | 素材 | 位置 |
|---|---|---|
| 1 | 30 秒无声视频 | `role_c_video/out/intro_30s_no_audio.mp4`（本地） |
| 2 | 流程图 | 需要新做 |
| 3 | 三种分数示意图；分数分布图；阈值规则图 | 示意图和规则图需要新做；`outputs/score_distributions.png` |
| 4 | 小表；Table 7、8；校准图；稳定性图 | 小表需要新做；共同报告；`outputs/` |
| 5 | 代价公式；热力图；选择规则卡 | 公式和规则卡需要新做；`outputs/method_choice_heatmap.png` |
| 6 | 终端录屏 | 现场录制 |
| 7 | 文件列表 | GitHub 页面 |
| 8 | 反思幻灯片 | 需要新做 |
