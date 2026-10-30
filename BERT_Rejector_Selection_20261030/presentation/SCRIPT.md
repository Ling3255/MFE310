# 个人视频讲稿（Yitong Ling，≤ 5 分钟，第三版）

第三版以拒识器的设计与选择为主线，讲完整的实验流程。第 1 段与第二版相同，第 2–6 段重写；每个数字和例子在全片只出现一次。

## 结构

| 段 | 作业要求 | 主线上的位置 | 时长 | 英文词数 |
|---|---|---|---|---|
| 1 | 个人信息与项目介绍 | 问题与场景 | 30 秒 | 约 60 |
| 2 | 数据流程与模型 | 分类之后如何拒识：三种分数、拟合、阈值 | 45 秒 | 约 105 |
| 3 | 实验结果 | 哪种分数分得开；噪声下的鲁棒性；代表性错误 | 50 秒 | 约 120 |
| 4 | 推理演示 | 同一句话，三种分数给出不同决定 | 30 秒 | 约 65 |
| 5 | 个人贡献 | 我写的拒识代码，以及我做的后续分析 | 65 秒 | 约 150 |
| 6 | 个人分析与反思 | 阈值、校准、代价三个判断，局限与下一步 | 75 秒 | 约 170 |

正文合计约 670 词，按每分钟 140 词约 4 分 45 秒。

## 不重复的分配

| 内容 | 只在这一段出现 |
|---|---|
| 数据划分、噪声规则、三种分数的定义、阈值规则 | 2 |
| AUROC、F1 差值及区间、噪声下的准确率与误拒、Table 8 的两个例子 | 3 |
| Android 与 stamps 两句 | 4 |
| 代码文件和分析步骤（不再报结果数字，只有 183 项核验） | 5 |
| 阈值 bootstrap、温度缩放、按代价选择、组合拒识器、局限、负责任使用、下一步 | 6 |

## 录制前必须处理

1. **第 5 段提到的组内代码要有证据。** AlexXu23 仓库只有两个提交，作者都是 AlexXu23，看不出每个文件是谁写的。按内容，拒识相关的代码是下面两处；请只保留你确实写的部分，并准备好能证明的材料（原始文件和时间、聊天或提交记录）：
   - `CLINC150_Team_20261019/src/evaluate_oos.py`（MSP、熵、margin 打分，温度缩放，只用验证集选阈值），以及 `test_oos_math.py`、`verify_oos_results.py`；
   - `BERT_Rejection_Study_20261019/` 下的 8 个脚本：`distance_model.py`（共享协方差 Mahalanobis 检测器）、`run_rejection_study.py`（只用训练集拟合、验证集定阈值、测试前冻结）、`run_typo_rejection.py`（拼写噪声压力测试）、`report_rejection.py`、`report_typo_rejection.py`、`verify_rejection.py`、`verify_typo_rejection.py`、`test_distance_model.py`。
   - 下面几个文件夹属于参考分布混合、工作点对齐和噪声下的对照评估，不是拒识器本身：`BERT_Rejection_Ratio_20261019`、`BERT_Rejection_Operating_Points_20261019`、`BERT_Rejection_Alignment_20261019`、`BERT_Mixed_Rejection_20261019`。不是你写的就不要在第 5 段提。
   - 录第 5 段时，画面停留在 AlexXu23 仓库的这两个文件夹即可，不需要也不要修改那个仓库。
2. **演示尽量做成实时推理。** 拿到 seed 42 的 `model.safetensors` 和 `distance_model.npz` 后用 `demo_rejector.py`；拿不到才用 `demo_replay.py`，并照第 4 段的回放版台词说明。
3. **AI 协助的说法** 保持第 5 段的写法。如果组内那部分代码也用过 AI 工具，同一句话里一并说明。
4. **数字** 来自共同报告（Table 5A、5B、6、7、8）和本文件夹的 `outputs/`，录制时不要改。

## 1. 个人信息与项目介绍（约 30 秒）

**画面：** 姓名、学号、NLP 问题、应用场景和数据集的信息卡，然后是报告标题页。

> Hello, I'm Yitong Ling, student ID 2362615. Our NLP problem is intent classification with out-of-scope rejection under spelling errors. The intended application is a customer-service assistant: it must route a supported request to the right intent and refuse requests it cannot handle, even when users mistype. Our dataset is CLINC150: 150 intents across ten domains, plus 1,000 out-of-scope test requests.

## 2. 数据流程与模型（约 45 秒）

**画面：** 一张流程图，从左到右：CLINC150 → 拼写噪声（light / moderate）→ 分类器（TF-IDF + LR 或 BERT）→ 三种拒识分数 → 与验证集阈值比较 → 接受或拒绝。讲到 Mahalanobis 时，在图上标出"768 维句向量、150 个类中心、共享协方差（只用训练集）"。

> Our pipeline keeps CLINC150's original split: 15,000 training, 3,000 validation and 4,500 test in-scope requests, and spelling noise deletes or swaps letters inside longer words at two levels. TF-IDF with logistic regression is the baseline; fine-tuned BERT over three seeds is the principal model, and spelling augmentation is the controlled improvement. Rejection runs after classification. I compare three scores: maximum softmax probability, predictive entropy, and the Mahalanobis distance from BERT's sentence vector to the nearest of 150 intent centres, using one shared covariance fitted on training data only. Every threshold is chosen on validation, and a request is rejected when its score reaches it.

## 3. 实验结果（约 50 秒）

**画面：** 先放 `outputs/score_distributions.png`（三种分数下 IS 与 OOS 的分布），再放报告 Table 7 中 Clean 与 Moderate 两行，最后放 Table 8 的第 62 行和第 36 行。

> On clean test data, Mahalanobis separates out-of-scope requests best: AUROC 0.976, against 0.954 for entropy and 0.945 for MSP. With F1-optimised thresholds it beats MSP by 6.8 points of OOS F1, interval 5.3 to 8.3, but entropy by only 0.3, an interval that crosses zero. Under moderate noise BERT falls below the baseline, 65.3 against 67.6 percent accuracy; augmentation recovers 6.0 and 11.5 points and, with Mahalanobis rejection, 4.7 and 6.9 points of correct acceptance. Yet noise still lifts false rejection of valid requests to 29 to 35 percent. A representative error: "set a 3 minue tiemr" is classified correctly but rejected; augmentation accepts it, while "can I make a trnasfer btween my accounts" becomes a wrong acceptance.

## 4. 推理演示（约 30 秒）

**画面：** 终端。有模型文件时运行实时推理：

```powershell
& .\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe BERT_Rejector_Selection_20261030/demo_rejector.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

没有模型文件时运行回放：

```powershell
& .\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe BERT_Rejector_Selection_20261030/demo_replay.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

**实时推理版：**

> The model takes raw text, BERT predicts an intent, and the rejector scores the same request three ways. For "how do I make my Android phone more secure", BERT predicts sync device; MSP and entropy stay under their thresholds and would accept it, but the Mahalanobis distance passes its threshold, so the request is rejected. "Order me a book of stamps" is the reverse: the probability scores reject it and Mahalanobis accepts it.

**回放版（没有模型文件时）：**

> Our trained weights were not in the shared snapshot, so this replays the model's saved outputs. For "how do I make my Android phone more secure", BERT predicted sync device; MSP and entropy accept it, but its Mahalanobis distance, 3,885, passes the threshold of 3,661, so it is rejected. "Order me a book of stamps" is the reverse: the probability scores reject it and Mahalanobis accepts it.

## 5. 个人贡献（约 65 秒）

**画面：** 先是 AlexXu23 仓库的两个文件夹：`CLINC150_Team_20261019/src/evaluate_oos.py` 和 `BERT_Rejection_Study_20261019/`。再切到你的仓库的 `BERT_Rejector_Selection_20261030/`：`RUN_README.txt`、`outputs/inputs_verification.json` 的 PASS 行、`outputs/` 文件列表、`rejector_candidates.json`。每句话对应屏幕上的一个文件。

> My own work is the rejector, at two levels. In the shared pipeline I wrote the rejection code: softmax scoring, temperature scaling and validation threshold selection in evaluate_oos, and the Mahalanobis study, which fits the detector, freezes thresholds before testing, stress-tests them on misspelled inputs, and is checked by independent verifiers and unit tests. I then built a follow-up analysis, BERT_Rejector_Selection, with an AI coding assistant: I set the questions and selection rules, ran every stage and checked every output. It re-verifies the inherited pipeline in 183 checks, measures where the three scores disagree, bootstraps threshold stability, compares calibration with separation, and selects a score and threshold for each of 24 cost settings, which I handed to the robustness evaluation. When feature files turned out to be missing, I recorded the limitation instead of retraining, so every number still matches the report.

## 6. 个人分析与反思（约 75 秒）

**画面：** 依次放 `outputs/threshold_stability.png`、`outputs/calibration_vs_separation.png`、`outputs/method_choice_heatmap.png`，最后一张文字幻灯片，分"Limitations""Responsible use""Next steps"三栏。

> What did I learn? First, the threshold matters as much as the score. The F1-optimised threshold rests on only 100 out-of-scope validation requests; resampling them moves entropy's test recall between 51 and 74 percent, far more than the gap between the scores, whereas thresholds set by in-scope acceptance stay within 80 to 86 percent. Second, calibration is not rejection: temperature scaling cuts calibration error from 35.6 to 0.85 percent, yet at equal acceptance it lets 16 points more out-of-scope requests through MSP, because it reorders requests rather than just rescaling them. Third, the right score depends on cost: when executing a wrong intent is the main risk, validation prefers MSP, since it best flags the model's own mistakes. The scores catch different requests, but combining them did not beat Mahalanobis alone. The limits are synthetic noise, three seeds and a test set we had already inspected. Responsibly, a rejected request should go to clarification or a human. Next, I would recalibrate thresholds on noisy validation data and confirm on natural typos.

## 视觉素材清单

| 段 | 素材 | 位置 |
|---|---|---|
| 1 | 30 秒无声视频 | `role_c_video/out/intro_30s_no_audio.mp4`（本地） |
| 2 | 流程图 | 需要新做一张 |
| 3 | 分数分布图；Table 7、Table 8 | `outputs/score_distributions.png`；共同报告 |
| 4 | 终端录屏 | 现场录制 |
| 5 | 两个仓库的文件夹与文件 | GitHub 页面（只浏览） |
| 6 | 阈值稳定性、校准、代价选择三张图；反思幻灯片 | `outputs/`；需要新做一张 |
