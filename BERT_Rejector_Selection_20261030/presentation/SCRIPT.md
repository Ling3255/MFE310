# 个人视频讲稿（Yitong Ling，≤ 5 分钟，第二版）

本版按作业要求的六项内容组织。

## 与评分标准的对应

| 段 | 作业要求 | 主要对应的评分项 | 时长 | 英文词数 |
|---|---|---|---|---|
| 1 | 个人信息与项目介绍（不含组号） | Structure, clarity and delivery | 30 秒 | 约 60 |
| 2 | 数据流程与模型 | Technical explanation | 40 秒 | 约 90 |
| 3 | 实验结果：定量结果、鲁棒性、代表性错误 | Technical explanation | 50 秒 | 约 115 |
| 4 | 推理演示 | Technical explanation, demonstration | 35 秒 | 约 65 |
| 5 | 个人贡献 | Individual contribution and ownership | 65 秒 | 约 145 |
| 6 | 个人分析与反思 | Personal critical insight and reflection | 75 秒 | 约 175 |

正文合计约 655 词，按每分钟 140 词约 4 分 40 秒，留出约 20 秒给演示命令。标有“可删”的句子默认不读。

## 录制前必须处理

1. **演示必须是模型推理：** `demo_replay.py` 只是回放保存输出，可能不满足“模型如何产生预测”的要求。尽量取得 `CLINC150_Team_20261019/outputs/bert/seed42/best_model/model.safetensors` 和 `BERT_Rejection_Study_20261019/outputs/seed42/distance_model.npz`，然后使用 `demo_rejector.py` 做实时推理。拿不到时才使用回放，并明确说明。
2. **AI 工具说明：** 第 5 段披露 AI 编程协助。按课程学术诚信规定保留适当声明，不要说成代码全部由自己独立手写。
3. **数字来源：** 数字来自共同报告和本实验的 `outputs/`，录制时不要改动。

## 1. 个人信息与项目介绍（约 30 秒）

**画面：** 姓名、学号、NLP 问题、应用场景和数据集的信息卡，然后是报告标题页。

> Hello, I'm Yitong Ling, student ID 2362615. Our NLP problem is intent classification with out-of-scope rejection under spelling errors. The intended application is a customer-service assistant: it must route a supported request to the right intent and refuse requests it cannot handle, even when users mistype. Our dataset is CLINC150: 150 intents across ten domains, plus 1,000 out-of-scope test requests.

## 2. 数据流程与模型（约 40 秒）

**画面：** CLINC150 → 拼写噪声 → 分类器 → 拒识器 → 接受或拒绝的流程图。

> Our pipeline keeps the original split: 15,000 training, 3,000 validation and 4,500 test in-scope requests. Spelling noise deletes or swaps internal letters in longer words, at a light and a moderate level. The reference baseline is TF-IDF with logistic regression; the principal model is fine-tuned bert-base-uncased over three seeds. Out-of-scope is not trained as a class: a separate rejector scores each request and refuses it above a threshold. Our controlled improvement is spelling augmentation, compared with clean continuation from the same checkpoints, with the same samples and the same 938 updates.

## 3. 实验结果（约 50 秒）

**画面：** Table 5A 的准确率对比图、Fig. 1，最后放 Table 8 中两行。

> Three results matter most. First, BERT beats the baseline on clean text, 95.0 against 90.6 percent accuracy, but under moderate noise it falls to 65.3, below the baseline's 67.6. Second, the robustness evidence: against matched clean continuation, augmentation adds 6.0 and 11.5 points of accuracy under light and moderate noise, and 4.7 and 6.9 points of correct acceptance. It does not, however, meet our strict clean-retention criterion. Third, representative errors. "Set a 3 minue tiemr" is still predicted as timer but was rejected; augmentation turns it into a correct acceptance. But "can I make a trnasfer btween my accounts" becomes a wrong acceptance as freeze account: fewer rejections also let some errors through.

## 4. 推理演示（约 35 秒）

**画面：** 终端。有模型文件时运行：

```powershell
$PY = "G:\MFE310\mfe310code\.venv\Scripts\python.exe"
& $PY BERT_Rejector_Selection_20261030/demo_rejector.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

没有模型文件时运行：

```powershell
& $PY BERT_Rejector_Selection_20261030/demo_replay.py --text "how do i make my android phone more secure" --text "order me a book of stamps and envelopes"
```

**实时推理版：**

> Here the model takes raw text, BERT predicts an intent, and the rejector scores the same request three ways. For "how do I make my Android phone more secure", BERT predicts sync device. MSP and entropy stay under their thresholds and would accept it, but its Mahalanobis distance passes the threshold, so it is rejected. For the stamps request it is the other way round.

**回放版（没有模型文件时用）：**

> Our trained weights were not in the shared snapshot, so this replays the model's saved outputs for real test requests. For "how do I make my Android phone more secure", BERT predicted sync device. MSP and entropy accept it, but its Mahalanobis distance, 3,885, passes the threshold of 3,661, so it is rejected. The stamps request shows the reverse.

两句的实际数字在实时推理时会显示在屏幕上，与回放的数字可能有极小差异，讲的时候以屏幕为准。

## 5. 个人贡献（约 65 秒）

**画面：** 依次显示本实验目录、提交记录、`inputs_verification.json` 的 PASS 行、`threshold_stability.png`、`method_choice_heatmap.png` 和 `rejector_candidates.json`。

> My responsibility was rejector design and selection, and the evidence is in our repository under BERT_Rejector_Selection_20261030. First, I took over the inherited rejection pipeline and verified it: 183 checks re-derive every frozen threshold and the five operating points, and all pass. Second, I built the rejection analysis with an AI coding assistant: I specified each step, reviewed and ran the code, and checked every output, from a six-stage runner and 18 unit tests to the demo. Third, I ran the experiments: score complementarity, a 2,000-draw threshold bootstrap, a calibration comparison and a cost-based selection over 24 settings, and I handed the chosen thresholds to the controlled robustness evaluation. Fourth, decisions: I fixed the selection rules before looking at test results, and when the feature files were missing I chose not to retrain, since new models would no longer match the report.（可删：I also drafted the report's rejection sections and Figure 3.）

## 6. 个人分析与反思（约 75 秒）

**画面：** `calibration_vs_separation.png`，然后一张文字幻灯片，分“Limitations”“Responsible use”“Next steps”三栏。

> How do I read these results? Mahalanobis separates out-of-scope requests best, with an AUROC of 0.976, but its F1 advantage over entropy, 0.3 points, is smaller than the noise in the threshold itself: with only 100 out-of-scope validation requests, entropy's test recall ranges from 51 to 74 percent across resamples. So choosing the threshold matters as much as choosing the score. Calibration is not rejection either: temperature scaling cuts calibration error to 0.85 percent, yet at equal acceptance it lets 16 points more out-of-scope requests through MSP. And the best score depends on cost: when executing a wrong intent is the main risk, validation prefers MSP, because it best flags the model's own mistakes. The main limitations are synthetic noise only, three seeds, and a test set we had already explored. For responsible use, a rejected request should go to clarification or a human, and sensitive actions need explicit confirmation. Next, I would set thresholds from explicit error costs and recalibrate them on noisy validation data, because false rejection rises sharply under noise, then confirm on natural typos.

## 视觉素材清单

| 段 | 素材 | 位置 |
|---|---|---|
| 1 | 信息卡；报告标题画面 | 需要新做；共同报告 |
| 2 | 流程图 | 需要新做，一张幻灯片即可 |
| 3 | 准确率柱状图、Fig. 1、Table 8 | 共同报告及 `report_evidence/` |
| 4 | 终端录屏 | 现场录制 |
| 5 | 提交记录、目录树、各输出文件 | GitHub；本实验 `outputs/` |
| 6 | 校准对比图；反思幻灯片 | `outputs/calibration_vs_separation.png`；需要新做 |
