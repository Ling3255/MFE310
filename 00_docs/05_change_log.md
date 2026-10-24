# 计划与实施变更记录

## 2026-10-24 修订 2

本次为计划和接口规范修订；功能代码尚未实现，没有需要重跑的实验或新实验结果。

1. FPR@95TPR 统一为 IS 正类：IS 接纳至少 95% 时的 OOS 误接纳率。AUROC 保留 OOS 正类；增加标签/分数转换、ties 和手算验收约定。
2. 将测试 ROC 的 FPR95 与验证阈值冻结后的 `OOS_FA@val95` 分开命名，二者业务方向一致、阈值来源不同。
3. B0 选择 C 改为 IS val macro-F1，与 BERT checkpoint、λ 的目标一致。
4. 声明 val 反复用于选择，选中数字可能偏乐观，只用于描述开发过程，不作为无偏性能估计。
5. 根据对本项目设计过程的澄清，说明领域辅助模型设计未使用 test；小组此前的 benchmark 测试访问另行披露。结果定位为冻结协议下的一次性评估，不整体自动降为探索性，也不声称完全独立确认。
6. 同步主计划、代码职责清单和指标定义。后续 M04/M08、S02、T02 必须按本修订实现。

后续每次变更继续记录日期、原因、受影响文件/运行、是否在查看 test 前、是否需要重跑及替代结果，不覆盖旧记录。

## 2026-10-24 实验框架实现与开发集验证

本次实现代码和开发阶段产物，没有运行正式 BERT 训练或正式 test 评估，不需要替代任何历史 test 结果。

1. 实现 M01–M11、S00–S09、T01–T03、J01–J02，共 27 个物理代码文件；补齐四份 YAML、依赖锁文件、冻结/贡献/AI 文档和证据索引。
2. 使用 `uv` 创建 Python 3.11 项目环境并真实解析、安装依赖。BERT 固定为 `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`，模型许可单独记录；本轮只获取 tokenizer，未使用预训练权重正式训练。
3. `PYTHONPATH=03_src .venv/Scripts/python.exe -m pytest 06_tests -q`：16 passed。覆盖数据、指标、阈值、检测器、配对初始化、λ=0 梯度、样本顺序、断点恢复和真实样本数累积归一化。所有 CLI `--help`、Python compileall 和两个 Shell 的 `bash -n` 通过。
4. 真实 E0–E1 通过：23,700 条、六分区计数、150 意图、10 领域和映射断言。精确重叠复核为 train–val 3 对、train–test 2 对；标签冲突为 4 个唯一文本/4 个样本对。max_len=64 下六分区 BERT 截断数均为 0。原始数据未修改。
5. 真实 E2 仅使用 train/val 完成。选择 C=100，IS val macro-F1=0.9045304077、accuracy=0.905；四个 C 均无收敛警告。保存 pipeline 重载后的 3,000 条 val 概率与归档逐值完全一致。这些数字只用于开发选择，不是 test 结果。
6. 未冻结配置调用 S06 已验证会被拒绝。15 次正式训练、λ 实际选择、真实 Mahalanobis 冻结、导师确认、E6–E8 正式预测/统计/图表和单句冻结演示仍待执行，不能由本轮代码完成推定完成。
7. 开发中修正 NumPy 2.x 标量 `nextafter` 边界拼接，并修正三个测试夹具的手算预期/梯度捕获方式；未改变主计划的指标方向、选择规则或实验协议。
8. 新增 `02_environment/03_environment.yml`，供 Windows CMD 创建 `mfe310` Conda 环境；Conda 只负责 Python 3.11/pip，其余依赖仍使用已验证锁文件。本机没有 Conda，因此该入口未在本机执行，uv 环境的 16 项测试保持通过。

## 2026-10-24 E3–E5 正式训练与冻结

1. 用户报告导师已批准本地 GPU 例外，依据记录在 `07_metadata/local_gpu_approval.md`。正式运行复用现有 `trans` 环境：Python 3.10.20、PyTorch 2.6.0+cu124、Transformers 5.13.0、RTX 4090；补装 pytest 8.4.2 后 16 项测试全部通过，`pip check` 无冲突。
2. 完成 5 个 λ × 3 个种子的全部 15 次正式训练，每份运行保存配置、日志、样本顺序、最佳/最后 checkpoint 和 train/val 特征。GPU 非确定性警告如实记录在 `07_metadata/gpu_determinism.md`，不声称逐位可复现。
3. S04 按预设规则选择 λ=0.1。三种子验证 macro-F1 均值为 CE 0.9657015880、λ0.1 0.9655910100、λ0.3 0.9648791503、λ0.5 0.9637161571、λ1.0 0.9634291159；λ0.1 是正 λ 最高均值及 0.002 容差集合中的最小值，不声称其验证性能优于 CE。
4. S05 为 6 个正式 BERT 模型分别拟合 train-only 类内残差 Ledoit–Wolf Mahalanobis，并用 IS val+OOS val 选择 MSP/Mahalanobis 阈值；B0 只使用 MSP。所有阈值满足 IS acceptance≥0.95。冻结文件为 `01_configs/04_frozen.yaml`，状态 `awaiting_supervisor_approval`，未批准时 S06 已实测拒绝执行。
5. λ 曲线首次生成时发现 `best_val_macro_f1` 字段别名缺失；已在 M10 修正并单独重跑 S04，未重训模型。批量脚本曾因等价相对/绝对数据路径产生不同配置哈希而在训练前停止；后续保持原始相对路径配置完成矩阵，未覆盖旧结果。该路径规范化问题留待后续修正，不影响本次数据和模型内容。
6. 正式 test、bootstrap、领域错误分析和单句冻结演示仍未运行，等待导师对 E5 冻结记录的明确确认。本地 GPU 批准不代替 E5 确认。

## 2026-10-24 导师审查修订（E5 之后、E6 之前）

审查依据：GitHub issue #1 与分支 `supervisor-review`。不需要重训；15 次正式训练、S04 的 λ=0.1 选择均保持有效。

1. `detectors.select_threshold`：OOS 召回平局时改为取 IS 接纳率最高的阈值（原规则取最小 tau，现有冻结阈值平局数 2–38 个，IS 接纳均被压在 0.95 边界）。需重跑 S05。
2. `evaluation.oos_metrics`：增加原计划方向 `fpr95_oos_positive`；`within_domain_error_rate` 改为直接计数，消除浮点残差。只影响 S02 指标末位和尚未运行的 S06。
3. 数据 manifest 增加与换行无关的 `content_sha256`；S06 核验实际数据文件，并拒绝对同一冻结配置第二次 test 评估。需重跑 S01，随后重跑 S05。
4. S05 冻结文件路径改为 POSIX 格式（原为 Windows 反斜杠，非 Windows 平台无法读取）。
5. S08 额外输出按 CE+Dom 合计错误排序的 top-15 表，用于检查只按 CE 排序带来的均值回归偏差；原表保留。
6. S02 支持 `--data-root`；S03 未加 `--allow-download` 时设置 `HF_HUB_OFFLINE=1`，同时阻止权重下载（不改变 config_hash）。
7. B0 的 `pipeline.joblib` 由 scikit-learn 1.9.1 保存，而正式 `trans` 环境为 1.7.2，跨版本反序列化不受支持；须在 `trans` 环境重跑 S02，再重跑 S05。
8. 旧冻结文件改名保留为 `01_configs/04_frozen_r1_superseded.yaml`，不得删除；修订原因即本节。

## 2026-10-24 导师修订重跑完成

1. PR #2 已以合并提交合入 main，保留 r1 冻结。未运行 S03、J01、S04，15 个 BERT 训练目录未改变。
2. S01 已重新生成 manifest，两个原始数据文件均包含 `content_sha256`，23,700 条计数和映射断言保持通过。S02 已在 trans/sklearn 1.7.2 下重训，仍选择 C=100，val macro-F1 约 0.904530，pipeline 重载概率逐值一致。
3. S05 已用现有 checkpoint 重新生成检测器与阈值，平局规则改为最高 IS acceptance 后，各阈值仍满足约束且不再全部压在 0.95；冻结路径均为 `/`。新冻结仍未获批准，S06 门禁实测拒绝执行。
4. E6 前已确定主指标为 `fpr95_is_positive`，补充 `fpr95_oos_positive`，已写入冻结配置、冻结记录、指标文档和仓库主计划。
5. 逐份复核 15 个 checkpoint_selection：实际有 **9 次**最佳 epoch 为最后一个 epoch，并非审查建议中的 8 次。CE 验证均值高于所有正 λ；λ=0.1 只表示正 λ 中最佳及容差规则下的选择。
6. 已提交仓库主计划、`02_environment/04_trans_freeze.txt` 和基于实际沟通记录的本地 GPU 批准说明。trans 环境完整测试为 **19 passed**，`pip check` 无冲突。正式 test、bootstrap、领域分析和演示仍未运行。
