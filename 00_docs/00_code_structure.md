# 代码结构与编写清单

版本：2026-10-24。项目根目录：`G:/MFE310/mfe310code/`。

本清单说明代码数量、位置、职责和执行顺序。M01–M11、S00–S09、T01–T03、J01–J02 已实现；代码完成不代表正式实验完成。实际验证和未执行阶段见 `05_change_log.md` 与根目录 README。

## 1. 数量与编号

| 类别 | 功能编号 | 数量 |
|---|---|---:|
| 可复用 Python 模块 | M01–M11 | 11 |
| 顺序执行 Python 入口 | S00–S09 | 10 |
| Python 测试 | T01–T03 | 3 |
| Linux Shell 作业 | J01–J02 | 2 |
| 功能代码合计 | 上述四类 | 26 |
| Python 包初始化文件 | `__init__.py`，非独立功能 | 1 |
| 实现完成后的物理代码文件总数 | 24 个功能 Python + 1 个初始化 Python + 2 个 Shell | 27 |

配置、说明文档、自动生成的 JSON/CSV、checkpoint 和日志不计入代码数量。此数量是当前范围的预算；以后确需增加文件时，先更新本清单并说明职责。

## 2. 可复用模块 M01–M11

这些模块位于 `03_src/clinc_domain_aux/`，文件名不以数字开头。每个文件在英文模块 docstring 中写功能编号，例如 `M05: BERT intent and domain auxiliary heads.`。

| 编号 | 文件 | 职责 | 输入 → 输出 |
|---|---|---|---|
| M01 | `common.py` | YAML 配置、相对路径、随机种子、哈希、环境、统一日志 | 配置/种子 → 解析配置、运行记录、哈希 |
| M02 | `data.py` | 六分区读取、样本 ID、意图/领域映射、Dataset 与 loader | 原始 JSON/映射 → 数据集、batch、元数据 |
| M03 | `audit.py` | 完整性、重复/冲突、句长、领域词汇重叠 | 数据集 → 审计表、热力图、检查结果 |
| M04 | `baseline.py` | TF-IDF + LR、按 IS val macro-F1 选 C、baseline 保存/加载 | train/val → 最佳词法模型和验证预测 |
| M05 | `domain_aux.py` | BERT 共享表征、意图头、领域头、两项损失 | token batch、标签、λ → logits、CLS、loss |
| M06 | `training.py` | 配对初始化、优化器、训练、早停、checkpoint、特征导出 | 模型/配置/种子 → checkpoint、日志和 train/val 特征 |
| M07 | `detectors.py` | MSP、类内残差协方差、Mahalanobis、阈值 | train 特征和 val 分数 → 检测参数、阈值 |
| M08 | `evaluation.py` | 批量推理、领域错误率、F1、AUROC_OOS、IS 正类 FPR95、冻结阈值 OOS_FA@val95 | 模型、数据、检测器 → 预测表、指标 JSON |
| M09 | `statistics.py` | 种子汇总、差值、分层配对 bootstrap、重叠行敏感性 | 保存预测 → 差值表、置信区间 |
| M10 | `domain_analysis.py` | λ 曲线、混淆矩阵、领域指标、错误案例、证据索引 | 验证/测试结果 → 图表、案例、证据映射 |
| M11 | `inference.py` | 复用 tokenizer、模型、检测器完成单句演示 | 一句话 + 冻结配置 → 意图、领域、分数、接纳/拒绝 |

模块只实现各自职责，CLI 参数处理集中在 `04_steps/`。M10 可调用 M03 的数据图表工具，避免重复绘图代码。M11 复用正式推理逻辑，不另训练展示模型。

另需 `03_src/clinc_domain_aux/__init__.py` 标识包，不放训练副作用。

## 3. 编号执行入口 S00–S09

| 编号 | 文件（均在 `04_steps/`） | 对应实验 | 执行内容与产物 |
|---|---|---|---|
| S00 | `00_check_environment.py` | 运行准备 | 检查依赖、GPU、写入权限和数据位置；导出环境记录，不自动安装包 |
| S01 | `01_prepare_and_audit.py` | E0–E1 | 调 M02/M03，保存映射、数据哈希、计数、审计、图表 |
| S02 | `02_train_baseline.py` | E2 | 调 M04/M08，统一按 IS val macro-F1 搜索 C，保存 baseline 和 val 结果 |
| S03 | `03_train_bert.py` | E3–E4 | 调 M05/M06，接收 λ/seed，训练一次 CE 或 Dom |
| S04 | `04_select_lambda.py` | E4 | 汇总所有验证运行，执行 0.2 个百分点规则，保存选中 λ 与曲线；不读取 test |
| S05 | `05_fit_detectors_and_freeze.py` | E5 | 拟合检测器、选验证阈值、生成待导师确认的冻结配置；不自动声称已获确认 |
| S06 | `06_evaluate_test.py` | E6 | 检查冻结及导师确认记录，评估选定模型，保存一次正式预测 |
| S07 | `07_bootstrap.py` | E6 统计 | 从保存预测生成均值、标准差、配对区间、敏感性结果 |
| S08 | `08_analyze_domains.py` | E7–E8 | 生成领域混淆、错误构成、案例和可选 OOS 分析 |
| S09 | `09_demo_inference.py` | 展示 | 调 M11 运行单句推理，保存展示输出 |

编号是执行依赖顺序，不是要求每次全部重跑。S07/S08 可以对冻结预测重新计算；S06 不因重画图而重复推理。

## 4. 作业脚本和测试

| 编号 | 文件 | 检查或运行内容 |
|---|---|---|
| J01 | `05_jobs/01_train_all.sh` | Linux 上枚举 λ={0,0.1,0.3,0.5,1.0} × seed={0,1,2}；调用 S03；检查完成标志和配置哈希；保存作业日志 |
| J02 | `05_jobs/02_evaluate_frozen.sh` | 核对冻结配置/确认记录；调用 S06；不搜索参数或覆盖旧正式预测 |
| T01 | `06_tests/test_01_data.py` | 标签映射、数据断言、稳定 ID、OOS 不进入 IS 训练 loader |
| T02 | `06_tests/test_02_metrics.py` | 用手算样本检查 F1、跨/同领域错误、AUROC/FPR95 正类转换、冻结阈值与测试 ROC 工作点区别、ties 及配对统计 |
| T03 | `06_tests/test_03_pairing.py` | CE/Dom 同种子初始化、训练样本序列、λ=0 的梯度路径及恢复 checkpoint |

测试应小而有针对性，不运行完整 BERT 搜索。重型训练属于实验，不伪装成单元测试。

## 5. 配置和文档

| 位置 | 文件 | 内容与建立时机 |
|---|---|---|
| 根目录 | `README.md` | 项目入口、结构导航、实施状态，已建立 |
| `00_docs` | `00_code_structure.md` | 本清单，已建立 |
| `00_docs` | `01_freeze_record.md` | E5 的配置、模型、阈值、测试历史与导师确认记录 |
| `00_docs` | `02_contributions.md` | 每阶段记录共享与个人代码、分析、提交证据 |
| `00_docs` | `03_metric_definitions.md` | 已建立：固定分母、AUROC 与 FPR95 各自正类、分数方向、单位、选择规则与验算例子 |
| `00_docs` | `04_ai_use_declaration.md` | 按实际使用记录工具、用途、人工核验 |
| `00_docs` | `05_change_log.md` | 已建立：记录本轮协议修订；实施时继续记录配置变更、bug 修复与重跑原因 |
| `01_configs` | `01_base.yaml` | 数据路径、模型 revision、训练公共配置 |
| `01_configs` | `02_baseline.yaml` | TF-IDF、C 网格与选择标准 |
| `01_configs` | `03_search.yaml` | λ 网格、种子、选择规则与作业路径 |
| `01_configs` | `04_frozen.yaml` | S05 生成后审阅冻结，供最终评估读取 |
| `02_environment` | `01_requirements.in` | 直接依赖及版本约束 |
| `02_environment` | `02_requirements-lock.txt` | 经真实环境验证后锁定的依赖版本 |

实施计划主文件位于上一级：[完整实验计划](../../CLINC150_domain_auxiliary_implementation_plan.md)。只有这份作为实施计划主版本，避免另建内容冲突的副本。

## 6. 元信息和产物归档

- `07_metadata/`：E0 映射、数据版本、审计摘要、测试暴露记录；主要由 S01 生成。
- `08_runs/ce_seed0/`：seed 0 的 CE 配置、初始化哈希、训练日志、checkpoint、验证预测与特征。其他运行按 λ/seed 独立建目录。
- `09_results/01_validation/`：C/λ 搜索、验证指标、检测参数与阈值。
- `09_results/02_test/`：冻结测试预测、逐模型指标；每次正式运行用固定 run_id 建子目录，避免覆盖。
- `09_results/03_statistics/`：种子汇总、Dom−CE 差值、bootstrap 分布/区间和敏感性结果。
- `09_results/04_figures/`：数据分析、λ、领域混淆、错误构成与可选可视化。
- `09_results/05_tables/`：可直接导入报告的 CSV/Markdown 表。
- `09_results/06_cases/`：修复/新增错误案例与抽样说明。
- `10_exports/`：最终报告图表、演示材料和提交快照；ZIP 不把自己再次包含进去。

图表命名采用 `fig_01_domain_overlap.png`、`fig_02_lambda_sensitivity.png` 等；表格采用 `table_01_main_results.csv`；运行 ID、配置哈希和生成脚本写入证据映射。编号含义一旦写进报告就保持稳定。

## 7. 分批编写和验收顺序

| 批次 | 编写范围 | 当前实际状态 |
|---|---|---|
| P1 数据与基础设施 | M01–M03、S00–S01、T01、环境/基础配置 | 代码完成；真实计数、映射、环境和审计已运行 |
| P2 指标与词法基线 | M04、M08 的指标部分、S02、T02 | 代码完成；手算测试、E2 和重载一致性已运行 |
| P3 深度模型与训练 | M05–M06、S03、T03、J01 | 代码和 tiny-BERT 测试完成；15 次正式训练未运行 |
| P4 选择和冻结 | M07、S04–S05、搜索/冻结文档 | 代码和检测器单元测试完成；等待正式训练产物 |
| P5 最终预测与统计 | M08 预测部分、M09、S06–S07、J02 | 代码和统计单元测试完成；冻结确认前不运行 test |
| P6 分析与展示 | M10–M11、S08–S09、贡献/AI 文档 | 代码完成；等待冻结预测生成正式图表和案例 |

各代码文件完成后在提交记录和本清单中更新实际状态，不提前填“完成”。今后新增正式代码一律归入上述分类；额外工具脚本也必须编号并登记。

验证结果仅用于开发选择，不作为无偏性能估计。本项目领域模型的设计未使用 test 结果；小组其他成员已有测试访问应另行披露，最终结果按冻结协议下的一次性评估表述。上述约定写入 S05 冻结记录，并与主计划、指标说明保持一致。
