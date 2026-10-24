# CLINC150 领域辅助学习代码项目

本目录包含 TF-IDF + LR、BERT-CE 与 BERT-Dom 的完整实验框架。正式代码只放在本目录；原始数据保留在 `../dataset/`，通过配置或 `DATA_ROOT` 读取。

- [完整实验实施计划](CLINC150_domain_auxiliary_implementation_plan.md)
- [代码结构与职责](00_docs/00_code_structure.md)
- [指标定义](00_docs/03_metric_definitions.md)
- [冻结状态](00_docs/01_freeze_record.md)
- [实施与验证记录](00_docs/05_change_log.md)

## 当前状态

已实现 M01–M11、S00–S09、T01–T03、J01–J02，共 27 个物理代码文件。导师 PR #2 已合并，trans 环境下 19 个单元测试、真实数据审计、E2 基线及模型重载一致性均已通过；具体结果见变更记录和产物。

**15 次正式 BERT 训练、λ 选择和修订后的 E5 检测器冻结已完成。** 选定 λ=0.1；新 `04_frozen.yaml` 记录真实 checkpoint、检测器、修订阈值和 IS 正类主 FPR95，当前状态为 `awaiting_supervisor_approval`。旧冻结以 `*_r1_superseded` 保留。导师冻结确认和正式 test 评估尚未执行，E6 门禁已验证有效。验证集结果仅用于开发选择，不是最终泛化性能证据。

## 环境与复现

Windows/Git Bash 可从本目录重建环境：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r 02_environment/02_requirements-lock.txt
export PYTHONPATH=03_src
export PYTHONUTF8=1
```

也可以使用 Conda。在 **CMD** 中执行（路径按实际位置调整）：

```bat
cd /d G:\MFE310\mfe310code
conda env create -f 02_environment\03_environment.yml
conda activate mfe310
python -m pip install -r 02_environment\02_requirements-lock.txt
set DATA_ROOT=G:\MFE310\dataset\data
set PYTHONPATH=%CD%\03_src
python 04_steps\00_check_environment.py --data-root "%DATA_ROOT%"
python -m pytest 06_tests -q
```

`03_environment.yml` 创建 Python 3.11 和 pip；锁文件安装其余固定版本依赖。每次新开 CMD 后需重新执行 `conda activate mfe310` 和两个 `set` 命令。

Linux 使用 `python` 和 `PYTHONPATH=03_src`，并设置：

```bash
export DATA_ROOT=/path/to/dataset/data
export PYTHON_BIN=python
```

依赖声明在 `02_environment/01_requirements.in`，实际解析锁文件为 `02_requirements-lock.txt`。BERT 使用 `google-bert/bert-base-uncased` 和固定 revision；模型许可单独记录在 `07_metadata/model_resource.json`。

## 执行顺序

```bash
python 04_steps/00_check_environment.py --data-root "$DATA_ROOT"
python 04_steps/01_prepare_and_audit.py --data-root "$DATA_ROOT" --output 07_metadata
python 04_steps/02_train_baseline.py --config 01_configs/02_baseline.yaml
python 04_steps/03_train_bert.py --config 01_configs/01_base.yaml --lambda-domain 0 --seed 0
bash 05_jobs/01_train_all.sh
python 04_steps/04_select_lambda.py --config 01_configs/03_search.yaml
python 04_steps/05_fit_detectors_and_freeze.py --config 01_configs/03_search.yaml
bash 05_jobs/02_evaluate_frozen.sh
python 04_steps/07_bootstrap.py --config 01_configs/04_frozen.yaml
python 04_steps/08_analyze_domains.py --config 01_configs/04_frozen.yaml
python 04_steps/09_demo_inference.py --config 01_configs/04_frozen.yaml --text "what is my account balance"
```

Windows 将 `python` 替换为 `.venv/Scripts/python.exe`。S01–S06 默认要求模型/tokenizer 已缓存，避免意外联网；首次获取训练资源可用 S03 的 `--allow-download`。所有命令从 `mfe310code/` 根目录运行。

J01 串行执行 5 个 λ × 3 个种子，完成标记和数据/配置/初始化哈希匹配时才跳过。S04 只读取验证产物。S05 只拟合 train 检测参数并用 val 选阈值，生成**待导师确认**的冻结文件。

S06/J02 只有在真实导师确认字段完整时才允许 test 评估，并核验配置、checkpoint、检测器和数据哈希。多次统计或绘图必须指定同一 `--test-run`，不重复推理或覆盖旧预测。

## 测试与已验证入口

```bash
PYTHONPATH=03_src .venv/Scripts/python.exe -m pytest 06_tests -q
.venv/Scripts/python.exe -m compileall -q 03_src 04_steps 06_tests
bash -n 05_jobs/01_train_all.sh
bash -n 05_jobs/02_evaluate_frozen.sh
```

测试使用小型临时数据和 tiny-BERT，不运行完整搜索。真实 E0–E1 产物位于 `07_metadata/`、`09_results/05_tables/` 和 `09_results/04_figures/`；E2 运行位于 `08_runs/baseline_tfidf_lr/`。

## 目录

| 目录 | 用途 |
|---|---|
| `00_docs` | 冻结、贡献、AI 声明、指标和证据索引 |
| `01_configs` | 基线、训练、搜索与冻结配置 |
| `02_environment` | 直接依赖与锁定环境 |
| `03_src/clinc_domain_aux` | M01–M11 可复用模块 |
| `04_steps` | S00–S09 顺序命令入口 |
| `05_jobs` | Linux 训练与冻结评估作业 |
| `06_tests` | 数据、指标、配对训练测试 |
| `07_metadata` | 映射、哈希、审计和环境记录 |
| `08_runs` | 配置、日志、checkpoint 和特征 |
| `09_results` | 验证、test、统计、图表、表格和案例 |
| `10_exports` | 最终演示与提交材料 |

新增正式代码须登记到结构清单。不得把 smoke 当正式运行，不得把计划或空冻结模板写成已完成实验。
