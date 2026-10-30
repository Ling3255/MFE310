# 第一、第二阶段实验概览

本文是第一、第二阶段的独立导航，与 `phase3_scl/README.md` 和 `phase4_dascl/README.md` 对应。代码、配置、运行记录和结果仍保留在仓库根目录的编号文件夹中，不改变已有路径。

## 研究范围

| 阶段 | 研究问题 | 方法与数据 | 主要结论 |
|---|---|---|---|
| 第一阶段 E0–E8 | 领域辅助监督是否改善意图分类和 OOS？ | CLINC150；TF-IDF + LR、BERT-CE、BERT-Dom；3 个种子 | Dom 的 macro-F1 小幅提高约 0.17 pp，跨领域错误减少，但只有 3 个种子，证据有限 |
| 第二阶段 E9–E10 | 小幅提升能否由更多种子和外部数据确认？ | CLINC 10 个种子；MASSIVE en-US 10 个种子；固定 λ=0.1 | 两项验证均为 inconclusive，第一阶段的小幅收益未得到确认 |

E11 是可选探索，未运行，不计入已完成结果。

## 代码与配置

- 可复用模块：`03_src/clinc_domain_aux/`
- 顺序步骤：`04_steps/`
- 批量作业：`05_jobs/`
- 单元测试：`06_tests/`
- 第一阶段配置：`01_configs/01_base.yaml`、`02_baseline.yaml`、`03_search.yaml`、`04_frozen.yaml`
- 第二阶段配置：`01_configs/05_confirm_clinc.yaml`、`06_confirm_massive.yaml`、`07_massive_base.yaml`
- 第一阶段计划：`../CLINC150_domain_auxiliary_implementation_plan.md`
- 第二阶段预注册计划：`00_docs/06_confirmatory_plan.md`

## 运行顺序

以下命令只用于说明正式流程，不代表整理文档时需要重跑。已完成的单次 test 评估不得重复执行。

1. 环境、数据和基线：S00–S02。
2. 第一阶段训练：S03/J01。
3. validation 选择 λ：S04。
4. train 拟合检测器、validation 选择阈值：S05。
5. 获得真实批准后执行一次冻结 test：S06/J02。
6. 从保存预测计算统计和领域分析：S07、S08。
7. 第二阶段数据准备：S10。
8. CLINC 多种子和 MASSIVE 训练：J03。
9. 获得真实批准后，各执行一次 S11，再运行 S12。

详细命令见根目录 `README.md`、第二阶段预注册计划和各作业脚本。Windows 使用当前项目 Python 环境；实际正式运行环境以 `02_environment/04_trans_freeze.txt` 为准。

## 结果位置

| 内容 | 位置 |
|---|---|
| 数据审计、映射、哈希和测试暴露记录 | `07_metadata/` |
| 配置、日志和训练记录 | `08_runs/` |
| 第一阶段 validation 与冻结 test | `09_results/01_validation/`、`09_results/02_test/` |
| 第一阶段配对统计 | `09_results/03_statistics/` |
| 图表、表格和案例 | `09_results/04_figures/`、`05_tables/`、`06_cases/` |
| 第二阶段开发与正式统计 | `09_results/07_confirmatory/` |
| 冻结、批准与变更记录 | `00_docs/01_freeze_record.md`、`05_change_log.md` |

## 解释边界

- 第一阶段的 0.17 pp 不能写成稳定的通用提升。
- 第二阶段未确认效果不等于证明两种方法完全等价。
- OOS 结果方向不一致，只能作描述性报告。
- 单次 test 门禁、批准记录和配置哈希必须保留；统计或绘图重跑不能变成新的 test 推理。
