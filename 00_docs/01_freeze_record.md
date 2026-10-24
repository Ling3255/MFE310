# E5 冻结记录

当前状态：**尚未生成冻结配置，等待 15 次正式 BERT 训练、λ 选择和检测器拟合。**

`01_configs/04_frozen.yaml` 当前只是 `frozen: false` 的占位模板，不含模型、检测器或阈值结果。S05 完成真实 E5 后会重写配置和本文件；仅有文件不代表已经冻结或获导师确认。

## 测试历史

领域辅助模型的设计未使用 test 结果。小组其他成员此前查看过该 benchmark 的 test；最终结果只能表述为冻结协议下的一次性评估，不声称完全独立确认。

## 导师确认

当前未确认。S06 和 J02 要求冻结配置同时包含：

- `supervisor_approval.approved: true`
- 非空 `confirmed_by`
- 有效 ISO 格式 `confirmed_at`
- 非空确认依据 `evidence`

确认前不得运行正式 E6。任何查看 test 后的协议变更须另建修订记录，不覆盖旧冻结证据。
