# Benchmark 状态

ReMe v4 正在向文件化长期记忆和个人知识库方向演进。当前仓库文档和代码中尚未发布 LongMemEval、LoCoMo、BFCL 或 AppWorld 的正式对比结果表。

## LongMemEval

目前没有可引用的 LongMemEval 测试结果数据。如果需要复现或贡献 LongMemEval 评测，请先在 issue 中对齐评测设置，至少说明：

- 数据集版本、切分方式和过滤规则。
- ReMe 版本、commit、配置文件和 workspace 初始化方式。
- LLM、embedding 模型、温度、并发数等实验参数。
- 评测脚本、指标定义和可复现命令。
- 是否使用 `auto_memory`、`auto_dream`、检索 rerank 或其他额外处理。

在没有这些信息前，不建议把单次本地跑分写成官方对比结果。

社区中可能会出现 LongMemEval 评测脚本或 harness 的尝试性 PR；在相关 PR 完成 CLA、评审和维护者确认前，
这些内容只能作为参考，不能视为官方结果或稳定复现入口。

## 历史 Benchmark

旧 issue 中提到的 BFCL、AppWorld、ReMeLight 或 `reme2` 流程多来自早期实现。当前 v4/main 代码已经切换到新的 workspace、Job、Step 和文件化记忆流程；复现历史实验前，应先确认对应 benchmark 脚本、配置和依赖仍然适用于当前版本。

## 贡献建议

如果要补充 benchmark 结果，建议以独立 PR 提交：

1. 在 issue 中说明计划评测的数据集和目标表格。
2. 提交可运行脚本或清晰的复现命令。
3. 将结果表和实验配置放在同一文档中。
4. 说明外部服务依赖和无法在 CI 中完整复现的部分。
