# LongMemEval 插件

[English](./README.md)

插件包含 LongMemEval 的记忆、回答、评分 Step、提示词，以及 `plugin.yaml` 中对应的 Job 默认配置。
ReMe 内置的 `benchmark.yaml` 负责公共评测 Job 和 Component；数据集处理、runner 和结果仍留在
[`benchmark/longmemeval`](../../benchmark/longmemeval/README_ZH.md)。

在仓库根目录以 editable 模式安装 ReMe 和本插件，再运行评测：

```bash
python -m pip install -e ".[as]"
reme plugins install ./plugins/lme --editable
reme plugins validate lme
python benchmark/longmemeval/run.py
```

editable 安装会注册 `lme` entry point，并让源码修改立即生效。runner 选择内置 `benchmark`
配置，并为每个 Application 显式启用 `lme`。安装只让插件可被发现，不会在所有应用中全局启用。

`plugin.yaml` 注册 backend，并通过 `application_defaults` 提供插件拥有的 `auto_memory`、
`agentic_answer` 和 `answer_judge` Job。安装后使用
`reme start config=benchmark plugins='["lme"]'`。公共评测配置不继承 `default`，只运行声明的 Job：
索引手动更新，dream 定时任务和可选的 `auto_dream`
均保持关闭。原有 `auto_memory`、`agentic_answer`、`answer_judge`、`bench`、`judge` 名称及模型环境变量
保持不变，显式应用参数和 CLI 覆盖仍优先。安装或启用插件不会自动开始评测。

共享回答基类位于 `reme.steps.benchmark.base_agentic_answer`。
原 `reme.steps.benchmark.lme` Python 导入路径已移除。自定义 Python 调用应从 `reme_lme`
导入记忆、搜索和回答 Step，并从 `judge_lme` 导入评判 Step。
卸载插件后，Application 和 CLI 服务必须移除插件选择，直到再次安装。
卸载不会删除数据集、工作区或结果。修改插件后需重启已有服务。
