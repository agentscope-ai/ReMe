# BEAM 插件

[English](./README.md)

插件包含 BEAM 的记忆、回答、评分 Step、提示词，以及 `plugin.yaml` 中对应的 Job 默认配置。
ReMe 内置的 `benchmark.yaml` 负责公共评测 Job 和 Component；数据集处理、runner 和结果仍留在
[`benchmark/beam`](../../benchmark/beam/README_ZH.md)。

在仓库根目录安装 ReMe 及运行依赖后，可以直接运行评测，无需安装插件包：

```bash
python -m pip install -e ".[as]"
python benchmark/beam/run.py
```

runner 的 `create_reme_app()` 在每个 worker 中将本地插件 `src` 加入 Python 导入路径，
选择内置 `benchmark` 配置，并通过 `Application(plugin_packages={"beam": "reme_beam"}, **config)`
加载配置中启用的插件。它不调用 pip，也不修改全局 registry。

`plugin_packages` 是运行时参数，请勿写入持久化配置。
省略它或传入 `None` 时，保持原有插件发现行为。

若要在 runner 以外通过 CLI 使用，仍需安装插件到同一个 Python 环境：

```bash
reme plugins install ./plugins/beam --editable
reme plugins validate beam
```

`plugin.yaml` 注册 backend，并通过 `application_defaults` 提供插件拥有的 `auto_memory`、
`agentic_answer` 和 `answer_judge` Job。安装后使用
`reme start config=benchmark plugins='["beam"]'`。公共评测配置不继承 `default`，只运行声明的 Job：
索引手动更新，dream 定时任务和可选的 `auto_dream`
均保持关闭。原有 `auto_memory`、`agentic_answer`、`answer_judge`、`bench`、`judge` 名称及模型环境变量
保持不变，显式应用参数和 CLI 覆盖仍优先。安装或启用插件不会自动开始评测。

共享回答基类位于 `reme.steps.benchmark.base_agentic_answer`。
原 `reme.steps.benchmark.beam` Python 导入路径已移除，自定义 Python 调用应从 `reme_beam` 导入插件 Step。
CLI 服务卸载插件后必须移除插件选择；仓库内的 runner 仍可直接加载源码。
卸载不会删除数据集、工作区或结果。修改插件后需重启已有服务。
