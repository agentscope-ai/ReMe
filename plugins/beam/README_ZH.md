# BEAM 插件

[English](./README.md)

插件包含 BEAM 的记忆、回答、评分 Step、提示词和完整应用配置。
数据集处理、runner 和结果仍留在 [`benchmark/beam`](../../benchmark/beam/README_ZH.md)。

在仓库根目录安装 ReMe 及运行依赖后，可以直接运行评测，无需安装插件包：

```bash
python -m pip install -e ".[as]"
python benchmark/beam/run.py
```

runner 的 `create_reme_app()` 在每个 worker 中将本地插件 `src` 加入 Python 导入路径，
直接读取评测配置，并通过 `Application(plugin_packages={"beam": "reme_beam"}, **config)`
加载配置中启用的插件。它不调用 pip，也不修改全局 registry。

`plugin_packages` 是运行时参数，请勿写入持久化配置。
省略它或传入 `None` 时，保持原有插件发现行为。

若要在 runner 以外使用 `reme start config=beam` 等 CLI 命令，仍需安装插件到同一个 Python 环境：

```bash
reme plugins install ./plugins/beam --editable
reme plugins validate beam
```

`plugin.yaml` 只注册 backend，不提供 `application_defaults`。
`reme start plugins='["beam"]'` 仅为默认应用注册实现，不增加评测 Job，也不改变默认定时任务。

`config=beam` 和兼容名称 `config=beam.yaml` 加载插件内的
[`configs/beam.yaml`](src/reme_beam/configs/beam.yaml)，其中显式启用插件。
评测配置不继承 `default`，只运行声明的 Job：索引手动更新，dream 定时任务和可选的 `auto_dream`
均保持关闭。原有 `auto_memory`、`agentic_answer`、`answer_judge`、`bench`、`judge` 名称及模型环境变量
保持不变，显式应用参数和 CLI 覆盖仍优先。安装或启用插件不会自动开始评测。

共享回答基类位于 `reme.steps.benchmark.base_agentic_answer`。
原 `reme.steps.benchmark.beam` Python 导入路径已移除，自定义 Python 调用应改用 `reme_beam.steps`。
核心不能再包含同名内置配置；仍内置 `beam.yaml` 的旧版核心与本插件不兼容。
CLI 服务卸载插件后需要改用其他应用配置；仓库内的 runner 仍可直接加载源码。
卸载不会删除数据集、工作区或结果。修改插件后需重启已有服务。

无需模型凭据或数据集即可运行 mock 单元测试：

```bash
python -m pytest -c plugins/beam/pyproject.toml plugins/beam/tests -q
```
