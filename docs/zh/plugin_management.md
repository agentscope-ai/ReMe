# 插件管理

ReMe 插件是通过 `reme.plugins` entry-point group 发现的普通 Python distribution。安装插件只表示它在当前 Python
环境中可用，并不会让所有 ReMe Application 自动启用该插件。

需要区分两个操作：

```text
reme plugins install ...        将插件包安装到当前 Python 环境
plugins: [auto-fin]             为一个 Application 启用已安装插件
```

插件包管理仅在本地 CLI 执行，不经过 ReMe HTTP 或 MCP service，也不会自动修改应用配置文件。

典型的插件使用流程分为三个阶段：

1. 安装 ReMe 和插件 distribution。
2. 按照 [ReMe 可选模型配置说明](../../README_ZH.md#可选模型配置)配置插件运行所需的环境变量。
3. 启动 Application 时显式启用插件，例如 `reme start plugins='["auto-fin"]'`。

## 查看已安装插件

```bash
reme plugins list
```

输出包含插件 entry-point 名称、Python distribution、版本和插件契约：

```text
PLUGIN    DISTRIBUTION   VERSION  FORMAT
--------  -------------  -------  --------
auto-fin  reme-auto-fin  0.1.0    manifest
```

`manifest` 表示插件使用当前的 package-level `plugin.yaml` 契约；`legacy` 表示插件使用仍然兼容的 Python descriptor
契约。

manifest 将 backend 注册与应用配置分开：

```yaml
backends:
  example_step: example_plugin.steps:ExampleStep

application_defaults:
  jobs:
    example:
      backend: base
      steps:
        - backend: example_step
```

`application_defaults` 是一段不完整的 `ApplicationConfig`。它与 manifest 的 `backends` 命名空间分开，因为 backend
导入声明属于插件发现协议，并不是应用配置。

本地工具需要结构化结果时可以使用 JSON：

```bash
reme plugins list --json
```

对照某个应用配置查看启用状态：

```bash
reme plugins list --config daily_cookbook
```

可选的 `ENABLED` 列只反映该配置解析出的 `plugins` 列表。其他运行中进程使用的 CLI override 不是全局启用状态。

## 安装插件包

安装已发布的 distribution：

```bash
reme plugins install reme-auto-fin
```

安装指定版本或升级：

```bash
reme plugins install 'reme-auto-fin==0.1.0'
reme plugins install reme-auto-fin --upgrade
```

安装本地插件项目：

```bash
reme plugins install ./plugins/auto-fin
```

开发插件时使用 editable 模式：

```bash
reme plugins install ./plugins/auto-fin --editable
```

ReMe 会通过运行 `reme` 命令的同一个 Python 解释器调用 pip。包解析、下载、依赖变更和构建执行仍由 pip 负责。请只安装
可信的包和本地项目。

安装后确认 ReMe 实际发现的插件名：

```bash
reme plugins list
reme plugins validate auto-fin
```

## 查看插件详情

```bash
reme plugins show auto-fin
```

对于 manifest 插件，结果包含注册的 backend 名称和默认 Job 名称。也可以输出 JSON：

```bash
reme plugins show auto-fin --json
```

`show` 只检查包契约，不构造 ReMe Application。

## 校验插件

校验已安装插件：

```bash
reme plugins validate auto-fin
```

安装前校验本地插件项目：

```bash
reme plugins validate ./plugins/auto-fin
```

校验范围包括 entry point、`plugin.yaml`、backend 导入和组件类型、registry 冲突、`application_defaults` 合并以及最终的
`ApplicationConfig`。校验过程会导入插件 backend 模块，因此只能对可信代码执行。

## 在服务中启用插件

只安装插件不会将插件代码加载到 Application。需要在配置中显式启用：

```yaml
plugins:
  - auto-fin
```

也可以只为本次服务启动追加插件：

```bash
reme start plugins='["auto-fin"]'
```

未传入 `config` 时，ReMe 加载 `default.yaml`。插件的 `application_defaults` 合并在该配置之下，因此显式配置和 CLI
override 优先。这个 mapping 是 `ApplicationConfig` 配置片段，并不是另一套配置 schema。插件 backend 只注册到该
Application 的局部 registry。

默认 HTTP service 启动后，可以通过 ReMe CLI client 或 HTTP 访问插件 Job：

```bash
reme auto_fin topics="黄金,AI,存储芯片"
```

```bash
curl -s http://127.0.0.1:2333/auto_fin \
  -H 'Content-Type: application/json' \
  -d '{"topics":"黄金,AI,存储芯片"}'
```

当应用使用 MCP service 时，允许对外服务的插件 Job 会显示为 MCP tool。

如果需要将插件叠加到其他应用配置，则显式选择该配置：

```bash
reme start config=daily_cookbook plugins='["auto-fin"]'
```

## 卸载插件

这里使用插件 entry-point 名称，它不一定等于 distribution 名称：

```bash
reme plugins uninstall auto-fin
```

需要跳过 pip 确认时：

```bash
reme plugins uninstall auto-fin --yes
```

ReMe 会将 `auto-fin` 解析为提供它的 distribution，例如 `reme-auto-fin`。如果一个 distribution 提供多个插件 entry
point，命令会列出同时被移除的其他插件。

卸载不会重写用户配置。请自行从相关 `plugins` 列表中删除插件，否则下一次启动 Application 时会因为配置的插件未安装而明确
失败。安装、升级或卸载包后，需要重启已经运行的 ReMe 进程。

## 常见问题

### 插件已经安装，但 ReMe 找不到

检查 `reme` 命令与安装插件使用的 pip 是否属于同一个 Python 解释器：

```bash
reme plugins list
python -c 'import sys; print(sys.executable)'
```

使用 `reme plugins install` 可以避免最常见的解释器不一致问题，因为它通过 ReMe 自己的解释器运行 `python -m pip`。

### 插件已经安装，但没有加载

将插件 entry-point 名称加入 Application 的 `plugins` 列表。ReMe 刻意不提供全局 enable/disable 状态。

### 启动时报插件未安装

当前配置仍然启用了缺失插件。请重新安装插件，或者从 `plugins` 中删除对应名称。

### 运行中的服务看不到插件变化

插件发现和 backend 注册发生在 Application 构造阶段。修改已安装包后需要重启服务。
