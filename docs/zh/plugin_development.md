---
title: 插件开发
description: 创建、注册、配置、测试和发布 ReMe 插件。
---

# 插件开发

ReMe 插件是一个普通 Python distribution，通过 `reme.plugins` entry-point group 暴露 package-level `plugin.yaml`。插件可以注册新的 Step、Component backend，并提供默认 Application 配置。

## 最小结构

```text
my-plugin/
├── pyproject.toml
└── src/my_plugin/
    ├── __init__.py
    ├── plugin.yaml
    └── steps.py
```

`pyproject.toml`：

```toml
[project.entry-points."reme.plugins"]
my-plugin = "my_plugin"
```

`plugin.yaml`：

```yaml
name: my-plugin
backends:
  my_step: my_plugin.steps:MyStep
application_defaults:
  jobs:
    my_action:
      backend: base
      description: Run my plugin action
      parameters:
        type: object
        properties:
          text: { type: string }
        required: [text]
      steps:
        - backend: my_step
```

## 实现 Step

```python
from reme.components.component_registry import R
from reme.steps.base_step import BaseStep


@R.register("my_step")
class MyStep(BaseStep):
    async def execute(self):
        self.context.response.answer = self.context.data["text"]
```

Step 实例属于单次 Job 调用。跨调用的内存状态应放在带命名空间的 `app_context.metadata`；需要生命周期、锁或持久化时，应升级为 Component 或 workspace 文件。

## 配置合并

`application_defaults` 是不完整的 `ApplicationConfig`。合并顺序为：

```text
插件默认值 < 选中/default 配置 < CLI 覆盖
```

插件不应自动修改用户的配置文件。只有在 Application 的 `plugins` 列表中显式启用后，插件 backend 才会加入该 Application 的局部 registry。

## 本地验证

```bash
reme plugins validate ./path/to/my-plugin
reme plugins install ./path/to/my-plugin --editable
reme plugins list
reme plugins show my-plugin
reme start plugins='["my-plugin"]'
reme my_action text=hello
```

校验会导入插件代码，因此只应对可信源码执行。

## 测试边界

- 使用 `tmp_path` 创建 workspace；
- Mock 网络、模型和子进程；
- 验证插件未启用时不会污染 built-in registry；
- 验证默认配置与显式配置的合并优先级；
- 验证后台任务、客户端和 executor 都由 Component 生命周期关闭；
- 验证失败不会删除或重写用户源文件。

仓库中的 `plugins/daily_paper`、`plugins/auto-fin`、`plugins/lme` 和 `plugins/beam` 是完整参考实现。

## 兼容性

迁移期间仍兼容旧的 Python Plugin descriptor 和 `reme.configs` entry point，但新插件应使用 `plugin.yaml`。不要依赖进程级全局 enable/disable 状态；插件启用始终属于具体 Application 配置。

包管理命令、升级与卸载行为见[插件管理](./plugin_management.md)。
