---
title: Plugin Development
description: Create, register, configure, test, and publish a ReMe plugin.
---

# Plugin Development

A ReMe plugin is a regular Python distribution exposed through the `reme.plugins` entry-point group. Its package-level `plugin.yaml` can register Step and Component backends and provide default Application configuration.

## Minimal structure

```text
my-plugin/
├── pyproject.toml
└── src/my_plugin/
    ├── __init__.py
    ├── plugin.yaml
    └── steps.py
```

`pyproject.toml`:

```toml
[project.entry-points."reme.plugins"]
my-plugin = "my_plugin"
```

`plugin.yaml`:

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

## Implement a Step

```python
from reme.core.registry import R
from reme.steps.base_step import BaseStep


@R.register("my_step")
class MyStep(BaseStep):
    async def execute(self):
        self.context.response.answer = self.context.data["text"]
```

Step instances belong to one Job invocation. Put shared in-memory state under a narrow `app_context.metadata` key. Promote state that needs lifecycle, locking, or persistence to a Component or workspace file.

## Configuration merge

`application_defaults` is a partial `ApplicationConfig`:

```text
plugin defaults < selected/default config < CLI overrides
```

Only effective plugin defaults require their environment variables to be set or to have a fallback. Set `plugins` to a list of strings, such as `plugins: [my_plugin]`.

Plugins must not rewrite user configuration. Their backends enter an Application-local registry only when the plugin appears in that Application's `plugins` list.

## Local validation

```bash
reme plugins validate ./path/to/my-plugin
reme plugins install ./path/to/my-plugin --editable
reme plugins list
reme plugins show my-plugin
```

Start the enabled plugin's service in one terminal and leave it running:

```bash
reme start plugins='["my-plugin"]'
```

From another terminal, call its Job:

```bash
reme my_action text=hello
```

Validation imports plugin code, so run it only for trusted sources.

## Test boundaries

- create workspaces with `tmp_path`;
- mock network, model, and subprocess boundaries;
- verify disabled plugins do not mutate the built-in registry;
- verify plugin defaults and explicit configuration precedence;
- keep tasks, clients, and executors under Component lifecycle;
- never delete or rewrite user source files to repair derived state.

The repository's Daily Paper, Auto Fin, LME, and BEAM plugins are complete examples.

## Entry-point contract

Plugins must use a package-only `reme.plugins` entry point and declare contributions in `plugin.yaml`.
The separate `reme.configs` entry point provides named configuration presets. Enablement belongs to an Application.

See [Plugin Management](./plugin_management.md) for installation, upgrades, and removal.
