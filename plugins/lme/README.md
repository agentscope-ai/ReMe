# LongMemEval plugin

[中文说明](./README_ZH.md)

This plugin owns the LongMemEval memory, agentic-answer and judge Steps, their prompts,
and their Job defaults in `plugin.yaml`. ReMe's built-in `benchmark.yaml` owns the
shared evaluation Jobs and components. Dataset handling, the runner and results remain
in [`benchmark/longmemeval`](../../benchmark/longmemeval/README.md).

From the repository root, install ReMe and its runtime dependencies, then run directly without installing the plugin:

```bash
python -m pip install -e ".[as]"
python benchmark/longmemeval/run.py
```

The runner's `create_reme_app()` adds the local plugin `src` to Python's import path
inside each worker, selects the built-in `benchmark` preset, and passes
`plugin_packages={"lme": "reme_lme"}` to `Application`. Only plugins enabled
in the configuration are loaded. It does not invoke pip or mutate the global registry.

`plugin_packages` is a runtime argument; keep it out of saved configurations.
Omitting it or passing `None` preserves the original plugin discovery behavior.

For CLI usage outside the runner, install the
plugin into the same Python environment:

```bash
reme plugins install ./plugins/lme --editable
reme plugins validate lme
```

`plugin.yaml` registers backends and contributes the plugin-owned `auto_memory`,
`agentic_answer` and `answer_judge` Job defaults. Start the installed plugin with
`reme start config=benchmark plugins='["lme"]'`. The shared preset does not inherit
`default`: only declared Jobs run, indexing is manual, and neither scheduled dream
nor the optional `auto_dream` Job is enabled.
The existing `auto_memory`, `agentic_answer`, `answer_judge`, `bench` and `judge`
names and model environment variables are unchanged. Explicit application/CLI overrides
still take precedence. Installing this plugin does not start an evaluation.

The shared answer base class lives in `reme.steps.benchmark.base_agentic_answer`.
The old core-owned `reme.steps.benchmark.lme` Python import path is removed.
Custom Python callers should import plugin Steps from `reme_lme` instead. After uninstalling,
CLI services must omit the plugin; checkout runners can still load the source directly.
Uninstallation never removes datasets, workspaces or results.
Restart an existing service after changing plugins.
