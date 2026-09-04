# LongMemEval plugin

[中文说明](./README_ZH.md)

This plugin owns the LongMemEval memory, agentic-answer and judge Steps, their prompts,
and their Job defaults in `plugin.yaml`. ReMe's built-in `benchmark.yaml` owns the
shared evaluation Jobs and components. Dataset handling, the runner and results remain
in [`benchmark/longmemeval`](../../benchmark/longmemeval/README.md).

From the repository root, install ReMe and this plugin in editable mode before running the benchmark:

```bash
python -m pip install -e ".[as]"
reme plugins install ./plugins/lme --editable
reme plugins validate lme
python benchmark/longmemeval/run.py
```

Editable installation registers the `lme` entry point while keeping source changes immediately
visible. The runner selects the built-in `benchmark` preset and explicitly enables `lme` for
each Application. Installing the plugin makes it discoverable but does not enable it globally.

`plugin.yaml` registers backends and contributes the plugin-owned `auto_memory`,
`agentic_answer` and `answer_judge` Job defaults. Start the installed plugin with
`reme start config=benchmark plugins='["lme"]'`. The shared preset does not inherit
`default`: only declared Jobs run, indexing is manual, and neither scheduled dream
nor the optional `auto_dream` Job is enabled.
The existing `auto_memory`, `agentic_answer`, `answer_judge`, `bench` and `judge`
names and model environment variables are unchanged. Explicit CLI field overrides are
applied after the plugin's complete named resources. Installing this plugin does not start an evaluation.

The shared answer base class lives in `reme.steps.benchmark.base_agentic_answer`.
The old core-owned `reme.steps.benchmark.lme` Python import path is removed.
Custom Python callers should import memory, search and answer Steps from `reme_lme`, and the
judge Step from `judge_lme`. After uninstalling,
Applications and CLI services must omit the plugin until it is installed again.
Uninstallation never removes datasets, workspaces or results.
Restart an existing service after changing plugins.
