# BEAM plugin

[中文说明](./README_ZH.md)

This plugin owns the BEAM memory, agentic-answer and judge Steps, their prompts,
and the complete evaluation application preset. Dataset handling, the runner and
results remain in [`benchmark/beam`](../../benchmark/beam/README.md).

From the repository root, install ReMe and its runtime dependencies, then run directly without installing the plugin:

```bash
python -m pip install -e ".[as]"
python benchmark/beam/run.py
```

The runner's `create_reme_app()` adds the local plugin `src` to Python's import path
inside each worker, reads the evaluation preset directly, and passes
`plugin_packages={"beam": "reme_beam"}` to `Application`. Only plugins enabled
in the configuration are loaded. It does not invoke pip or mutate the global registry.

For CLI usage outside the runner, such as `reme start config=beam`, install the
plugin into the same Python environment:

```bash
reme plugins install ./plugins/beam --editable
reme plugins validate beam
```

`plugin.yaml` registers backends only; it does not contribute application defaults.
`reme start plugins='["beam"]'` registers those backends in the ordinary default
application without adding evaluation Jobs or changing its scheduled tasks.

`config=beam` and the compatible `config=beam.yaml` name select the packaged
[`configs/beam.yaml`](src/reme_beam/configs/beam.yaml) preset, which explicitly
enables the plugin. It does not inherit `default`: only declared Jobs run, indexing
is manual, and neither scheduled dream nor the optional `auto_dream` Job is enabled.
The existing `auto_memory`, `agentic_answer`, `answer_judge`, `bench` and `judge`
names and model environment variables are unchanged. Explicit application/CLI overrides
still take precedence. Installing this plugin does not start an evaluation.

The old core-owned `reme.steps.benchmark.beam` Python import path is removed.
Custom Python callers should import from `reme_beam.steps` instead. The core must
no longer ship `beam.yaml`; older cores with a same-name built-in config are incompatible.
After uninstalling, CLI services need a different configuration; checkout runners can still
load the source directly. Uninstallation never removes datasets, workspaces or results.
Restart an existing service after changing plugins.

Run isolated, mocked tests (no model credentials or dataset required):

```bash
python -m pytest -c plugins/beam/pyproject.toml plugins/beam/tests -q
```
