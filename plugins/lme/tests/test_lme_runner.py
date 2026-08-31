"""Fresh-worker checks for running from a checkout without plugin installation."""

# pylint: disable=missing-function-docstring

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("selection", ["lme", "lme.yaml", "custom", "default"])
def test_runner_loads_local_plugin_without_entry_points(tmp_path, selection):
    repo = Path(__file__).resolve().parents[3]
    script = r"""
import importlib.util
from importlib import metadata
import inspect
from pathlib import Path
import sys

import dotenv
import yaml
from reme import Application
from reme.components.component_registry import R
from reme.enumeration import ComponentEnum

repo, selection = Path(sys.argv[1]), sys.argv[2]
# Simulate an environment with no plugin distributions, including in CI where
# plugins are installed for other tests. Avoid reading the developer's .env.
metadata.entry_points = metadata.EntryPoints
dotenv.load_dotenv = lambda *args, **kwargs: None
assert "reme_lme" not in sys.modules
spec = importlib.util.spec_from_file_location("runner", repo / "benchmark/longmemeval/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
# Keep real application config, manifest registration and Jobs; do not create
# model/index clients or start background jobs in this offline construction test.
Application._init_components = lambda self: None
options = dict(
    log_config=False, enable_logo=False, log_to_console=False, log_to_file=False,
    service={"backend": "cli"},
    components={"as_llm": {"bench": {"model": "override-model"}}},
)
if selection == "custom":
    path = Path.cwd() / "custom.yaml"
    path.write_text(yaml.safe_dump({
        "extends": str(repo / "plugins/lme/src/reme_lme/configs/lme.yaml"),
        "app_name": "custom-evaluation",
        "components": {"as_llm": {"bench": {"model": "file-model"}}},
    }))
    selection = str(path)
first = runner.create_reme_app(config=selection, workspace_dir=str(Path.cwd() / "first"), **options)
second = runner.create_reme_app(config="default", workspace_dir=str(Path.cwd() / "second"), **options)
assert "plugin_packages" not in first.config.model_dump()
assert first.config.components["as_llm"]["bench"].model == "override-model"
backend = "lme_agentic_answer_step"
assert R.get(ComponentEnum.STEP, backend) is None
assert second.context.registry.get(ComponentEnum.STEP, backend) is None
assert "dream_cron" in second.config.jobs
if selection == "default":
    assert first.context.registry.get(ComponentEnum.STEP, backend) is None
else:
    cls = first.context.registry.get(ComponentEnum.STEP, backend)
    assert Path(inspect.getfile(cls)).is_relative_to(repo / "plugins/lme/src")
    assert cls().get_prompt("system_prompt")
    assert first.config.plugins == ["lme"]
    assert all(job.backend == "base" for job in first.config.jobs.values())
    assert "auto_dream" not in first.config.jobs
    if selection.endswith("custom.yaml"):
        assert first.config.app_name == "custom-evaluation"
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", script, str(repo), selection],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
