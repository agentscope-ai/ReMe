"""Offline installed-wheel smoke test; set REME_TEST_WHEEL_DIR to built release wheels."""

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="Real sandbox smoke test requires macOS")
def test_home_wheel_with_minimum_core(tmp_path):
    """Built home and minimum-core wheels enforce policy outside the source tree."""
    directory = os.environ.get("REME_TEST_WHEEL_DIR")
    if not directory:
        pytest.skip("Set REME_TEST_WHEEL_DIR to a directory containing built core and home wheels")
    wheels = Path(directory).resolve()
    core = list(wheels.glob("reme_ai-0.4.2.2-*.whl"))
    home = list(wheels.glob("reme_home-*.whl"))
    assert len(core) == len(home) == 1, "Provide exactly one minimum-core wheel and one home wheel"
    environment = tmp_path / "venv"
    # Reuse third-party test dependencies, but install both packages into a fresh
    # venv. No network or dependency upgrades are needed for this artifact test.
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    interpreter = str(environment / "bin" / "python")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run(
        [
            interpreter,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(core[0]),
            str(home[0]),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    script = """
import asyncio
import os
import shlex
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from packaging.requirements import Requirement
import reme
import reme_home
from reme.components.application_context import ApplicationContext
from reme_home.auto_bash import AutoBashStep

assert version('reme-ai') == '0.4.2.2'
for module in (reme, reme_home):
    assert Path(module.__file__).is_relative_to(sys.prefix), module.__file__
dist = distribution('reme-home')
requirement = next(Requirement(item) for item in dist.requires if Requirement(item).name == 'reme-ai')
assert requirement.extras == {'as'}
assert '0.4.2.1' not in requirement.specifier
assert '0.4.2.2' in requirement.specifier
next(point for point in dist.entry_points if point.group == 'reme.plugins' and point.name == 'home').load()

async def main():
    workspace = Path('workspace').resolve()
    workspace.mkdir()
    outside = Path('outside.txt').resolve()
    outside.write_text('private')
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(workspace)))
    denied = await step(command='true', is_read_only=False, writable_paths=[str(workspace.parent)])
    assert not denied.success and 'must be configured on the Step' in denied.answer
    denied = await step(command=f'echo changed > {shlex.quote(str(outside))}', is_read_only=False)
    assert not denied.success, denied.answer
    assert outside.read_text() == 'private'
    os.environ['REME_WHEEL_TEST_SECRET'] = 'must-not-inherit'
    command = shlex.join([sys.executable, '-c', "import os; print(os.getenv('REME_WHEEL_TEST_SECRET', 'clean'))"])
    result = await step(command=command, is_read_only=False)
    assert result.success and result.answer.strip() == 'clean', result.answer

asyncio.run(main())
"""
    result = subprocess.run(
        [interpreter, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
