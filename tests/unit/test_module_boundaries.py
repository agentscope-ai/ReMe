"""Enforce existing source dependency boundaries without pinning the directory layout.

These checks inspect explicit imports, including function-local and type-only
imports. They do not claim that importing a submodule avoids the eager package
bootstrap, or inspect dynamic imports and transitive dependencies.
"""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_PACKAGES = ("reme.config", "reme.schema", "reme.enumeration")


def _imports(source: str, package: str):
    """Yield imported names and source lines, resolving relative and package-level imports."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            for alias in node.names:
                yield f"{module}.{alias.name}", node.lineno


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import reme.application as app", "reme.application"),
        ("from reme import application as app", "reme.application"),
        ("from .. import application", "reme.application"),
        ("from ..application import Application", "reme.application.Application"),
        ("def load():\n    from ..steps import BaseStep", "reme.steps.BaseStep"),
        ("if TYPE_CHECKING:\n    from ..application import Application", "reme.application.Application"),
    ],
)
def test_import_inspection_resolves_runtime_layer_references(source, expected):
    """Aliases, relative imports and deferred imports must not hide a forbidden dependency."""
    assert [name for name, _ in _imports(source, "reme.schema")] == [expected]


@pytest.mark.parametrize(
    ("directory", "allowed"),
    [
        ("reme/schema", (*DATA_PACKAGES, "reme.utils.hash_text")),
        ("reme/enumeration", ("reme.enumeration",)),
        (
            "reme/config",
            (*DATA_PACKAGES, "reme.entry_point", "reme.utils.get_logger", "reme.utils.logger_utils"),
        ),
    ],
)
def test_data_and_config_modules_do_not_import_runtime_layers(directory, allowed):
    """Data contracts and config parsing depend only on their low-level capabilities."""
    violations = []
    for path in sorted((ROOT / directory).rglob("*.py")):
        package = ".".join(path.relative_to(ROOT).parts[:-1])
        for name, line in _imports(path.read_text(encoding="utf-8"), package):
            if name != "reme" and not name.startswith("reme."):
                continue
            if not any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed):
                violations.append(f"{path.relative_to(ROOT)}:{line} -> {name}")
    assert not violations, "Unexpected runtime dependencies:\n" + "\n".join(violations)
