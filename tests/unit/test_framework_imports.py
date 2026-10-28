"""Compatibility of public imports after grouping application and config modules."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("canonical_module", "names", "compatibility_modules"),
    [
        ("reme.app.application", ("Application",), ("reme", "reme.application")),
        (
            "reme.app.context",
            ("ApplicationContext",),
            ("reme.components", "reme.components.application_context"),
        ),
        (
            "reme.app.runtime_context",
            ("RuntimeContext",),
            ("reme.components", "reme.components.runtime_context"),
        ),
        (
            "reme.config.models",
            ("ApplicationConfig", "ComponentConfig", "JobConfig"),
            ("reme.schema", "reme.schema.application_config"),
        ),
    ],
)
def test_compatibility_imports_share_the_canonical_classes(canonical_module, names, compatibility_modules):
    """Old imports must preserve class identity for schemas, dependencies and isinstance checks."""
    canonical = import_module(canonical_module)
    for module_name in compatibility_modules:
        compatibility = import_module(module_name)
        for name in names:
            assert getattr(compatibility, name) is getattr(canonical, name), f"{module_name}.{name}"
