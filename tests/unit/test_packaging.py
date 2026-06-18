"""Tests for package metadata that affects runtime assets."""

import tomllib
from pathlib import Path


def test_reme_packages_tokenizer_stopwords():
    """The default tokenizer stopwords file must be included in built packages."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]

    assert "stopwords" in package_data["reme.components.tokenizer"]
