"""Validate independently published package metadata and release helpers."""

import importlib.util
import json
from pathlib import Path
import tomllib
from types import ModuleType

from packaging.requirements import Requirement
from packaging.version import Version
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    script = REPOSITORY / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bump_version = _load_script("bump_version")
package_studio = _load_script("package_studio")


def test_studio_packages_have_independent_identity() -> None:
    """Keep both Studio distributions independent from the ReMe release version."""
    main_config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    studio_config = tomllib.loads((REPOSITORY / "reme_studio" / "pyproject.toml").read_text(encoding="utf-8"))
    npm_config = json.loads((REPOSITORY / "reme_studio" / "package.json").read_text(encoding="utf-8"))
    auto_fin_config = tomllib.loads(
        (REPOSITORY / "plugins" / "auto-fin" / "pyproject.toml").read_text(encoding="utf-8"),
    )
    daily_paper_config = tomllib.loads(
        (REPOSITORY / "plugins" / "daily_paper" / "pyproject.toml").read_text(encoding="utf-8"),
    )

    assert studio_config["project"]["name"] == "reme_studio"
    assert npm_config["name"] == "@agentscope-ai/reme_studio"
    assert studio_config["project"]["version"] == npm_config["version"]
    assert main_config["project"]["optional-dependencies"]["as"] == ["agentscope[model-ollama]==2.0.7"]
    assert main_config["project"]["optional-dependencies"]["web"] == ["reme_studio"]
    assert main_config["project"]["optional-dependencies"]["core"].count("reme-ai[as]") == 1
    assert main_config["project"]["optional-dependencies"]["core"].count("reme_studio") == 1
    assert "qwenpaw" not in main_config["project"]["optional-dependencies"]
    assert auto_fin_config["project"]["version"] == "0.1.2"
    assert daily_paper_config["project"]["version"] == "0.1.2"
    assert main_config["tool"]["setuptools"]["packages"]["find"]["include"] == ["reme", "reme.*"]
    assert "reme_studio*" in main_config["tool"]["setuptools"]["packages"]["find"]["exclude"]


def _write_version_fixture(repository: Path) -> None:
    (repository / "reme").mkdir()
    (repository / "reme" / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")


def test_bump_version_updates_only_reme(tmp_path: Path) -> None:
    """Do not couple an independent Studio release to the ReMe version helper."""
    _write_version_fixture(tmp_path)
    studio_config = tmp_path / "reme_studio" / "pyproject.toml"
    studio_config.parent.mkdir()
    studio_config.write_text('[project]\nname = "reme_studio"\nversion = "0.1.0"\n', encoding="utf-8")

    previous_version = bump_version.bump_version("1.2.4", tmp_path)

    assert previous_version == "1.2.3"
    assert (tmp_path / "reme" / "__init__.py").read_text(encoding="utf-8") == '__version__ = "1.2.4"\n'
    assert 'version = "0.1.0"' in studio_config.read_text(encoding="utf-8")


@pytest.mark.parametrize("version", ["release", "1..2", "1.2.3_"])
def test_bump_version_rejects_invalid_pep440_versions(tmp_path: Path, version: str) -> None:
    """Reject invalid release versions before changing the source file."""
    _write_version_fixture(tmp_path)
    version_file = tmp_path / "reme" / "__init__.py"
    original_version_text = version_file.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid PEP 440 version"):
        bump_version.bump_version(version, tmp_path)

    assert version_file.read_text(encoding="utf-8") == original_version_text


def test_check_versions_reports_release_tag_mismatch(tmp_path: Path) -> None:
    """Explain which source must change when a release tag does not match."""
    _write_version_fixture(tmp_path)

    with pytest.raises(ValueError, match=r"package version is '1.2.3'; release expects '1.2.4'"):
        bump_version.check_versions(tmp_path, "v1.2.4")


def test_read_version_identifies_invalid_source_file(tmp_path: Path) -> None:
    """Point maintainers to the invalid Python version declaration."""
    _write_version_fixture(tmp_path)
    version_file = tmp_path / "reme" / "__init__.py"
    version_file.write_text('__version__ = "1..2"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{version_file}.*Fix the __version__ declaration"):
        bump_version.check_versions(tmp_path)


def test_bump_version_rolls_back_when_validation_fails(monkeypatch, tmp_path: Path) -> None:
    """Restore the previous version if post-write validation fails."""
    _write_version_fixture(tmp_path)
    version_file = tmp_path / "reme" / "__init__.py"
    original = version_file.read_text(encoding="utf-8")

    def fail_validation(*_args, **_kwargs) -> None:
        raise ValueError

    monkeypatch.setattr(bump_version, "check_versions", fail_validation)

    with pytest.raises(RuntimeError, match="previous version was restored"):
        bump_version.bump_version("1.2.4", tmp_path)

    assert version_file.read_text(encoding="utf-8") == original


def _studio_package_fixture(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    studio_dir = tmp_path / "reme_studio"
    source_dir = studio_dir / "dist-static"
    source_dir.mkdir(parents=True)
    (source_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    static_dir = studio_dir / "src" / "reme_studio" / "static"
    license_file = tmp_path / "LICENSE"
    license_file.write_text("test license\n", encoding="utf-8")
    monkeypatch.setattr(package_studio, "STUDIO_DIR", studio_dir)
    monkeypatch.setattr(package_studio, "STATIC_DIR", static_dir)
    monkeypatch.setattr(package_studio, "LICENSE_FILE", license_file)
    return studio_dir, static_dir


def test_studio_package_preparation_copies_license(monkeypatch, tmp_path: Path) -> None:
    """Include the repository license in the independently distributed Studio package."""
    studio_dir, _ = _studio_package_fixture(monkeypatch, tmp_path)

    package_studio.prepare_package()

    assert (studio_dir / "LICENSE").read_text(encoding="utf-8") == "test license\n"


def test_auto_fin_license_matches_repository() -> None:
    """Keep the independently distributed Auto Fin license complete and current."""
    assert (REPOSITORY / "plugins" / "auto-fin" / "LICENSE").read_text(encoding="utf-8") == (
        REPOSITORY / "LICENSE"
    ).read_text(encoding="utf-8")


def test_auto_fin_requires_reme_base() -> None:
    """Keep the plugin dependency limited to ReMe's public base package."""
    config = tomllib.loads((REPOSITORY / "plugins" / "auto-fin" / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(value) for value in config["project"]["dependencies"]]
    reme_requirements = [requirement for requirement in requirements if requirement.name == "reme-ai"]

    assert len(reme_requirements) == 1
    assert not reme_requirements[0].extras
    assert Version("0.4.1.8") not in reme_requirements[0].specifier
    assert Version("0.4.1.9") in reme_requirements[0].specifier


def test_daily_paper_license_matches_repository() -> None:
    """Keep the independently distributed Daily Paper license complete and current."""
    assert (REPOSITORY / "plugins" / "daily_paper" / "LICENSE").read_text(encoding="utf-8") == (
        REPOSITORY / "LICENSE"
    ).read_text(encoding="utf-8")


def test_daily_paper_declares_runtime_dependencies() -> None:
    """Keep Daily Paper's minimal ReMe and PDF dependencies explicit."""
    config = tomllib.loads((REPOSITORY / "plugins" / "daily_paper" / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(value) for value in config["project"]["dependencies"]]
    by_name = {requirement.name: requirement for requirement in requirements}

    assert not by_name["reme-ai"].extras
    assert Version("0.4.1.8") not in by_name["reme-ai"].specifier
    assert Version("0.4.1.9") in by_name["reme-ai"].specifier
    assert "pypdf" in by_name


def test_studio_package_preparation_preserves_static_gitignore(monkeypatch, tmp_path: Path) -> None:
    """Keep generated static assets ignored after staging the Studio build."""
    _, static_dir = _studio_package_fixture(monkeypatch, tmp_path)

    package_studio.prepare_package()

    assert (static_dir / "index.html").is_file()
    assert (static_dir / ".gitignore").read_text(encoding="utf-8") == package_studio.STATIC_GITIGNORE
