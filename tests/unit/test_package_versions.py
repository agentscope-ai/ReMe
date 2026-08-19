"""Keep the separately published distributions version-compatible."""

import importlib.util
from pathlib import Path
import tomllib
from types import ModuleType

from packaging.requirements import Requirement
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


def test_studio_package_and_extra_versions_match_reme() -> None:
    """Require one version bump to update both wheels and their exact pin."""
    main_config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    studio_config = tomllib.loads(
        (REPOSITORY / "packages" / "reme_ai_studio" / "pyproject.toml").read_text(encoding="utf-8"),
    )
    version = bump_version.read_version(REPOSITORY)
    expected_dependency = f"reme-ai-studio=={version}"

    assert studio_config["project"]["version"] == version
    assert main_config["project"]["optional-dependencies"]["web"] == [expected_dependency]
    assert expected_dependency in main_config["project"]["optional-dependencies"]["core"]


def _write_version_fixture(repository: Path, *, studio_version: str = "1.2.3") -> None:
    (repository / "reme").mkdir()
    (repository / "packages" / "reme_ai_studio").mkdir(parents=True)
    (repository / "reme" / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        """[project]
name = "reme-ai"

[project.optional-dependencies]
web = ["reme-ai-studio==1.2.3"]
core = ["example", "reme-ai-studio==1.2.3"]
""",
        encoding="utf-8",
    )
    (repository / "packages" / "reme_ai_studio" / "pyproject.toml").write_text(
        f'[project]\nname = "reme-ai-studio"\nversion = "{studio_version}"\n',
        encoding="utf-8",
    )


def test_bump_version_updates_both_packages_and_exact_pins(tmp_path: Path) -> None:
    """Update the two package versions and both dependency declarations together."""
    _write_version_fixture(tmp_path)

    previous_version = bump_version.bump_version("1.2.4", tmp_path)

    assert previous_version == "1.2.3"
    assert (tmp_path / "reme" / "__init__.py").read_text(encoding="utf-8") == '__version__ = "1.2.4"\n'
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8").count("reme-ai-studio==1.2.4") == 2
    studio_config = tomllib.loads(
        (tmp_path / "packages" / "reme_ai_studio" / "pyproject.toml").read_text(encoding="utf-8"),
    )
    assert studio_config["project"]["version"] == "1.2.4"


def test_bump_version_rejects_inconsistent_sources_before_writing(
    tmp_path: Path,
) -> None:
    """Refuse to update any file if the current release metadata has drifted."""
    _write_version_fixture(tmp_path, studio_version="1.2.2")
    version_file = tmp_path / "reme" / "__init__.py"
    original_version_text = version_file.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match=r"project.version is '1.2.2'; expected '1.2.3'"):
        bump_version.bump_version("1.2.4", tmp_path)

    assert version_file.read_text(encoding="utf-8") == original_version_text


@pytest.mark.parametrize("version", ["release", "1..2", "1.2.3_"])
def test_bump_version_rejects_invalid_pep440_versions(tmp_path: Path, version: str) -> None:
    """Reject invalid release versions before changing any source file."""
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


def test_check_versions_identifies_invalid_toml_file(tmp_path: Path) -> None:
    """Point maintainers to malformed package metadata."""
    _write_version_fixture(tmp_path)
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text("[project\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{config_file}.*cannot read valid TOML.*Fix this file"):
        bump_version.check_versions(tmp_path)


def test_check_versions_identifies_missing_table(tmp_path: Path) -> None:
    """Explain which required TOML table must be restored."""
    _write_version_fixture(tmp_path)
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text('[project]\nname = "reme-ai"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{config_file}.*\[project.optional-dependencies\].*Add or fix"):
        bump_version.check_versions(tmp_path)


def test_bump_version_rolls_back_when_a_write_fails(monkeypatch, tmp_path: Path) -> None:
    """Restore earlier files when a later atomic replacement fails."""
    # pylint: disable=protected-access
    _write_version_fixture(tmp_path)
    paths = bump_version._paths(tmp_path)
    original = {path: path.read_text(encoding="utf-8") for path in paths}
    write_atomic = bump_version._write_atomic
    attempts = 0

    def fail_second_write(path: Path, content: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("simulated write failure")
        write_atomic(path, content)

    monkeypatch.setattr(bump_version, "_write_atomic", fail_second_write)

    with pytest.raises(RuntimeError, match="all changed files were restored"):
        bump_version.bump_version("1.2.4", tmp_path)

    assert {path: path.read_text(encoding="utf-8") for path in paths} == original


def test_studio_readme_is_generated_from_website_docs() -> None:
    """Keep the packaged PyPI description synchronized with the source docs."""
    packaged_readme = REPOSITORY / "packages" / "reme_ai_studio" / "README.md"
    assert packaged_readme.read_text(encoding="utf-8") == package_studio.build_readme()


def test_studio_package_preparation_copies_license(monkeypatch, tmp_path: Path) -> None:
    """Include the repository license in the independently distributed Studio package."""
    package_dir = tmp_path / "reme_ai_studio"
    package_dir.mkdir()
    monkeypatch.setattr(package_studio, "PACKAGE_DIR", package_dir)

    package_studio.prepare_package(copy_static=False)

    assert (package_dir / "LICENSE").read_text(encoding="utf-8") == (REPOSITORY / "LICENSE").read_text(
        encoding="utf-8",
    )


def test_auto_fin_license_matches_repository() -> None:
    """Keep the independently distributed Auto Fin license complete and current."""
    assert (REPOSITORY / "plugin" / "auto-fin" / "LICENSE").read_text(encoding="utf-8") == (
        REPOSITORY / "LICENSE"
    ).read_text(encoding="utf-8")


def test_auto_fin_requires_reme_core() -> None:
    """Install the optional runtime packages needed while loading Auto Fin's entry points."""
    config = tomllib.loads((REPOSITORY / "plugin" / "auto-fin" / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(value) for value in config["project"]["dependencies"]]
    reme_requirements = [requirement for requirement in requirements if requirement.name == "reme-ai"]

    assert len(reme_requirements) == 1
    assert set(reme_requirements[0].extras) == {"core"}


def test_studio_package_preparation_preserves_static_gitignore(monkeypatch, tmp_path: Path) -> None:
    """Keep generated static assets ignored after staging the Studio build."""
    package_dir = tmp_path / "reme_ai_studio"
    package_dir.mkdir()
    website_dir = tmp_path / "website"
    source_dir = website_dir / "dist-static"
    source_dir.mkdir(parents=True)
    (source_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    static_dir = package_dir / "src" / "reme_ai_studio" / "static"
    static_dir.mkdir(parents=True)
    (static_dir / ".gitignore").write_text(package_studio.STATIC_GITIGNORE, encoding="utf-8")

    monkeypatch.setattr(package_studio, "PACKAGE_DIR", package_dir)
    monkeypatch.setattr(package_studio, "WEBSITE_DIR", website_dir)
    monkeypatch.setattr(package_studio, "STATIC_DIR", static_dir)
    monkeypatch.setattr(package_studio, "build_readme", lambda: "# ReMe Studio\n")

    package_studio.prepare_package()

    assert (static_dir / "index.html").is_file()
    assert (static_dir / ".gitignore").read_text(encoding="utf-8") == package_studio.STATIC_GITIGNORE
