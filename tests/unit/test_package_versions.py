"""Keep the separately published distributions version-compatible."""

import importlib.util
from pathlib import Path
import tomllib
from types import ModuleType

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

    with pytest.raises(ValueError, match="does not match 1.2.3"):
        bump_version.bump_version("1.2.4", tmp_path)

    assert version_file.read_text(encoding="utf-8") == original_version_text


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
