"""Keep the separately published distributions version-compatible."""

from pathlib import Path
import tomllib

import reme
from scripts import package_studio


def test_studio_package_and_extra_versions_match_reme() -> None:
    """Require one version bump to update both wheels and their exact pin."""
    repository = Path(__file__).resolve().parents[2]
    main_config = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    studio_config = tomllib.loads(
        (repository / "packages" / "reme_ai_studio" / "pyproject.toml").read_text(encoding="utf-8"),
    )
    expected_dependency = f"reme-ai-studio=={reme.__version__}"

    assert studio_config["project"]["version"] == reme.__version__
    assert main_config["project"]["optional-dependencies"]["web"] == [expected_dependency]
    assert expected_dependency in main_config["project"]["optional-dependencies"]["core"]


def test_studio_readme_is_generated_from_website_docs() -> None:
    """Keep the packaged PyPI description synchronized with the source docs."""
    repository = Path(__file__).resolve().parents[2]
    packaged_readme = repository / "packages" / "reme_ai_studio" / "README.md"
    assert packaged_readme.read_text(encoding="utf-8") == package_studio.build_readme()


def test_studio_package_preparation_copies_license(monkeypatch, tmp_path: Path) -> None:
    """Include the repository license in the independently distributed Studio package."""
    repository = Path(__file__).resolve().parents[2]
    package_dir = tmp_path / "reme_ai_studio"
    package_dir.mkdir()
    monkeypatch.setattr(package_studio, "PACKAGE_DIR", package_dir)

    package_studio.prepare_package(copy_static=False)

    assert (package_dir / "LICENSE").read_text(encoding="utf-8") == (repository / "LICENSE").read_text(
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
