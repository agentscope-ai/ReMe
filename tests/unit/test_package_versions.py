"""Keep the separately published distributions version-compatible."""

from pathlib import Path
import tomllib

import reme
from scripts.package_studio import build_readme


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
    assert packaged_readme.read_text(encoding="utf-8") == build_readme()
