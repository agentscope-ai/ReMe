"""Sync ReMe's AgentScope pin with QwenPaw's direct dependency."""

import argparse
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

_PIN = re.compile(r'"agentscope\[model-ollama\]==[^"\n]+"')


def _exact_agentscope_version(dependencies: list[str]) -> str:
    requirements = [Requirement(value) for value in dependencies]
    matches = [requirement for requirement in requirements if requirement.name.lower() == "agentscope"]
    if len(matches) != 1:
        raise ValueError(f"Expected one AgentScope dependency, found {len(matches)}")
    requirement = matches[0]
    specifiers = list(requirement.specifier)
    if (
        requirement.extras != {"model-ollama"}
        or requirement.marker is not None
        or len(specifiers) != 1
        or specifiers[0].operator != "=="
    ):
        raise ValueError(
            f"Expected an exact agentscope[model-ollama] pin, found {requirement}",
        )
    version = Version(specifiers[0].version)
    if version.is_prerelease or version.is_devrelease:
        raise ValueError(f"Expected a stable AgentScope version, found {version}")
    return str(version)


def sync_pin(qwenpaw_text: str, reme_text: str) -> tuple[str, str]:
    """Return the QwenPaw version and updated ReMe manifest, failing on ambiguous pins."""
    qwenpaw = tomllib.loads(qwenpaw_text)
    reme = tomllib.loads(reme_text)
    version = _exact_agentscope_version(qwenpaw["project"]["dependencies"])
    _exact_agentscope_version(reme["project"]["optional-dependencies"]["as"])
    matches = list(_PIN.finditer(reme_text))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one replaceable ReMe AgentScope pin, found {len(matches)}",
        )
    replacement = f'"agentscope[model-ollama]=={version}"'
    return (
        version,
        reme_text[: matches[0].start()] + replacement + reme_text[matches[0].end() :],
    )


def main() -> None:
    """Synchronize the local manifest from a downloaded QwenPaw manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qwenpaw_manifest", type=Path)
    parser.add_argument("reme_manifest", type=Path)
    args = parser.parse_args()
    original = args.reme_manifest.read_text(encoding="utf-8")
    version, updated = sync_pin(
        args.qwenpaw_manifest.read_text(encoding="utf-8"),
        original,
    )
    if updated != original:
        args.reme_manifest.write_text(updated, encoding="utf-8")
        print(f"Updated ReMe AgentScope pin to QwenPaw's {version}")
    else:
        print(f"ReMe already matches QwenPaw's AgentScope {version}")


if __name__ == "__main__":
    main()
