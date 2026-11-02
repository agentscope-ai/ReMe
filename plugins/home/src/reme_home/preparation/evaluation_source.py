"""Read-only import of a pinned Home candidate and its fixed runtime."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import yaml

from ..code.repository import resolve_revision, validate_repository, validate_tree
from ..code.snapshot import export_blob
from ..evaluation.artifacts import contained, read_json
from .code import git, tree_fingerprint
from .data.models import fingerprint
from .spec import PreparationSpec, relative_path


@dataclass(frozen=True)
class EvaluationSource:
    """Host-owned source identity; imports never change the source checkout."""

    home: Path
    commit: str
    spec: PreparationSpec
    record: dict

    @classmethod
    def load(cls, workspace: Path, commit: str):
        """Require a prepared workspace and an unambiguous commit SHA or prefix."""
        if re.fullmatch(r"[0-9a-fA-F]{4,64}", commit) is None:
            raise ValueError("code_commit must be a full commit SHA or hexadecimal prefix")
        home = contained(workspace, "home", directory=True)
        record = read_json(contained(home, "metadata/preparation.json"))
        domain = yaml.safe_load(contained(home, "domain_spec.yaml").read_text(encoding="utf-8"))
        if fingerprint(domain) != record["config_fingerprint"]:
            raise ValueError("Source preparation configuration has changed")
        spec = PreparationSpec.model_validate(domain)
        repository, _ = validate_repository(home, record)
        revision = resolve_revision(repository, commit)
        git(repository, "merge-base", "--is-ancestor", record["code"]["initial_commit"], revision)
        validate_tree(repository, record, revision)
        runtime = contained(home, "code/runtime", directory=True)
        if tree_fingerprint(runtime) != record["code"]["runtime_fingerprint"]:
            raise ValueError("Source fixed runtime has changed since preparation")
        return cls(home, revision, spec, record)

    @property
    def provenance(self):
        """Stable identity checked when reopening a standalone test workspace."""
        return {
            "source_workspace": str(self.home.parent),
            "source_preparation_id": self.record["preparation_id"],
            "source_commit_sha": self.commit,
            "runtime_fingerprint": self.record["code"]["runtime_fingerprint"],
        }

    def install(self, runtime: Path, candidate: Path) -> dict:
        """Copy fixed sources and export regular Git blobs, preserving executable bits."""
        source = contained(self.home, "code/runtime", directory=True)
        expected = self.record["code"]["runtime_fingerprint"]
        if tree_fingerprint(source) != expected:
            raise ValueError("Source fixed runtime changed before import")
        shutil.copytree(source, runtime)
        if tree_fingerprint(runtime) != expected or tree_fingerprint(source) != expected:
            raise ValueError("Source fixed runtime changed during import")
        repository = contained(self.home, "code/repo", directory=True)
        candidate.mkdir()
        entries = git(repository, "ls-tree", "-r", "-z", "--full-tree", self.commit, strip=False)
        for entry in filter(None, entries.split("\0")):
            header, name = entry.split("\t", 1)
            mode, kind, oid = header.split()
            path = Path(relative_path(name))
            if mode not in {"100644", "100755"} or kind != "blob" or ".git" in path.parts:
                raise ValueError(f"Candidate must contain regular files only: {name}")
            destination = candidate / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            export_blob(repository, oid, destination)
            destination.chmod(0o755 if mode == "100755" else 0o644)
        # The isolated repository's init is the selected snapshot, not the optimizer baseline.
        git(candidate, "init", "--initial-branch=init", "--template=")
        git(candidate, "add", "--force", "--all")
        git(candidate, "-c", "user.name=Home", "-c", "user.email=home@localhost", "commit", "-m", "Test candidate")
        return {
            **self.record["code"],
            "initial_commit": resolve_revision(candidate, "HEAD"),
            "candidate_fingerprint": tree_fingerprint(candidate),
        }
