"""Run-owned generated skills and shared native skill publication."""

from hashlib import sha256
import os
from pathlib import Path
import re
from stat import S_ISREG
from uuid import uuid4

from pydantic import TypeAdapter
import yaml

from ...schema.agent_skill import SkillSpec
from ...utils.files.workspace_files import WorkspaceFiles

BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
_SKILLS = TypeAdapter(list[SkillSpec])


def skill_specs(value) -> list[SkillSpec]:
    """Validate the single list-based skill input contract."""
    return _SKILLS.validate_python(value)


class SkillScope:
    """Own generated files and their links until all calls in one run finish."""

    def __init__(self, specs: list[SkillSpec], project: Path, generated: Path):
        self.specs = specs
        self.project = project
        self.generated = generated
        self.sources: dict[str, Path] = {}
        self.temporary: set[str] = set()
        self._generated: dict[Path, str] = {}
        self._links: dict[Path, Path] = {}
        self._locks: list[int] = []
        self._published: set[Path] = set()
        self.active = True

    @property
    def names(self) -> list[str]:
        """Names passed to backends that select skills by name."""
        return list(self.sources)

    @property
    def directories(self) -> list[str]:
        """Directories passed directly to AgentScope's local loaders."""
        return [str(source) for source in self.sources.values()]

    def _lock(self, name: str) -> None:
        """Reserve a generated name until this process exits or the scope closes."""
        files = WorkspaceFiles(self.generated.parent)
        self.generated.mkdir(parents=True, exist_ok=True)
        path = files.checked(self.generated / f".{name}.lock")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Skill lock must be a regular file")
            if os.name == "nt":
                import msvcrt

                if not os.fstat(descriptor).st_size:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            os.close(descriptor)
            if isinstance(exc, OSError):
                raise FileExistsError(f"Generated skill is already in use: {name}") from exc
            raise
        self._locks.append(descriptor)

    def _check_generated(self, source: Path, token: str) -> bool:
        """Preserve replacements and extra files instead of recursively deleting them."""
        if not source.exists() and not source.is_symlink():
            return False
        files = WorkspaceFiles(self.generated.parent)
        files.checked(source)
        if (
            not source.is_dir()
            or {path.name for path in source.iterdir()} - {"SKILL.md", ".reme-generated"}
            or files.read(source / ".reme-generated") != token
        ):
            raise FileExistsError(f"Generated skill was replaced during execution: {source}")
        manifest = source / "SKILL.md"
        if manifest.exists() or manifest.is_symlink():
            files.checked(manifest)
            if not S_ISREG(manifest.stat().st_mode):
                raise FileExistsError(f"Generated skill was replaced during execution: {source}")
        return True

    def _remove_generated(self, source: Path, token: str) -> None:
        if self._check_generated(source, token):
            (source / "SKILL.md").unlink(missing_ok=True)
            (source / ".reme-generated").unlink()
            source.rmdir()

    def prepare(self) -> None:
        """Resolve references and exclusively create generated skill directories."""
        if len({spec.name for spec in self.specs}) != len(self.specs):
            raise ValueError("Duplicate skill name")
        for spec in self.specs:
            source = self.project / spec.name
            builtin = BUILTIN_SKILLS_ROOT / spec.name
            if spec.content is None:
                if not (source / "SKILL.md").is_file() and (builtin / "SKILL.md").is_file():
                    source = builtin
                if not (source / "SKILL.md").is_file():
                    raise FileNotFoundError(f"Skill is missing SKILL.md: {source}")
            else:
                for existing in (source, builtin):
                    if existing.exists() or existing.is_symlink():
                        raise FileExistsError(f"Skill name already exists: {existing}")
                self._lock(spec.name)
                files = WorkspaceFiles(self.generated.parent)
                token = uuid4().hex
                source = files.checked(self.generated / f"{spec.name}-{token}")
                source.mkdir()
                (source / ".reme-generated").write_text(token, encoding="utf-8")
                self._generated[source] = token
                header = yaml.safe_dump(
                    {"name": spec.name, "description": spec.description},
                    allow_unicode=True,
                    sort_keys=False,
                ).strip()
                files.write(source / "SKILL.md", f"---\n{header}\n---\n\n{spec.content}\n")
                self.temporary.add(spec.name)
            self.sources[spec.name] = source

    def publish(self, root: Path) -> bool:
        """Expose skills in one backend's effective discovery directory."""
        if not self.active:
            raise RuntimeError("Skill scope is closed")
        if not self.sources:
            return False
        root = root.absolute()
        if root in self._published:
            return False
        for name, source in self.sources.items():
            target = root / name
            if name not in self.temporary and target.is_dir() and target.resolve() == source.resolve():
                continue
            if root.resolve() != root:
                raise FileExistsError(f"Skill discovery directory contains a symlink: {root}")
            root.mkdir(parents=True, exist_ok=True)
            if source.parent == BUILTIN_SKILLS_ROOT:
                self._publish_builtin(root, source)
                continue
            if target.is_symlink() and target.resolve() == source.resolve():
                if name not in self.temporary or target in self._links:
                    continue
            if name in self.temporary and target.is_symlink():
                previous = target.resolve()
                token = previous.name.removeprefix(name + "-")
                if (
                    previous.parent == self.generated
                    and previous.name == f"{name}-{token}"
                    and re.fullmatch(r"[0-9a-f]{32}", token)
                ):
                    # The name lock proves the previous scope is no longer running.
                    self._check_generated(previous, token)
                    target.unlink()
                    self._remove_generated(previous, token)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Skill name already exists: {target}")
            target.symlink_to(os.path.relpath(source.resolve(), target.parent), target_is_directory=True)
            if name in self.temporary:
                self._links[target] = source
        self._published.add(root)
        return True

    @staticmethod
    def _publish_builtin(root: Path, source: Path) -> None:
        """Refresh only regular built-in copies still owned by ReMe."""
        files = WorkspaceFiles(root)
        target = root / source.name
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise FileExistsError(f"Skill name already exists: {target}")
        manifest, receipt = target / "SKILL.md", target / ".reme-skill"
        if manifest.is_symlink() or receipt.is_symlink():
            raise FileExistsError(f"Skill name already exists: {target}")
        content = (source / "SKILL.md").read_text(encoding="utf-8")
        digest = sha256(content.encode("utf-8")).hexdigest()
        if manifest.exists():
            if not manifest.is_file() or not receipt.is_file():
                return
            current = files.read(manifest)
            if current == content:
                return
            previous = sha256(current.encode("utf-8")).hexdigest()
            if previous not in files.read(receipt).splitlines():
                return
            files.write(receipt, f"{previous}\n{digest}\n")
        else:
            files.write(receipt, f"{digest}\n")
        files.write(manifest, content)

    def close(self) -> None:
        """Remove only resources created by this scope, in reverse order."""
        self.active = False
        errors = []
        for path, source in reversed(self._links.items()):
            try:
                if not path.exists() and not path.is_symlink():
                    continue
                if not path.is_symlink() or path.resolve() != source:
                    raise FileExistsError(f"Generated skill was replaced during execution: {path}")
                path.unlink()
            except OSError as exc:
                errors.append(exc)
        for source, token in reversed(self._generated.items()):
            try:
                self._remove_generated(source, token)
            except (OSError, ValueError) as exc:
                errors.append(exc)
        for descriptor in reversed(self._locks):
            os.close(descriptor)
        self._links.clear()
        self._generated.clear()
        self._locks.clear()
        if errors:
            raise ExceptionGroup("Could not clean generated skills", errors)
