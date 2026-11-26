"""Run-owned generated skills and shared native skill publication."""

from hashlib import sha256
import os
from pathlib import Path
import shutil

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
        self._owned: list[tuple[Path, int, int]] = []
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

    def _own(self, path: Path) -> None:
        stat = path.lstat()
        self._owned.append((path, stat.st_dev, stat.st_ino))

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
                files = WorkspaceFiles(self.generated.parent)
                source = files.checked(self.generated / spec.name)
                self.generated.mkdir(parents=True, exist_ok=True)
                source.mkdir()  # Atomic reservation, including across processes.
                self._own(source)
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
                if name not in self.temporary or any(path == target for path, _, _ in self._owned):
                    continue
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Skill name already exists: {target}")
            target.symlink_to(os.path.relpath(source.resolve(), target.parent), target_is_directory=True)
            if name in self.temporary:
                self._own(target)
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
        for path, device, inode in reversed(self._owned):
            try:
                stat = path.lstat()
                if (stat.st_dev, stat.st_ino) != (device, inode):
                    raise FileExistsError(f"Generated skill was replaced during execution: {path}")
                if path.is_symlink():
                    path.unlink()
                else:
                    shutil.rmtree(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Could not clean generated skills", errors)
