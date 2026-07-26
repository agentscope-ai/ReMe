"""Mixin providing request-scoped path permission checks for file I/O steps.

``PrefixCheck`` is designed as a **mixin** that must be combined with a class
that inherits ``BaseStep`` (or otherwise provides ``self.context``,
``self.workspace_path``, ``self.logger``, and ``self.name``).
It cannot be instantiated on its own.

Usage::

    class ReadStep(PrefixCheck, BaseStep):
        ...

The allowed scope is injected per invocation by the server (e.g.
``AutoMemoryStep`` through agent-wrapper ``injected_job_kwargs``) under the
reserved ``_allowed_paths`` runtime-context key. The underscore-prefixed key
is intentionally absent from every public job parameter schema, so the model
never sees it as a tool argument and cannot override it. No permission state
is retained on the Step between calls.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from ._path import is_relative_to, resolve_path

if TYPE_CHECKING:
    from loguru import Logger

    from ...components.runtime_context import RuntimeContext

ALLOWED_PATHS_KEY = "_allowed_paths"


class PrefixCheck:
    """Request-scoped path permission mixin for Steps.

    Requires the host class to provide:
        - ``self.context: RuntimeContext | None``
        - ``self.workspace_path: Path``  (property)
        - ``self.logger: Logger``
        - ``self.name: str``

    ``_allowed_paths`` semantics (read from the RuntimeContext per call):
        * absent / ``None`` — no restriction; every workspace path passes.
        * existing files — permit that exact resolved file only.
        * existing directories — permit their resolved descendants, using
          path-component containment rather than string-prefix matching.
        * nonexistent entries are logged and ignored; an empty effective
          scope denies access. Entries that fail workspace-safe resolution
          deny access (invalid permission configuration fails closed).
    """

    # Type hints for attributes provided by the BaseStep host class.
    context: "RuntimeContext | None"
    workspace_path: Path
    logger: "Logger"
    name: str

    def _check_path_permission(self, target: Path) -> bool:
        """Check ``target`` against the request-scoped ``_allowed_paths`` constraint.

        Returns True if access is allowed, False otherwise. The caller is
        responsible for reporting the failure (e.g. via ``_fail``).
        """
        raw_allowed = self.context.get(ALLOWED_PATHS_KEY) if self.context is not None else None
        if raw_allowed is None:
            return True
        if isinstance(raw_allowed, (str, Path)):
            raw_allowed = [raw_allowed]
        if not isinstance(raw_allowed, (list, tuple)) or not raw_allowed:
            self.logger.warning(f"[{self.name}] invalid {ALLOWED_PATHS_KEY} constraint; denying access (fail closed)")
            return False

        allowed_files: list[Path] = []
        allowed_dirs: list[Path] = []
        for raw in raw_allowed:
            resolved, err = resolve_path(self.workspace_path, str(raw))
            if err or resolved is None:
                self.logger.warning(
                    f"[{self.name}] invalid {ALLOWED_PATHS_KEY} entry {str(raw)!r} ({err}); "
                    "denying access (fail closed)",
                )
                return False
            if resolved.is_file():
                allowed_files.append(resolved)
            elif resolved.is_dir():
                allowed_dirs.append(resolved)
            else:
                self.logger.warning(
                    f"[{self.name}] nonexistent or unsupported {ALLOWED_PATHS_KEY} entry {str(raw)!r}; skipping",
                )

        resolved_target = target.resolve()
        return resolved_target in allowed_files or any(
            is_relative_to(resolved_target, directory) for directory in allowed_dirs
        )
