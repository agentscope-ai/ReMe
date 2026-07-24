"""Mixin providing white/black path-prefix permission checks for file I/O steps.

``PrefixCheck`` is designed as a **mixin** that must be combined with a class
that inherits ``BaseStep`` (or otherwise provides ``self.kwargs``,
``self.workspace_path``, ``self.logger``, and ``self.name``).
It cannot be instantiated on its own.

Usage::

    class ReadStep(PrefixCheck, BaseStep):
        ...

Configuration (via ``kwargs`` in yaml ``steps:`` block):
    white_path_prefix (list[str] | None): whitelist of workspace-relative path
        prefixes. ``None`` disables the white stage; ``[]`` denies every file.
    black_path_prefix (list[str] | None): blacklist of workspace-relative path
        prefixes. ``None`` or ``[]`` disables the black stage.
"""

from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from ._path import resolve_path

if TYPE_CHECKING:
    from loguru import Logger


class PrefixCheck:
    """Path-prefix permission mixin for Steps.

    Requires the host class to provide:
        - ``self.kwargs: dict``
        - ``self.workspace_path: Path``  (property)
        - ``self.logger: Logger``
        - ``self.name: str``
    """

    # Type hints for attributes provided by the BaseStep host class.
    kwargs: dict
    workspace_path: Path
    logger: "Logger"
    name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ``None`` (absent / yaml null) means the filter stage is inactive;
        # an explicit empty list is preserved so a white-list of [] denies all.
        self._white_prefixes: list[str] | None = self._normalize_prefixes(
            self.kwargs.get("white_path_prefix"),
        )
        self._black_prefixes: list[str] | None = self._normalize_prefixes(
            self.kwargs.get("black_path_prefix"),
        )

    def _normalize_prefixes(self, raw: list[str] | None) -> list[str] | None:
        """Convert prefix paths to workspace-relative; drop those that fail.

        Returns ``None`` when ``raw`` is ``None`` (stage inactive). Returns
        ``[]`` when ``raw`` is empty (white-list stage denies every file;
        black-list stage is a no-op, same as ``None``).
        """
        if raw is None:
            return None
        workspace = self.workspace_path.resolve()
        result: list[str] = []
        for item in raw:
            s = str(item).strip()
            if not s:
                continue
            # Expand ~ before resolution; resolve_path handles absolute/relative,
            # workspace containment, and filename-component validation.
            target, err = resolve_path(self.workspace_path, str(Path(s).expanduser()))
            if err:
                self.logger.warning(
                    f"[{self.name}] path prefix invalid, dropped: {s} ({err})",
                )
                continue
            result.append(str(target.relative_to(workspace)))
        return result

    @staticmethod
    def _under_prefix(rel_path: str, prefix: str) -> bool:
        """Path-aware prefix test: True when ``rel_path`` equals ``prefix`` or is nested under it.

        Unlike ``str.startswith``, this respects path-component boundaries so
        a prefix of ``"daily"`` matches ``daily/notes.md`` but not
        ``daily-report/notes.md``.
        """
        return PurePath(rel_path).is_relative_to(PurePath(prefix))

    def _check_path_permission(self, target: Path) -> bool:
        """Check target against the white (allow) then black (deny) prefix lists.

        White stage:
            * ``None``  — stage inactive, file passes this stage.
            * ``[]``    — denies every file.
            * non-empty — only paths starting with one of the prefixes pass.
        Black stage (applied after the white stage):
            * ``None`` or ``[]`` — stage inactive.
            * non-empty          — paths starting with one of the prefixes
              are denied.

        Returns True if access is allowed, False otherwise. The caller is
        responsible for reporting the failure (e.g. via ``_fail``).
        """
        # Fast path: no filtering configured at all.
        if self._white_prefixes is None and not self._black_prefixes:
            return True

        try:
            rel_path = str(target.relative_to(self.workspace_path.resolve()))
        except ValueError:
            return False

        # White stage: None = inactive; [] = deny all; non-empty = allowlist.
        if self._white_prefixes is not None:
            if not any(self._under_prefix(rel_path, prefix) for prefix in self._white_prefixes):
                return False

        # Black stage: None or [] = inactive; non-empty = denylist.
        if self._black_prefixes:
            if any(self._under_prefix(rel_path, prefix) for prefix in self._black_prefixes):
                return False
        return True
