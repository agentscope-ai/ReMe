"""Read a markdown file from workspace_dir, with line-range slicing and byte-truncation."""

from pathlib import Path, PurePath

from ._file_io import read_file_lines_safe, read_file_safe, truncate_text_output
from ._path import NON_MD_WARNING, gate_md, is_relative_to, resolve_path
from ..base_step import BaseStep
from ...components import R
from ...constants import DEFAULT_MAX_BYTES, MAX_FILE_READ_BYTES
from ...utils import expand_links, render_expansion_lines


@R.register("read_step")
class ReadStep(BaseStep):
    """Read a markdown file. Optional `start_line`/`end_line` for ranged reads.

    Step-level attributes (``kwargs``, configured in yaml under ``steps:`` —
    not exposed to LLM):
        with_neighbors (bool, default False): when true and the file is
            markdown, append a block listing first-order bidirectional
            neighbors (out/in link targets) with name/description meta,
            fetched via the file_store. Same rendering as SearchStep.
        max_neighbors_per_direction (int, default 10): cap per direction.
        white_path_prefix (list[str] | None, default None): whitelist of path
            prefixes; only files whose workspace-relative path starts with one
            of these prefixes pass the white stage. ``None`` disables the
            white stage; an empty list denies every file.
        black_path_prefix (list[str] | None, default None): blacklist of path
            prefixes; files whose workspace-relative path starts with one of
            these prefixes are denied. ``None`` and an empty list both
            disable the black stage.
        The white stage is applied first, then the black stage.
    """

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
            p = Path(s).expanduser()
            if p.is_absolute():
                resolved = p.resolve()
            else:
                resolved = (workspace / p).resolve()
            if not is_relative_to(resolved, workspace):
                self.logger.warning(
                    f"[{self.name}] path prefix outside workspace, dropped: {s}",
                )
                continue
            rel = str(resolved.relative_to(workspace))
            result.append(rel)
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

        Returns True if access is allowed, False otherwise (sets _fail).
        """
        # Fast path: no filtering configured at all.
        if self._white_prefixes is None and not self._black_prefixes:
            return True

        try:
            rel_path = str(target.relative_to(self.workspace_path.resolve()))
        except ValueError:
            self._fail("no permission to access this file", path=str(target))
            return False

        # White stage: None = inactive; [] = deny all; non-empty = allowlist.
        if self._white_prefixes is not None:
            if not any(self._under_prefix(rel_path, prefix) for prefix in self._white_prefixes):
                self._fail("no permission to access this file", path=str(target))
                return False

        # Black stage: None or [] = inactive; non-empty = denylist.
        if self._black_prefixes:
            if any(self._under_prefix(rel_path, prefix) for prefix in self._black_prefixes):
                self._fail("no permission to access this file", path=str(target))
                return False
        return True

    def _fail(self, message: str, **meta) -> None:
        """Mark the response failed and stash a human-readable error."""
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _resolve_target(self, raw: str) -> Path | None:
        """Resolve ``raw`` under workspace and gate the markdown suffix.

        Non-md suffixes only warn (compatibility mode), not fail. Returns
        the absolute path, or ``None`` when ``raw`` is empty/invalid.
        """
        target, err = resolve_path(self.workspace_path, raw)
        if err:
            self._fail(err)
            return None
        target, is_md = gate_md(target)
        if not is_md:
            self.logger.info(f"[{self.name}] {NON_MD_WARNING} path={target}")
        return target

    def _validate_line_args(self, start_line, end_line) -> bool:
        """Accept ``None`` or any value that parses via ``int()`` (JSON/CLI often stringify)."""
        for label, value in (("start_line", start_line), ("end_line", end_line)):
            if value is None:
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                self._fail(f"{label} must be an integer, got {value!r}")
                return False
        return True

    def _check_file(self, target: Path) -> bool:
        """Confirm ``target`` exists and is a regular file."""
        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return False
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return False
        return True

    def _resolve_range(self, total: int, start_line, end_line, target: Path) -> tuple[int, int] | None:
        """Normalize 1-based inclusive ``[s, e]``; reject past-EOF or inverted ranges."""
        s = max(1, int(start_line) if start_line is not None else 1)
        e = min(total, int(end_line) if end_line is not None else total)
        if s > total:
            self._fail(f"start_line {s} exceeds file length ({total} lines)", path=str(target), total_lines=total)
            return None
        if s > e:
            self._fail(f"start_line ({s}) > end_line ({e})", path=str(target))
            return None
        return s, e

    async def _load_content(self, target: Path) -> str | None:
        """Read via the encoding-aware helper; convert exceptions to ``_fail``."""
        try:
            content, _ = await read_file_safe(target)
            return content
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"read failed: {e}", path=str(target))
            return None

    # pylint: disable=too-many-return-statements
    async def execute(self):
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        start_line, end_line = self.context.get("start_line"), self.context.get("end_line")
        with_neighbors: bool = bool(self.kwargs.get("with_neighbors", False))
        max_neighbors_per_direction: int = int(self.kwargs.get("max_neighbors_per_direction", 10))

        # Validate inputs and target before touching the filesystem twice.
        target = self._resolve_target(raw)
        if target is None:
            return None
        if not self._check_path_permission(target):
            return None
        if not self._validate_line_args(start_line, end_line):
            return None
        if not self._check_file(target):
            return None

        if target.stat().st_size <= MAX_FILE_READ_BYTES:
            content = await self._load_content(target)
            if content is None:
                return None

            all_lines = content.split("\n")
            total = len(all_lines)
            bounds = self._resolve_range(total, start_line, end_line, target)
            if bounds is None:
                return None
            s, e = bounds
            excerpt = "\n".join(all_lines[s - 1 : e])
        else:
            s = max(1, int(start_line) if start_line is not None else 1)
            requested_end = int(end_line) if end_line is not None else None
            if requested_end is not None and s > requested_end:
                self._fail(f"start_line ({s}) > end_line ({requested_end})", path=str(target))
                return None
            try:
                excerpt, total, _encoding = await read_file_lines_safe(
                    target,
                    s,
                    requested_end,
                    max_collect_bytes=DEFAULT_MAX_BYTES * 2,
                )
            except Exception as exc:  # pylint: disable=broad-except
                self._fail(f"read failed: {exc}", path=str(target))
                return None
            if s > total:
                self._fail(f"start_line {s} exceeds file length ({total} lines)", path=str(target), total_lines=total)
                return None
            e = min(total, requested_end if requested_end is not None else total)

        text = truncate_text_output(excerpt, start_line=s, total_lines=total, file_path=str(target))

        self.context.response.success = True
        self.context.response.answer = text
        self.logger.info(f"[{self.name}] read path={target} lines={s}-{e}/{total} bytes={len(text.encode('utf-8'))}")

        if with_neighbors and target.suffix.lower() == ".md":
            await self._maybe_inject_neighbors(target, text, max_neighbors_per_direction)

        return self.context.response

    # -- neighbor injection (opt-in) -----------------------------------------

    async def _maybe_inject_neighbors(self, target: Path, text: str, max_per_direction: int) -> None:
        """Append the rendered neighbor block + stash raw expansion in metadata."""
        assert self.context is not None
        try:
            rel_path = str(target.relative_to(self.workspace_path))
        except ValueError:
            self.logger.info(f"[{self.name}] skip neighbors: path outside workspace_path path={target}")
            return

        try:
            expansion = await expand_links(self.file_store, [rel_path], max_per_direction)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"[{self.name}] neighbor fetch failed: {type(exc).__name__}: {exc}")
            return

        per_path = expansion.get(rel_path, {})
        lines = render_expansion_lines(per_path)
        if not lines:
            return

        out_n = len(per_path.get("outlinks") or [])
        in_n = len(per_path.get("inlinks") or [])
        header = f"========== Related neighbors (outlinks={out_n}, inlinks={in_n}) =========="
        block = "\n".join([header, *lines])
        self.context.response.answer = f"{text}\n\n{block}"
        self.context.response.metadata["link_expansion"] = expansion
