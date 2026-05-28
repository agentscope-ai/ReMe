"""Read a markdown file from vault_dir, with line-range slicing and byte-truncation."""

from pathlib import Path

from ._file_io import NON_MD_WARNING, gate_md, read_file_safe, resolve_path, truncate_text_output
from ..base_step import BaseStep
from ...components import R


@R.register("read_step")
class ReadStep(BaseStep):
    """Read a markdown file. Optional `start_line`/`end_line` for ranged reads."""

    def _fail(self, message: str, **meta) -> None:
        """Mark the response failed and stash a human-readable error."""
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta: self.context.response.metadata.update(meta)

    def _resolve_target(self, raw: str) -> Path | None:
        """Resolve ``raw`` under vault and gate the markdown suffix.

        Non-md suffixes only warn (compatibility mode), not fail. Returns
        the absolute path, or ``None`` when ``raw`` is empty/invalid.
        """
        target, err = resolve_path(self.vault_path, raw)
        if err: self._fail(err); return None
        target, is_md = gate_md(target)
        if not is_md: self.logger.info(f"[{self.name}] {NON_MD_WARNING} path={target}")
        return target

    def _validate_line_args(self, start_line, end_line) -> bool:
        """Accept ``None`` or any value that parses via ``int()`` (JSON/CLI often stringify)."""
        for label, value in (("start_line", start_line), ("end_line", end_line)):
            if value is None: continue
            try:
                int(value)
            except (TypeError, ValueError):
                self._fail(f"{label} must be an integer, got {value!r}");
                return False
        return True

    def _check_file(self, target: Path) -> bool:
        """Confirm ``target`` exists and is a regular file."""
        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target));
            return False
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target));
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
            self._fail(f"start_line ({s}) > end_line ({e})", path=str(target));
            return None
        return s, e

    async def _load_content(self, target: Path) -> str | None:
        """Read via the encoding-aware helper; convert exceptions to ``_fail``."""
        try:
            content, _ = await read_file_safe(target);
            return content
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"read failed: {e}", path=str(target));
            return None

    async def execute(self):
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        start_line, end_line = self.context.get("start_line"), self.context.get("end_line")

        # Validate inputs and target before touching the filesystem twice.
        target = self._resolve_target(raw)
        if target is None: return None
        if not self._validate_line_args(start_line, end_line): return None
        if not self._check_file(target): return None

        content = await self._load_content(target)
        if content is None: return None

        all_lines = content.split("\n")
        total = len(all_lines)
        bounds = self._resolve_range(total, start_line, end_line, target)
        if bounds is None: return None
        s, e = bounds

        text = truncate_text_output("\n".join(all_lines[s - 1: e]), start_line=s, total_lines=total,
                                    file_path=str(target))

        self.context.response.success = True
        self.context.response.answer = text
        self.logger.info(f"[{self.name}] read path={target} lines={s}-{e}/{total} bytes={len(text.encode('utf-8'))}")
        return self.context.response
