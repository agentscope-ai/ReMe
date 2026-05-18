"""Read a markdown file from the vault, with line-range slicing and byte-truncation."""

from pathlib import Path

from ._file_io import (
    DEFAULT_MAX_BYTES,
    read_file_safe,
    truncate_text_output,
)
from ..base_step import BaseStep
from ...components import R


@R.register("read_step")
class ReadStep(BaseStep):
    """Read a markdown file. Optional `start_line`/`end_line` for ranged reads."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _resolve_target_or_fail(self) -> Path | None:
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        target, err = self.resolve_path(raw, require_md=True)
        if err:
            self._fail(err)
            return None
        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return None
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None
        return target

    def _validate_line_params_or_fail(self) -> bool:
        assert self.context is not None
        for label in ("start_line", "end_line"):
            value = self.context.get(label)
            if value is None:
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                self._fail(f"{label} must be an integer, got {value!r}")
                return False
        return True

    async def _load_content_or_fail(self, target: Path) -> str | None:
        try:
            return await read_file_safe(target)
        except Exception as ex:
            self._fail(f"read failed: {ex}", path=str(target))
            return None

    def _compute_range_or_fail(
        self,
        target: Path,
        all_lines: list[str],
    ) -> tuple[int, int, int] | None:
        assert self.context is not None
        start_line = self.context.get("start_line")
        end_line = self.context.get("end_line")
        total = len(all_lines)
        s = max(1, int(start_line) if start_line is not None else 1)
        e = min(total, int(end_line) if end_line is not None else total)
        if s > total:
            self._fail(
                f"start_line {s} exceeds file length ({total} lines)",
                path=str(target),
                total_lines=total,
            )
            return None
        if s > e:
            self._fail(f"start_line ({s}) > end_line ({e})", path=str(target))
            return None
        return s, e, total

    def _emit_response(
        self,
        target: Path,
        all_lines: list[str],
        s: int,
        e: int,
        total: int,
    ) -> None:
        assert self.context is not None
        max_bytes = int(self.context.get("max_bytes") or DEFAULT_MAX_BYTES)
        selected = "\n".join(all_lines[s - 1 : e])
        text = truncate_text_output(
            selected,
            start_line=s,
            total_lines=total,
            max_bytes=max_bytes,
            file_path=str(target),
        )
        self.context.response.success = True
        self.context.response.answer = text
        self.logger.info(
            f"[{self.name}] read path={target} lines={s}-{e}/{total} bytes={len(text.encode('utf-8'))}",
        )

    async def execute(self):
        assert self.context is not None
        target = self._resolve_target_or_fail()
        if target is None:
            return None
        if not self._validate_line_params_or_fail():
            return None
        content = await self._load_content_or_fail(target)
        if content is None:
            return None
        all_lines = content.split("\n")
        rng = self._compute_range_or_fail(target, all_lines)
        if rng is None:
            return None
        s, e, total = rng
        self._emit_response(target, all_lines, s, e, total)
        return self.context.response
