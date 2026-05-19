"""Read a markdown file from the vault, with line-range slicing and byte-truncation."""

from ._file_io import (
    gate_md,
    resolve_path,
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

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        start_line = self.context.get("start_line")
        end_line = self.context.get("end_line")

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, err = gate_md(target, raw)
        if err:
            self._fail(err)
            return None

        for label, value in (("start_line", start_line), ("end_line", end_line)):
            if value is None:
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                self._fail(f"{label} must be an integer, got {value!r}")
                return None

        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return None
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        try:
            content = await read_file_safe(target)
        except Exception as e:
            self._fail(f"read failed: {e}", path=str(target))
            return None

        all_lines = content.split("\n")
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

        selected = "\n".join(all_lines[s - 1 : e])
        text = truncate_text_output(
            selected,
            start_line=s,
            total_lines=total,
            file_path=str(target),
        )

        self.context.response.success = True
        self.context.response.answer = text
        self.logger.info(
            f"[{self.name}] read path={target} lines={s}-{e}/{total} bytes={len(text.encode('utf-8'))}",
        )
        return self.context.response
