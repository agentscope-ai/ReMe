"""Find-and-replace text in a markdown file (global replacement)."""

from ._file_io import gate_md, read_file_safe, resolve_path, write_file_safe
from ..base_step import BaseStep
from ...components import R


@R.register("edit_step")
class EditStep(BaseStep):
    """Replace every occurrence of ``old`` with ``new`` in the target file."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        old = self.context.get("old")
        new = self.context.get("new")

        if old is None or str(old) == "":
            self._fail("`old` is required and must be non-empty")
            return None
        old_str = str(old)
        new_str = "" if new is None else str(new)

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, err = gate_md(target, raw)
        if err:
            self._fail(err)
            return None

        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return None
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        try:
            content = await read_file_safe(target)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"read failed: {e}", path=str(target))
            return None

        if old_str not in content:
            self._fail(
                f"text to replace was not found in {target}",
                path=str(target),
            )
            return None

        count = content.count(old_str)
        new_content = content.replace(old_str, new_str)

        try:
            await write_file_safe(target, new_content)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", path=str(target))
            return None

        self.context.response.success = True
        self.context.response.answer = f"Replaced {count} occurrence(s) in {target}"
        self.logger.info(f"[{self.name}] edited path={target} count={count}")
        return self.context.response
