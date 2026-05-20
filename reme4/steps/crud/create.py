"""Create a new markdown file with optional YAML front matter (any fields)."""

import frontmatter
import yaml

from ._file_io import gate_md, resolve_path, write_file_safe
from ..base_step import BaseStep
from ...components import R

_RESERVED_KEYS = {"path", "content", "metadata"}


def _coerce_frontmatter_value(raw):  # pylint: disable=too-many-return-statements
    """Coerce a context value into a YAML-friendly frontmatter value.

    Returns ``(value, error)``. ``value is None`` means "skip this key"
    (empty / missing). Strings that look like YAML literals (start with
    ``[`` or ``{``) are parsed so LLM tool calls can pass collections as
    JSON/YAML strings, e.g. ``tags='["foo","bar"]'``. Other strings are
    kept verbatim to avoid surprising coercions (``"yes"`` → ``True`` etc.).
    """
    if raw is None:
        return None, None
    if isinstance(raw, (list, dict, bool, int, float)):
        return raw, None
    s = str(raw).strip()
    if not s:
        return None, None
    if s[0] in "[{":
        try:
            parsed = yaml.safe_load(s)
        except yaml.YAMLError as e:
            return None, f"invalid yaml literal: {e}"
        if parsed is None:
            return None, None
        return parsed, None
    return s, None


@R.register("create_step")
class CreateStep(BaseStep):
    """Create or overwrite a markdown file. When the target already exists,
    its contents are replaced and a system notice is appended to the answer.

    Front matter fields are taken from any context key other than ``path``,
    ``content`` and ``metadata`` (Request system field). Keys starting with
    ``_`` are skipped as internal."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        content = self.context.get("content")
        content = "" if content is None else str(content)

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, err = gate_md(target, raw)
        if err:
            self._fail(err)
            return None

        existed = target.exists()

        meta: dict = {}
        for key, value in self.context.data.items():
            if key in _RESERVED_KEYS or key.startswith("_"):
                continue
            coerced, err = _coerce_frontmatter_value(value)
            if err:
                self._fail(f"`{key}`: {err}")
                return None
            if coerced is None:
                continue
            meta[key] = coerced

        if meta:
            # Use python-frontmatter to serialize so the output round-trips
            # through any standard front-matter reader (and our DefaultFileParser).
            post = frontmatter.Post(content, **meta)
            body = frontmatter.dumps(post)
        else:
            body = content
        if not body.endswith("\n"):
            body += "\n"

        try:
            await write_file_safe(target, body)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"create failed: {e}", path=str(target))
            return None

        nbytes = len(body.encode("utf-8"))
        self.context.response.success = True
        if existed:
            self.context.response.answer = (
                f"Created {target} ({nbytes} bytes) " f"[system notice: target already existed and was overwritten]"
            )
        else:
            self.context.response.answer = f"Created {target} ({nbytes} bytes)"
        self.logger.info(
            f"[{self.name}] created path={target} bytes={nbytes} overwritten={existed}",
        )
        return self.context.response
