"""Create a new markdown file with optional YAML front matter (title/tags/status)."""

import frontmatter
import yaml

from ._file_io import gate_md, resolve_path, write_file_safe
from ..base_step import BaseStep
from ...components import R


def _parse_tags(raw) -> tuple[list[str] | None, str | None]:  # pylint: disable=too-many-return-statements
    """Accept a list, a YAML/JSON-ish string (e.g. "[]", '["a","b"]'), or empty.

    Returns ``(tags, error)``. ``tags is None`` means no tags key should be emitted.
    """
    if raw is None:
        return None, None
    if isinstance(raw, list):
        return [str(t) for t in raw] or None, None
    s = str(raw).strip()
    if not s:
        return None, None
    try:
        parsed = yaml.safe_load(s)
    except yaml.YAMLError as e:
        return None, f"`tags` must be a list (got invalid yaml: {e})"
    if parsed is None:
        return None, None
    if not isinstance(parsed, list):
        return None, f"`tags` must be a list, got {type(parsed).__name__}"
    return [str(t) for t in parsed] or None, None


@R.register("create_step")
class CreateStep(BaseStep):
    """Create or overwrite a markdown file. When the target already exists,
    its contents are replaced and a system notice is appended to the answer."""

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
        title = self.context.get("title")
        status = self.context.get("status")
        tags_raw = self.context.get("tags")

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, err = gate_md(target, raw)
        if err:
            self._fail(err)
            return None

        existed = target.exists()

        tags, err = _parse_tags(tags_raw)
        if err:
            self._fail(err)
            return None

        meta: dict = {}
        if title not in (None, ""):
            meta["title"] = str(title)
        if tags:
            meta["tags"] = tags
        if status not in (None, ""):
            meta["status"] = str(status)

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
