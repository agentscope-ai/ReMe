"""``frontmatter_delete_step`` — remove keys from a markdown file's frontmatter.

Returns both ``deleted`` (keys that were present and removed) and
``missing`` (keys that weren't there) so the agent can tell whether
a no-op happened. The file is rewritten only when at least one key
is actually removed — calling delete with all-missing keys is a
zero-side-effect read.

``path`` is a path relative to the vault.
"""

from pathlib import Path

import frontmatter

from ..base_step import BaseStep
from ...components import R


@R.register("frontmatter_delete_step")
class FrontmatterDeleteStep(BaseStep):
    """Remove keys from a Markdown file's frontmatter."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        keys = self.context.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        keys = list(keys)

        target = (Path(self.file_store.vault_path or ".") / path).resolve()
        if not target.is_file():
            payload: dict = {"path": path, "error": "not found"}
        elif target.suffix != ".md":
            payload = {"path": path, "error": "not markdown"}
        elif not keys:
            payload = {"path": path, "error": "keys is empty"}
        else:
            post = frontmatter.loads(target.read_text(encoding="utf-8"))
            deleted: list[str] = []
            missing: list[str] = []
            for k in keys:
                if k in post.metadata:
                    del post.metadata[k]
                    deleted.append(k)
                else:
                    missing.append(k)
            if deleted:
                target.write_text(frontmatter.dumps(post), encoding="utf-8")
            payload = {
                "path": path,
                "deleted": deleted,
                "missing": missing,
                "frontmatter": dict(post.metadata),
            }

        if "error" in payload:
            self.context.response.success = False
            self.context.response.answer = f"Error: {payload['error']}"
            self.logger.info(f"[{self.name}] delete failed path={path} error={payload['error']!r}")
        else:
            self.context.response.success = True
            self.context.response.answer = f"Deleted {len(payload['deleted'])} key(s) from {path}"
            self.logger.info(
                f"[{self.name}] path={path} deleted={payload['deleted']} missing={payload['missing']}",
            )
        self.context.response.metadata.update(payload)
