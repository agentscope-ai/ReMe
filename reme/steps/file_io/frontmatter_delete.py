"""``frontmatter_delete_step`` — remove keys from a markdown file's frontmatter.

Returns both ``deleted`` (keys that were present and removed) and
``missing`` (keys that weren't there) so the agent can tell whether
a no-op happened. The file is rewritten only when at least one key
is actually removed — calling delete with all-missing keys is a
zero-side-effect read.

``path`` is a path relative to the workspace.

Permission: honors the request-scoped ``_allowed_paths`` constraint
injected by the server into the RuntimeContext; without it, restricting
read/edit/write alone would still leave frontmatter of arbitrary workspace
Markdown files mutable.
"""

import frontmatter

from ._base import frontmatter_target
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

        allowed_paths = self.context.get("_allowed_paths")
        async with frontmatter_target(self.file_store.workspace_path, path, allowed_paths) as (target, payload):
            if target is not None:
                if not keys:
                    payload["error"] = "keys is empty"
                else:
                    post = frontmatter.loads(target.read_text(encoding="utf-8"))
                    deleted: list[str] = []
                    missing: list[str] = []
                    for key in keys:
                        if key in post.metadata:
                            del post.metadata[key]
                            deleted.append(key)
                        else:
                            missing.append(key)
                    if deleted:
                        target.write_text(frontmatter.dumps(post), encoding="utf-8")
                    payload.update(deleted=deleted, missing=missing, frontmatter=dict(post.metadata))

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
