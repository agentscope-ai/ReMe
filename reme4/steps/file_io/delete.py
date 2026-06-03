"""``file_delete`` — hard-delete a file or folder under the vault (reports inbound refs).

Removes the file (or folder tree) from disk; the watcher then prunes
the affected chunks from all projections (vector / keyword / file_graph).
For **soft** deletion that keeps a file addressable, use
``property_update(status=archived)`` instead — that's the path
structure.md's Decay algorithm is designed around.

Reference reporting (no auto-fix). Before deleting, the inbound
wikilink count for each doomed ``.md`` file is captured from the
file_graph's reverse index (literal full-path matching, same rule
as the wikilink retarget helper). For folder deletes, sources that live inside
the same folder are filtered out — those links die alongside the
delete and can't surface as dangling. The remaining inbound list
is the agent's punch list for follow-up rewrites.

The delete itself is unconditional: this step does not rewrite
inbound references on the agent's behalf because there is no
canonical "new target" — the agent decides per-reference whether
to rewrite (via ``file_move`` to a merge target), edit the
citing prose, or accept dangling.
"""

import shutil
from pathlib import Path

from ..base_step import BaseStep
from ...components import R
from ...utils.wikilink_handler import WikilinkHandler


def _is_inside(rel: str, folder_rel: str) -> bool:
    """``rel`` is the same as or nested under ``folder_rel`` (relative to the vault)."""
    prefix = folder_rel.rstrip("/") + "/"
    return rel == folder_rel or rel.startswith(prefix)


@R.register("delete_step")
class DeleteStep(BaseStep):
    """Hard-delete the path at ``path`` (file or folder, relative to the vault)."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        payload = await self._delete(path)
        if "error" in payload:
            self.context.response.success = False
            self.context.response.answer = f"Error: {payload['error']}"
            self.logger.info(f"[{self.name}] delete failed path={path} error={payload['error']!r}")
        elif payload.get("is_dir"):
            self.context.response.success = True
            self.context.response.answer = f"Deleted directory {path} ({len(payload['deleted_files'])} file(s))"
            self.logger.info(
                f"[{self.name}] deleted dir path={path} files={len(payload['deleted_files'])} "
                f"inbound_files={payload['inbound']['files_touched']} "
                f"inbound_links={payload['inbound']['links_total']}",
            )
        else:
            self.context.response.success = True
            self.context.response.answer = f"Deleted {path}"
            self.logger.info(
                f"[{self.name}] deleted file path={path} "
                f"inbound_files={payload['inbound']['files_touched']} "
                f"inbound_links={payload['inbound']['links_total']}",
            )
        self.context.response.metadata.update(payload)

    async def _delete(self, path: str) -> dict:
        if not path:
            return {"path": path, "error": "not found"}
        vault_dir = Path(self.file_store.vault_path or ".").resolve()
        target = (vault_dir / path).resolve()
        if target.is_file():
            inbound = await WikilinkHandler.find_inbound(self.file_store, target=path)
            target.unlink()
            return {
                "path": path,
                "deleted": True,
                "is_dir": False,
                "deleted_files": [path],
                "inbound": {
                    "files_touched": inbound.get("files_touched", 0),
                    "links_total": inbound.get("links_total", 0),
                    "by_file": inbound.get("by_file", []),
                },
            }

        if target.is_dir():
            folder_rel = path.rstrip("/")
            deleted_files: list[str] = []
            per_target: list[dict] = []
            external_sources: set[str] = set()
            links_total = 0

            for md in sorted(target.rglob("*.md")):
                try:
                    rel = str(md.relative_to(vault_dir))
                except ValueError:
                    continue
                deleted_files.append(rel)
                inbound = await WikilinkHandler.find_inbound(self.file_store, target=rel)
                # Drop sources that also live inside the doomed folder —
                # their links vanish with them and aren't actionable.
                external = [row for row in inbound.get("by_file", []) if not _is_inside(row["path"], folder_rel)]
                if not external:
                    continue
                target_total = sum(row["count"] for row in external)
                links_total += target_total
                external_sources.update(row["path"] for row in external)
                per_target.append(
                    {
                        "target": rel,
                        "files_touched": len(external),
                        "links_total": target_total,
                        "by_file": external,
                    },
                )

            # Also enumerate non-md files for the deleted_files report.
            for entry in sorted(target.rglob("*")):
                if not entry.is_file() or entry.suffix == ".md":
                    continue
                try:
                    rel = str(entry.relative_to(vault_dir))
                except ValueError:
                    continue
                deleted_files.append(rel)

            shutil.rmtree(target)
            return {
                "path": path,
                "deleted": True,
                "is_dir": True,
                "deleted_files": sorted(deleted_files),
                "inbound": {
                    "files_touched": len(external_sources),
                    "links_total": links_total,
                    "by_target": per_target,
                },
            }

        return {"path": path, "error": "not found"}
