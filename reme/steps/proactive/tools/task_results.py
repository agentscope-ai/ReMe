"""Desktop task-result browsing with an independent user-read receipt."""

# pylint: disable=protected-access

import asyncio

import frontmatter

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R
from ....utils.async_utils import complete_in_thread
from ....utils.file_utils import get_path_lock


@R.register("task_results_step")
class TaskResultsStep(BaseStep):
    """Browse validated result documents and record explicit user reading."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    def _post(self, path):
        post = frontmatter.loads(self.pa.tasks.read(path))
        if post.get("task_id") != path.parent.parent.name or not post.content.strip():
            raise ValueError("Invalid task result identity or body")
        return post

    def _result_path(self, raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError("A result path is required")
        files = self.pa.tasks
        path = files.checked(self.workspace_path / raw)
        parts = path.relative_to(files.root_path).parts
        if len(parts) != 3 or path not in files.paths(parts[0]):
            raise ValueError("path must identify a published task result")
        return path

    def _list(self, task_id, unread, offset, limit):
        files = self.pa.tasks
        roots = list(files.root_path.iterdir()) if files.root_path.exists() else []
        task_ids = sorted(root.name for root in roots if root.is_dir() and files.paths(root.name))
        roots = [files.task_path(task_id)] if task_id else roots
        results = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in files.paths(root.name):
                post = self._post(path)
                if unread and post.get("user_read_at") is not None:
                    continue
                results.append(
                    {
                        "path": self.to_workspace_relative(path),
                        "task_id": post["task_id"],
                        "created_at": post.get("created_at"),
                        "user_read_at": post.get("user_read_at"),
                        "status": post.get("status"),
                        "preview": post.content[:240],
                    },
                )
        results.sort(key=lambda item: (str(item["created_at"] or ""), item["path"]), reverse=True)
        return {
            "items": results[offset : offset + limit],
            "total": len(results),
            "offset": offset,
            "limit": limit,
            "task_ids": task_ids,
        }

    async def execute(self):
        operation = self.context.get("action", "list")
        files = self.pa.tasks
        if not _check_path_permission(self.workspace_path, files.root_path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to browse task results")
        if operation == "list":
            offset, limit = self.context.get("offset", 0), self.context.get("limit", 50)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
                raise ValueError("offset must be nonnegative and limit must be 1-100")
            unread = self.context.get("unread", False)
            if type(unread) is not bool:
                raise ValueError("unread must be boolean")
            self.context.response.answer = await asyncio.to_thread(
                self._list,
                self.context.get("task_id"),
                unread,
                offset,
                limit,
            )
        elif operation == "read":
            path = self._result_path(self.context.get("path"))
            mark_read = self.context.get("mark_read", False)
            if type(mark_read) is not bool:
                raise ValueError("mark_read must be boolean")
            async with files.lock:
                async with await get_path_lock(path):
                    post = await asyncio.to_thread(self._post, path)
                    if mark_read and post.get("user_read_at") is None:
                        post["user_read_at"] = self.pa._now().isoformat()
                        await complete_in_thread(self.pa._write_text, path, files.markdown(post))
                    self.context.response.answer = {
                        "path": self.to_workspace_relative(path),
                        "content": frontmatter.dumps(post) + "\n",
                        "user_read_at": post.get("user_read_at"),
                    }
        else:
            raise ValueError("action must be list or read")
        return self.context.response
