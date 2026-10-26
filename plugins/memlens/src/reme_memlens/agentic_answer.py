"""Agentic BM25 answer with scene-local, optional multimodal source access."""

from pathlib import Path
from uuid import uuid4

from agentscope.message import Base64Source, DataBlock, TextBlock, ToolResultState, UserMsg
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.steps.base_step import BaseStep
from reme.steps.file_io._path import _check_path_permission, resolve_path


def evidence_paths(workspace: Path, daily_dir: str, digest_dir: str, session_dir: str, sources: bool) -> list[str]:
    """List exact allowed evidence files, never agent state, settings, or another workspace."""
    roots = [(daily_dir, "*.md"), (digest_dir, "*.md")]
    if sources:
        roots.append((f"{session_dir}/dialog", "*.jsonl"))
    workspace = workspace.resolve()
    paths = []
    for directory, pattern in roots:
        root = (workspace / directory).resolve()
        if not root.is_relative_to(workspace):
            raise ValueError("Evidence directory must stay inside this scene workspace")
        for path in sorted(root.rglob(pattern)):
            if path.is_file() and path.resolve().is_relative_to(root):
                paths.append(path.relative_to(workspace).as_posix())
    return paths


def image_reader(step: BaseStep) -> FunctionTool:
    """Bridge the native read_image Job response to an AgentScope image result."""
    workspace = step.workspace_path.resolve()
    image_directory = f"{step.config_value('session_dir')}/images"

    async def read_image(path: str) -> ToolChunk:
        """View an image using its session/images path cited in a memory note."""
        target, error = resolve_path(workspace, path)
        if error or target is None or not _check_path_permission(workspace, target, [image_directory]):
            return ToolChunk(
                content=[TextBlock(text=error or "Image path must stay inside this scene's session image directory.")],
                state=ToolResultState.ERROR,
            )
        relative = target.relative_to(workspace).as_posix()
        response = await step.run_job("read_image", path=relative)
        metadata = response.metadata
        media_type = metadata.get("mime", "")
        if (
            not response.success
            or metadata.get("oversized")
            or metadata.get("non_image_warning")
            or not media_type.startswith("image/")
        ):
            notice = str(response.answer) if not response.success or metadata.get("oversized") else "Not an image file."
            return ToolChunk(content=[TextBlock(text=notice)], state=ToolResultState.ERROR)
        image = DataBlock(source=Base64Source(media_type=media_type, data=response.answer))
        return ToolChunk(content=[TextBlock(text=f"Image source: {relative}"), image], state=ToolResultState.SUCCESS)

    return FunctionTool(func=read_image, name="read_image", is_concurrency_safe=True, is_read_only=True)


class MemLensAgenticAnswerStep(BaseStep):
    """Run a fresh Agent for each question, with the official answer prompt unchanged."""

    async def execute(self):
        """Answer through a fresh, scene-scoped Agent and its allowed tools."""
        wrapper = self.agent_wrapper
        if not isinstance(wrapper, AsAgentWrapper):
            raise NotImplementedError("MemLens agentic_answer requires the AgentScope wrapper")
        query = self.context.get("query", "")
        if not query:
            raise ValueError("query must contain the benchmark answer prompt")
        max_iters = self.context.get("max_iters", 30)
        if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
            raise ValueError("max_iters must be a positive integer")
        sources = bool(self.context.get("include_source_sessions", True))
        images = bool(self.context.get("include_images", False))
        paths = evidence_paths(
            self.workspace_path,
            self.config_value("daily_dir"),
            self.config_value("digest_dir"),
            self.config_value("session_dir"),
            sources,
        )
        tool_context_id = f"memlens_answer_{uuid4().hex}"
        tools = []
        if images:
            tools.append(image_reader(self))
        injected = {
            "_allowed_paths": paths,
            "read_step_format_session": True,
            "search_filter": {"paths": paths},
            "vector_weight": 0.0,
        }
        # Standard ReMe job tools keep their normal file handling and lifecycle.
        tools.extend(
            wrapper._make_tool(job, tool_context_id, injected)  # pylint: disable=protected-access
            for job in wrapper._resolve_job_tools(["search", "read"])  # pylint: disable=protected-access
        )
        blocks = [TextBlock(text=query)]
        blocks.extend(DataBlock.model_validate(block) for block in self.context.get("query_images", []))
        if any(not block.source.media_type.startswith("image/") for block in blocks[1:]):
            raise ValueError("query_images accepts image DataBlocks only")
        prompt = self.get_prompt("system_prompt")
        if query_time := self.context.get("query_time"):
            prompt += f"\nQuestion time: {query_time}"
        if images:
            prompt += (
                "\nUse read_image(path=...) for the session/images paths cited in memory, including source_images."
            )
        else:
            prompt += "\nHistory images cannot be reopened in this setting."
        try:
            result = await wrapper.reply(
                UserMsg(name="user", content=blocks),
                system_prompt=prompt,
                toolkit=Toolkit(tools=tools),
                job_tools=[],
                builtin_tools=False,
                session_id=str(uuid4()),
                resume="",
                react_config={"max_iters": max_iters},
            )
            self.context.response.answer = (result.get("result") or "").strip()
            self.context.response.metadata.update(
                usage=result.get("usage"),
                agentscope_usage=(result.get("last_message") or {}).get("usage"),
                agent_session_id=result.get("session_id"),
                include_source_sessions=sources,
                include_images=images,
                max_iters=max_iters,
            )
        finally:
            if self.app_context is not None:
                self.app_context.metadata.get("tool_contexts", {}).pop(tool_context_id, None)
        return self.context.response
