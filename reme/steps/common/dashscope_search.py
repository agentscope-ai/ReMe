"""Web search through DashScope, preserving provider search results."""

import asyncio
import os
from contextlib import aclosing
from http import HTTPStatus

from ..base_step import BaseStep
from ...components import R


@R.register("dashscope_search_step")
class DashScopeSearchStep(BaseStep):
    """Return raw search sources and separate model-generated content."""

    async def execute(self):
        assert self.context is not None
        response = self.context.response
        query = self.context.get("query")
        query = "" if query is None else str(query).strip()
        api_key = (self.kwargs.get("api_key") or os.getenv("DASHSCOPE_API_KEY", "")).strip()
        if not query or not api_key:
            response.success = False
            response.answer = "query is required" if not query else "DASHSCOPE_API_KEY is required"
            return response

        from aiohttp import ClientSession
        from dashscope import AioMultiModalConversation

        model = self.kwargs.get("model", "qwen3.8-flash")
        search_info, sources, content, usage = {}, {}, [], {}
        request_id = ""
        async with asyncio.timeout(self.kwargs.get("timeout", 60)), ClientSession(trust_env=True) as session:
            stream = await AioMultiModalConversation.call(
                api_key=api_key,
                session=session,
                model=model,
                messages=[{"role": "user", "content": [{"text": query}]}],
                enable_search=True,
                enable_thinking=False,
                stream=True,
                incremental_output=True,
                search_options={
                    "forced_search": True,
                    "enable_source": True,
                    "search_strategy": self.kwargs.get("search_strategy", "turbo"),
                },
            )
            async with aclosing(stream):
                async for chunk in stream:
                    if chunk.status_code != HTTPStatus.OK:
                        raise RuntimeError(f"DashScope {chunk.code}: {chunk.message} (request_id={chunk.request_id})")
                    request_id = chunk.request_id or request_id
                    if any((chunk.usage or {}).values()):
                        usage = dict(chunk.usage)
                    output = chunk.output or {}
                    info = output.get("search_info") or {}
                    search_info.update({key: value for key, value in info.items() if key != "search_results"})
                    for item in info.get("search_results") or []:
                        sources.setdefault(item["url"], item)
                    for choice in output.get("choices") or []:
                        text = (choice.get("message") or {}).get("content") or []
                        if isinstance(text, str):
                            content.append(text)
                        else:
                            content.extend(part["text"] for part in text if "text" in part)

        search_info["search_results"] = list(sources.values())
        response.answer = {"search_info": search_info, "content": "".join(content)}
        response.metadata.update(model=model, request_id=request_id, usage=usage)
        return response
