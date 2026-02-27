"""Dashscope web search tool.

This module provides an operation that uses Alibaba Cloud's Dashscope API
to perform web searches with various search strategies.
"""

import os
from typing import Literal

from loguru import logger

from ...op import BaseTool
from ...schema import ToolCall


class DashscopeSearch(BaseTool):
    """Operation for performing web searches using Dashscope API.

    This operation uses Alibaba Cloud's Dashscope service to search the web
    with support for different search strategies (turbo, max, agent) and
    optional role-based prompting.
    """

    def __init__(
        self,
        model: str = "qwen-plus",  # qwen-flash
        search_strategy: Literal["turbo", "max", "agent"] = "turbo",  # agent only for qwen3-max
        enable_role_prompt: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.model: str = model
        self.search_strategy: Literal["turbo", "max", "agent"] = search_strategy
        self.enable_role_prompt: bool = enable_role_prompt

        # see ref: https://help.aliyun.com/zh/model-studio/web-search
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query",
                        },
                    },
                    "required": ["query"],
                },
            },
        )

    async def execute(self):
        query: str = self.context.query
        if self.enable_cache:
            cached_result = self.cache.load(query)
            if cached_result:
                return cached_result["response_content"]

        if self.enable_role_prompt:
            user_query = self.prompt_format("role_prompt", query=query)
        else:
            user_query = query
        logger.info(f"user_query={user_query}")
        messages: list = [{"role": "user", "content": user_query}]

        import dashscope

        response = await dashscope.AioGeneration.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            enable_search=True,
            search_options={
                "forced_search": True,
                "enable_source": True,
                "enable_citation": False,
                "search_strategy": self.search_strategy,
            },
            result_format="message",
        )

        search_results = []
        response_content = ""

        if response.output:
            if response.output.search_info:
                search_results = response.output.search_info.get("search_results", [])

            if response.output.choices and len(response.output.choices) > 0:
                response_content = response.output.choices[0].message.content

        final_result = {
            "query": query,
            "search_results": search_results,
            "response_content": response_content,
            "model": self.model,
            "search_strategy": self.search_strategy,
        }

        if self.enable_cache:
            self.cache.save(query, final_result, expire_hours=self.cache_expire_hours)

        return final_result["response_content"]
