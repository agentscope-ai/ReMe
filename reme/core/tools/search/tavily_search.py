"""Tavily web search tool.

This module provides an operation that uses the Tavily API to perform
web searches and optionally extract content from search results.
"""

import json
import os

from loguru import logger

from ...op import BaseTool
from ...schema import ToolCall


class TavilySearch(BaseTool):
    """Operation for performing web searches using Tavily API.

    This operation uses the Tavily search service to find web content
    and optionally extract raw content from the results, with configurable
    character limits for individual items and total content.
    """

    def __init__(
        self,
        enable_extract: bool = True,
        item_max_char_count: int = 20000,
        all_max_char_count: int = 50000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_extract: bool = enable_extract
        self.item_max_char_count: int = item_max_char_count
        self.all_max_char_count: int = all_max_char_count
        self._client = None

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

    @property
    def client(self):
        """Get or create the Tavily async client instance.

        Returns:
            AsyncTavilyClient: The Tavily client instance, lazily initialized.
        """
        if self._client is None:
            from tavily import AsyncTavilyClient

            self._client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        return self._client

    async def execute(self):
        query: str = self.context.query
        logger.info(f"tavily_search query={query}")

        if self.enable_cache:
            cached_result = self.cache.load(query)
            if cached_result:
                return json.dumps(cached_result, ensure_ascii=False, indent=2)

        response = await self.client.search(query=query)
        logger.info(f"tavily_search response={response}")

        if not self.enable_extract:
            if not response.get("results"):
                raise RuntimeError("tavily return empty result")

            final_result = {item["url"]: item for item in response["results"]}

            if self.enable_cache and final_result:
                self.cache.save(query, final_result, expire_hours=self.cache_expire_hours)

            return json.dumps(final_result, ensure_ascii=False, indent=2)

        url_info_dict = {item["url"]: item for item in response["results"]}
        response_extract = await self.client.extract(urls=[item["url"] for item in response["results"]])
        logger.info(f"tavily.response_extract: {response_extract}")

        final_result = {}
        all_char_count = 0
        for item in response_extract["results"]:
            url = item["url"]
            raw_content: str = item["raw_content"]
            if len(raw_content) > self.item_max_char_count:
                raw_content = raw_content[: self.item_max_char_count]
            if all_char_count + len(raw_content) > self.all_max_char_count:
                raw_content = raw_content[: self.all_max_char_count - all_char_count]

            if raw_content:
                final_result[url] = url_info_dict[url]
                final_result[url]["raw_content"] = raw_content
                all_char_count += len(raw_content)

        if not final_result:
            raise RuntimeError("tavily return empty result")

        if self.enable_cache and final_result:
            self.cache.save(query, final_result, expire_hours=self.cache_expire_hours)

        return json.dumps(final_result, ensure_ascii=False, indent=2)
