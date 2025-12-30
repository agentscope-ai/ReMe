import os

from loguru import logger

from ...context import C
from ...op import BaseOp
from ...schema import ToolCall


@C.register_op()
class DashscopeSearch(BaseOp):

    def __init__(
        self,
        model: str = "qwen-plus",
        search_strategy: str = "max",
        enable_role_prompt: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model: str = model
        self.search_strategy: str = search_strategy
        self.enable_role_prompt: bool = enable_role_prompt
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.prompt.get("tool_description"),
                "input_schema": {
                    "query": {
                        "type": "string",
                        "description": "search keyword",
                        "required": True,
                    },
                },
            },
        )

    async def execute(self):
        query: str = self.context.query

        if self.enable_cache:
            cached_result = self.cache.load(query)
            if cached_result:
                self.output = cached_result["response_content"]
                return

        if self.enable_role_prompt:
            user_query = self.prompt.format(prompt_name="role_prompt", query=query)
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
            self.cache.save(query, final_result)

        self.output = final_result["response_content"]
