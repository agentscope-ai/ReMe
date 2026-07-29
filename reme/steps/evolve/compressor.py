"""compressor — compress text via a direct LLM call (no agent)."""

from agentscope.message import UserMsg
from agentscope.model import ChatResponse

from ..base_step import BaseStep
from ...components import R


@R.register("compressor_step")
class CompressorStep(BaseStep):
    """Compress ``text`` with a direct LLM call (no agent wrapper).

    Inputs (from RuntimeContext):
        text    (str, required): the text to compress.
        queries (list[str], optional, default []): when provided, compression
            keeps anything potentially relevant to any of the queries and may
            drop content that is certainly irrelevant and certainly unhelpful
            for all of them.

    Output (written to context.response.answer):
        The compressed text.
    """

    async def execute(self):
        assert self.context is not None
        text: str = self.context.get("text", "") or ""
        raw_queries = self.context.get("queries") or []
        queries: list[str] = [str(q).strip() for q in raw_queries if str(q).strip()]

        if not text.strip():
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty text"
            return

        if queries:
            queries_block = "\n".join(f"- {q}" for q in queries)
            user_message = self.prompt_format("compress_query_prompt", queries=queries_block, text=text)
        else:
            user_message = self.prompt_format("compress_prompt", text=text)

        result = await self.as_llm([UserMsg(name="user", content=user_message)])
        compressed = await self._response_text(result)
        self.logger.info(
            f"[{self.name}] compressed {len(text)} -> {len(compressed)} chars (queries={len(queries)})",
        )

        self.context.response.success = True
        self.context.response.answer = compressed
        self.context.response.metadata.update(
            {
                "queries": queries,
                "original_length": len(text),
                "compressed_length": len(compressed),
            },
        )

    @staticmethod
    async def _response_text(result) -> str:
        """Extract text blocks from a streaming or non-streaming ChatResponse.

        The async-iteration check is done on the type: ``ChatResponse`` inherits
        agentscope's ``DictMixin`` whose instance ``__getattr__`` raises
        ``KeyError`` (not ``AttributeError``), so ``hasattr(result, ...)``
        would crash instead of returning False.
        """
        if hasattr(type(result), "__aiter__"):
            last: ChatResponse | None = None
            async for chunk in result:
                last = chunk
            result = last
        if result is None:
            return ""
        # Content blocks may be plain dicts or pydantic models (TextBlock/ThinkingBlock).
        parts: list[str] = []
        for block in result.content or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            elif getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts).strip()
