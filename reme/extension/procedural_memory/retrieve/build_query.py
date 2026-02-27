"""Query building operation module.

This module provides functionality to build retrieval queries from either
explicit query strings or conversation messages, optionally using LLM to
generate optimized queries.
"""

from loguru import logger

from ....core.enumeration import Role
from ....core.op import BaseOp
from ....core.schema.message import Message
from ..utils import merge_messages_content


class BuildQuery(BaseOp):
    """Build retrieval query from context or messages.

    This operation constructs a query string for memory retrieval. It can use
    an explicit query from context, or generate one from conversation messages
    using either LLM-based generation or simple message concatenation.
    """

    async def execute(self):
        """Execute the query building operation.

        Builds a query string from either:
        1. An explicit query in the context
        2. Conversation messages (using LLM or simple concatenation)

        Stores the built query in context.query.
        """
        if "query" in self.context:
            query = self.context.query

        elif "messages" in self.context:
            if self.context.get("enable_llm_build", True):
                execution_process = merge_messages_content(self.context.messages)
                prompt = self.prompt_format(prompt_name="query_build", execution_process=execution_process)
                message = await self.llm.chat(messages=[Message(role=Role.USER, content=prompt)])
                query = message.content

            else:
                context_parts = []
                message_summaries = []
                for message in self.context.messages[-3:]:  # Last 3 messages
                    content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                    message_summaries.append(f"- {message.role.value}: {content}")
                if message_summaries:
                    context_parts.append("Recent messages:\n" + "\n".join(message_summaries))

                query = "\n\n".join(context_parts)

        else:
            raise RuntimeError("query or messages is required!")

        logger.info(f"build.query={query}")
        self.context.query = query
