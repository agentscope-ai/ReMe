"""Answer a query directly from the supplied session context."""

from ...base_step import BaseStep
from ....components import R


@R.register("context_answer_step")
class ContextAnswerStep(BaseStep):
    """Answer a query using the LongMemEval direct-reading prompt."""

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        session_context: str = self.context.get("session_context", "")
        current_date: str = self.context.get("current_date", "")

        if not query:
            raise ValueError("context_answer_step requires non-empty query")
        if not session_context:
            raise ValueError("context_answer_step requires non-empty session_context")
        if self.agent_wrapper is None:
            raise ValueError("context_answer_step requires agent_wrapper")

        user_prompt = self.prompt_format(
            "user_message",
            session_context=session_context,
            current_date=current_date,
            query=query,
        )
        result = await self.agent_wrapper.reply(user_prompt)
        answer = (result.get("result") or "").strip()

        self.logger.info(f"[{self.name}] context answer: {answer}")
        self.context["context_answer"] = answer
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "query": query,
                "session_context": session_context,
                "current_date": current_date,
                "context_answer": answer,
            },
        )
        return self.context.response
