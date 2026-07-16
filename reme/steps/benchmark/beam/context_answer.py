"""Answer a probing question from retrieved memory context (BEAM benchmark)."""

from ...base_step import BaseStep
from ....components import R


@R.register("beam_context_answer_step")
class BeamContextAnswerStep(BaseStep):
    """Answer a probing question using BEAM's original RAG prompt.

    Context inputs:
        retrieved_context: concatenated memory chunks from search
        question: the probing question to answer
    """

    async def execute(self):
        assert self.context is not None
        retrieved_context: str = self.context.get("retrieved_context", "")
        question: str = self.context.get("question", "")

        if not question:
            raise ValueError("beam_context_answer_step requires non-empty question")

        if not retrieved_context:
            retrieved_context = "(no relevant context found)"

        if self.agent_wrapper is None:
            raise ValueError("beam_context_answer_step requires agent_wrapper")

        user_prompt = self.prompt_format(
            "user_message",
            context=retrieved_context,
            question=question,
        )
        result = await self.agent_wrapper.reply(user_prompt)
        answer = (result.get("result") or "").strip()

        self.logger.info(f"[{self.name}] beam context answer: {answer[:200]}")
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "question": question,
                "retrieved_context_preview": retrieved_context[:500],
                "answer": answer,
            },
        )
        return self.context.response
