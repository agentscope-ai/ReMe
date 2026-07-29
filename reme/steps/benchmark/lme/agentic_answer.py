"""LongMemEval agentic answer step – ReAct agent that answers questions using the search tool."""

from ....components import R
from ..base import BaseAgenticAnswerStep


@R.register("lme_agentic_answer_step")
class LmeAgenticAnswerStep(BaseAgenticAnswerStep):
    """Answer a LongMemEval query via ReAct agent with access to the search tool.

    The agent uses the ``agent_wrapper`` component in ReAct mode, calling the
    ``search`` job tool to retrieve relevant memory chunks before generating
    a final answer.

    Every job tool call carries an injected ``_search`` payload that enables
    session-transcript compression in ``search_v2_step`` and forwards the
    original benchmark query as the compression relevance filter
    (query-aware compression).
    """

    TOOL_CONTEXT_PREFIX = "lme_agentic_answer"

    def _injected_job_kwargs(self, query: str) -> dict:
        injected = super()._injected_job_kwargs(query)
        injected["_search"] = {"_compress": {"session": "true"}, "queries": [query], "type": "query-aware"}
        return injected
