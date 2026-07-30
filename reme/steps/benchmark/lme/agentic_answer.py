"""LongMemEval agentic answer step – ReAct agent that answers questions using the search tool."""

from ....components import R
from ..base import BaseAgenticAnswerStep


@R.register("lme_agentic_answer_step")
class LmeAgenticAnswerStep(BaseAgenticAnswerStep):
    """Answer a LongMemEval query via ReAct agent with access to the search tool.

    The agent uses the ``agent_wrapper`` component in ReAct mode, calling the
    ``search`` job tool to retrieve relevant memory chunks before generating
    a final answer.
    """

    TOOL_CONTEXT_PREFIX = "lme_agentic_answer"
