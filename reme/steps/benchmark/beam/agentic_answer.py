"""BEAM agentic answer step – ReAct agent that answers questions using the search tool."""

from ....components import R
from ..base import BaseAgenticAnswerStep


@R.register("beam_agentic_answer_step")
class BeamAgenticAnswerStep(BaseAgenticAnswerStep):
    """Answer a BEAM probing question via ReAct agent with access to the search tool.

    The agent uses the ``agent_wrapper`` component in ReAct mode, calling the
    ``search`` job tool to retrieve relevant memory chunks before generating
    a final answer.
    """

    TOOL_CONTEXT_PREFIX = "beam_agentic_answer"
