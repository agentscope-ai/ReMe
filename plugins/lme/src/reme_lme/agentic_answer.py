"""LongMemEval agentic answer step – ReAct agent that answers questions using the search tool."""

from reme.steps.benchmark import BaseAgenticAnswerStep


class LmeAgenticAnswerStep(BaseAgenticAnswerStep):
    """Answer a LongMemEval query via ReAct agent with access to the search tool.

    The agent uses the ``agent_wrapper`` component in ReAct mode, calling the
    ``search`` job tool to retrieve relevant memory chunks before generating
    a final answer.

    Session-transcript compression in the plugin's search Step is controlled by the
    ``compress_session`` flag in the runtime context (set by the benchmark
    runner from ``evaluation.compress_session``); it is off by default.
    """

    TOOL_CONTEXT_PREFIX = "lme_agentic_answer"
