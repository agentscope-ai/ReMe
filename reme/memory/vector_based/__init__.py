"""memory agent"""

from .base_memory_agent import BaseMemoryAgent
from .personal.personal_retriever import PersonalRetriever
from .personal.personal_summarizer import PersonalSummarizer
from .procedural.procedural_retriever import ProceduralRetriever
from .procedural.procedural_summarizer import ProceduralSummarizer
from .reme_retriever import ReMeRetriever
from .reme_summarizer import ReMeSummarizer
from .tool.tool_retriever import ToolRetriever
from .tool.tool_summarizer import ToolSummarizer
from ...core import R

__all__ = [
    "BaseMemoryAgent",
    "PersonalRetriever",
    "PersonalSummarizer",
    "ProceduralRetriever",
    "ProceduralSummarizer",
    "ReMeRetriever",
    "ReMeSummarizer",
    "ToolRetriever",
    "ToolSummarizer",
]

for name in __all__:
    agent_class = globals()[name]
    if (
        isinstance(agent_class, type)
        and issubclass(agent_class, BaseMemoryAgent)
        and agent_class is not BaseMemoryAgent
    ):
        R.ops.register(agent_class)
