"""memory agent"""

from .base_memory_agent import BaseMemoryAgent
from .personal.personal_longmemeval_retriever import PersonalLongmemevalRetriever
from .personal.personal_longmemeval_summarizer import PersonalLongmemevalSummarizer
from .personal.personal_retriever import PersonalRetriever
from .personal.personal_summarizer import PersonalSummarizer
from .personal.personal_v1_retriever import PersonalV1Retriever
from .personal.personal_v1_summarizer import PersonalV1Summarizer
from .personal.personal_halumem_retriever import PersonalHalumemRetriever
from .personal.personal_halumem_summarizer import PersonalHalumemSummarizer
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
    "PersonalV1Retriever",
    "PersonalV1Summarizer",
    "PersonalHalumemRetriever",
    "PersonalHalumemSummarizer",
    "PersonalLongmemevalRetriever",
    "PersonalLongmemevalSummarizer",
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
