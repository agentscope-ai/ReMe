"""memory tools"""

from .base_memory_tool import BaseMemoryTool
from .delegate_task import DelegateTask
from .history.add_history import AddHistory
from .history.read_history import ReadHistory
from .profiles.add_draft_and_read_all_profiles import AddDraftAndReadAllProfiles
from .profiles.profile_handler import ProfileHandler
from .profiles.read_all_profiles import ReadAllProfiles
from .profiles.update_profile import UpdateProfile
from .return_relevant_content import ReturnRelevantContent
from .vector.add_draft_and_retrieve_similar_memory import AddDraftAndRetrieveSimilarMemory
from .vector.add_memory import AddMemory
from .vector.delete_memory import DeleteMemory
from .vector.memory_handler import MemoryHandler
from .vector.retrieve_memory import RetrieveMemory
from .vector.retrieve_recent_memory import RetrieveRecentMemory
from .vector.update_memory import UpdateMemory
from .vector.update_memory_v2 import UpdateMemoryV2
from ...core import R

__all__ = [
    # Base
    "BaseMemoryTool",
    "DelegateTask",
    # History
    "AddHistory",
    "ReadHistory",
    # Profiles
    "AddDraftAndReadAllProfiles",
    "ProfileHandler",
    "ReadAllProfiles",
    "UpdateProfile",
    # Return
    "ReturnRelevantContent",
    # Vector
    "AddDraftAndRetrieveSimilarMemory",
    "AddMemory",
    "DeleteMemory",
    "MemoryHandler",
    "RetrieveMemory",
    "RetrieveRecentMemory",
    "UpdateMemory",
    "UpdateMemoryV2",
]

for name in __all__:
    tool_class = globals()[name]
    if isinstance(tool_class, type) and issubclass(tool_class, BaseMemoryTool) and tool_class is not BaseMemoryTool:
        R.op.register()(tool_class)
