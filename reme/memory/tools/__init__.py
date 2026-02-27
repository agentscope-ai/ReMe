"""memory tools"""

from .base_memory_tool import BaseMemoryTool

# chunk tools
from .chunk.memory_get import MemoryGet
from .chunk.memory_search import MemorySearch
from .delegate_task import DelegateTask

# history tools
from .history.add_history import AddHistory
from .history.read_history import ReadHistory
from .history.read_history_v2 import ReadHistoryV2

# profiles tools
from .profiles.add_draft_and_read_all_profiles import AddDraftAndReadAllProfiles
from .profiles.add_profile import AddProfile
from .profiles.delete_profile import DeleteProfile
from .profiles.read_all_profiles import ReadAllProfiles
from .profiles.update_profile import UpdateProfile
from .profiles.update_profiles_v1 import UpdateProfilesV1

# record tools
from .record.add_and_retrieve_similar_memory import AddAndRetrieveSimilarMemory
from .record.add_draft_and_retrieve_similar_memory import AddDraftAndRetrieveSimilarMemory
from .record.add_memory import AddMemory
from .record.delete_memory import DeleteMemory
from .record.retrieve_memory import RetrieveMemory
from .record.retrieve_recent_memory import RetrieveRecentMemory
from .record.update_memory import UpdateMemory
from .record.update_memory_v1 import UpdateMemoryV1
from .record.update_memory_v2 import UpdateMemoryV2
from ...core import R

__all__ = [
    # base
    "BaseMemoryTool",
    "DelegateTask",
    # chunk tools
    "MemoryGet",
    "MemorySearch",
    # history tools
    "AddHistory",
    "ReadHistory",
    "ReadHistoryV2",
    # profiles tools
    "AddDraftAndReadAllProfiles",
    "AddProfile",
    "DeleteProfile",
    "ReadAllProfiles",
    "UpdateProfile",
    "UpdateProfilesV1",
    # record tools
    "AddAndRetrieveSimilarMemory",
    "AddDraftAndRetrieveSimilarMemory",
    "AddMemory",
    "DeleteMemory",
    "RetrieveMemory",
    "RetrieveRecentMemory",
    "UpdateMemory",
    "UpdateMemoryV1",
    "UpdateMemoryV2",
]

for name in __all__:
    tool_class = globals()[name]
    R.ops.register(tool_class)
