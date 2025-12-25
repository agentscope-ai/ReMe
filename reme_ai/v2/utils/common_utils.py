import datetime
from typing import List, Dict

from ..enumeration import Role
from ..schema import Message, MemoryNode


def get_now_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_messages(messages: List[dict | Message]) -> str:
    messages = [Message(**x) if isinstance(x, dict) else x for x in messages]
    messages = [x for x in messages if x.role is not Role.SYSTEM]
    messages_context = "\n".join([x.format_message(
        add_time_created=True,
        use_name_first=True,
        add_reasoning_content=True,
        add_tool_calls=True,
    ) for x in messages])
    return messages_context


def deduplicate_memories(memories: List[MemoryNode]) -> List[MemoryNode]:
    seen_memories: Dict[str, MemoryNode] = {}
    for memory in memories:
        if memory.memory_id not in seen_memories:
            seen_memories[memory.memory_id] = memory
    return list(seen_memories.values())
