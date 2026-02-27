"""Operation for dumping memories from vector store to JSONL file."""

import json
from pathlib import Path
from typing import List

from loguru import logger

from ...core.op import BaseOp
from ...core.schema.memory_node import MemoryNode
from ...core.schema.vector_node import VectorNode


class DumpMemory(BaseOp):
    """Operation that dumps memories from vector store to a JSONL file.

    This operation retrieves all memories from the vector store, converts them
    to MemoryNode objects, and writes them to a JSONL file (one JSON object
    per line) for backup or export purposes.
    """

    async def execute(self):
        """Execute the memory dump operation.

        Dumps all memories from the vector store to a JSONL file:
        1. Retrieves all VectorNodes from the vector store
        2. Converts them to MemoryNode objects
        3. Writes each MemoryNode as a JSON line to the output file

        Expected context attributes:
            dump_file_path: Path to the output JSONL file.

        Sets context attributes:
            dumped_count: Number of memories dumped to the file.
        """
        # Support both dump_file_path and path for backward compatibility
        dump_file_path: str = self.context.dump_file_path
        if not dump_file_path:
            logger.error("dump_file_path is required in context")
            return

        file_path = Path(dump_file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Retrieve all nodes from vector store
        vector_nodes: List[VectorNode] = await self.vector_store.list()
        logger.info(f"Retrieved {len(vector_nodes)} nodes from vector store")

        # Convert to MemoryNodes and write to JSONL file
        dumped_count = 0
        with open(file_path, "w", encoding="utf-8") as f:
            for node in vector_nodes:
                try:
                    memory = MemoryNode.from_vector_node(node)
                    # Write as JSON line (one JSON object per line)
                    json_line = json.dumps(memory.model_dump(exclude_none=True), ensure_ascii=False)
                    f.write(json_line + "\n")
                    dumped_count += 1
                except Exception as e:
                    logger.warning(f"Failed to convert and dump node {node.vector_id}: {e}")
                    continue

        logger.info(f"Dumped {dumped_count} memories to {dump_file_path}")
