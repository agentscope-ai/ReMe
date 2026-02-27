"""Operation for loading memories from JSONL file to vector store."""

import json
import asyncio
from pathlib import Path
from typing import List

from loguru import logger

from ...core.op import BaseOp
from ...core.schema.memory_node import MemoryNode
from ...core.schema.vector_node import VectorNode


class LoadMemory(BaseOp):
    """Operation that loads memories from a JSONL file to vector store.

    This operation reads MemoryNode objects from a JSONL file (one JSON object
    per line), converts them to VectorNode objects, and inserts them into the
    vector store.
    """

    async def execute(self):
        """Execute the memory load operation.

        Loads memories from a JSONL file to the vector store:
        1. Reads each line from the JSONL file
        2. Parses JSON and creates MemoryNode objects
        3. Converts MemoryNodes to VectorNodes
        4. Inserts them into the vector store

        Expected context attributes:
            load_file_path: Path to the input JSONL file.
            clear_existing: Optional. If True, clears existing memories before loading (default: False).

        Sets context attributes:
            loaded_count: Number of memories loaded from the file.
        """
        load_file_path: str = self.context.load_file_path
        if not load_file_path:
            logger.error("load_file_path is required in context")
            return

        file_path = Path(load_file_path)
        if not file_path.exists():
            logger.error(f"File not found: {load_file_path}")
            return

        try:
            # Attempt to retrieve the event loop associated with the current thread
            loop = asyncio.get_running_loop()
            print(f"Running event loop found: {loop}")
        except RuntimeError:
            # Start a new event loop to run the coroutine to completion
            print("No running event loop found, starting a new one")

        clear_existing: bool = self.context.get("clear_existing", False)
        if clear_existing:
            await self.vector_store.delete_all()
            logger.info("Cleared existing memories from vector store")

        # Read and parse JSONL file
        memory_nodes: List[MemoryNode] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    memory = MemoryNode.model_validate(data)
                    memory_nodes.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num} in {load_file_path}: {e}")
                    continue
        logger.info(f"Parsed {len(memory_nodes)} memories from {load_file_path}")

        # Convert to VectorNodes and insert into vector store
        if memory_nodes:
            vector_nodes: List[VectorNode] = [memory.to_vector_node() for memory in memory_nodes]
            await self.vector_store.insert(nodes=vector_nodes)
            logger.info(f"Loaded {len(memory_nodes)} memories into vector store")
