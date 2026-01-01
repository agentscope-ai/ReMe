"""Base embedding model interface for ReMe.

Defines the abstract base class and standard API for all embedding model implementations.
"""

import asyncio
import time
from abc import ABC

from loguru import logger

from ..schema import VectorNode


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding model implementations.

    Provides a standard interface for text-to-vector generation with
    built-in batching, retry logic, and error handling.
    """

    def __init__(
        self,
        model_name: str = "",
        dimensions: int = 1024,
        max_batch_size: int = 10,
        max_retries: int = 3,
        raise_exception: bool = True,
        **kwargs,
    ):
        """Initialize model configuration and parameters."""
        self.model_name = model_name
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.raise_exception = raise_exception
        self.kwargs = kwargs

    async def _get_embeddings(self, input_text: list[str]) -> list[list[float]]:
        """Internal async implementation for calling the embedding API with batch input."""

    def _get_embeddings_sync(self, input_text: list[str]) -> list[list[float]]:
        """Internal synchronous implementation for calling the embedding API with batch input."""

    async def get_embedding(self, input_text: str) -> list[float]:
        """Async get embedding for a single text with exponential backoff retries."""
        for i in range(self.max_retries):
            try:
                result = await self._get_embeddings([input_text])
                return result[0]
            except Exception as e:
                logger.error(f"Model {self.model_name} failed: {e}")
                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise
                    return []
                await asyncio.sleep(i + 1)
        return []

    async def get_embeddings(self, input_text: list[str]) -> list[list[float]]:
        """Async get embeddings with automatic batching and exponential backoff retries."""
        # Split into batches and process sequentially to respect rate limits
        results = []
        for i in range(0, len(input_text), self.max_batch_size):
            batch = input_text[i : i + self.max_batch_size]
            # Process each batch with retry logic
            for retry in range(self.max_retries):
                try:
                    batch_res = await self._get_embeddings(batch)
                    if batch_res:
                        results.extend(batch_res)
                    break
                except Exception as e:
                    logger.error(f"Model {self.model_name} batch failed: {e}")
                    if retry == self.max_retries - 1:
                        if self.raise_exception:
                            raise
                    else:
                        await asyncio.sleep(retry + 1)
        return results

    def get_embedding_sync(self, input_text: str) -> list[float]:
        """Synchronous get embedding for a single text with retry logic."""
        for i in range(self.max_retries):
            try:
                result = self._get_embeddings_sync([input_text])
                return result[0]
            except Exception as exc:
                logger.error(f"Model {self.model_name} failed: {exc}")
                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise
                    return []
                time.sleep(i + 1)
        return []

    def get_embeddings_sync(self, input_text: list[str]) -> list[list[float]]:
        """Synchronous get embeddings with automatic batching and retry logic."""
        results = []
        for i in range(0, len(input_text), self.max_batch_size):
            batch = input_text[i : i + self.max_batch_size]
            # Process each batch with retry logic
            for retry in range(self.max_retries):
                try:
                    batch_res = self._get_embeddings_sync(batch)
                    if batch_res:
                        results.extend(batch_res)
                    break
                except Exception as exc:
                    logger.error(f"Model {self.model_name} batch failed: {exc}")
                    if retry == self.max_retries - 1:
                        if self.raise_exception:
                            raise
                    else:
                        time.sleep(retry + 1)
        return results

    async def get_node_embedding(self, node: VectorNode) -> VectorNode:
        """Async generate and populate vector field for a single VectorNode object."""
        node.vector = await self.get_embedding(node.content)
        return node

    async def get_node_embeddings(self, nodes: list[VectorNode]) -> list[VectorNode]:
        """Async generate and populate vector fields for a batch of VectorNode objects."""
        contents = [node.content for node in nodes]
        embeddings: list[list[float]] = await self.get_embeddings(contents)

        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.vector = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(nodes)} nodes")
        return nodes

    def get_node_embedding_sync(self, node: VectorNode) -> VectorNode:
        """Synchronously generate and populate vector field for a single VectorNode object."""
        node.vector = self.get_embedding_sync(node.content)
        return node

    def get_node_embeddings_sync(self, nodes: list[VectorNode]) -> list[VectorNode]:
        """Synchronously generate and populate vector fields for a batch of VectorNode objects."""
        contents = [node.content for node in nodes]
        embeddings: list[list[float]] = self.get_embeddings_sync(contents)

        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.vector = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(nodes)} nodes")
        return nodes

    def close_sync(self):
        """Synchronously release resources and close connections."""

    async def close(self):
        """Asynchronously release resources and close connections."""
