"""Base embedding model interface for ReMe.

This module provides an abstract base class for all embedding model implementations in ReMe.
It defines the standard interface for generating embeddings from text or VectorNode objects,
with built-in retry logic, error handling, and batch processing support.
"""

import asyncio
from abc import ABC
from typing import List

from loguru import logger

from ..schema import VectorNode


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for all embedding model implementations.

    This class defines the standard interface for interacting with embedding models,
    supporting both synchronous and asynchronous operations, batch processing, and
    automatic retry logic for robust operation in production environments.

    Attributes:
        model_name: Name of the embedding model to use
        dimensions: Dimension of the embedding vectors
        max_batch_size: Maximum number of texts to embed in a single batch
        max_retries: Maximum number of retries on failure
        raise_exception: Whether to raise exception on final failure
        kwargs: Additional parameters passed to the embedding API
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
        """
        Initialize the BaseEmbeddingModel.

        Args:
            model_name: Name of the embedding model to use (default: "")
            dimensions: Dimension of the embedding vectors (default: 1024)
            max_batch_size: Maximum number of texts to embed in a single batch (default: 10)
            max_retries: Maximum number of retries on failure (default: 3)
            raise_exception: Whether to raise exception on final failure (default: True)
            **kwargs: Additional parameters passed to the embedding API
        """
        self.model_name: str = model_name
        self.dimensions: int = dimensions
        self.max_batch_size: int = max_batch_size
        self.max_retries: int = max_retries
        self.raise_exception: bool = raise_exception
        self.kwargs: dict = kwargs

    async def _get_embeddings(self, input_text: str | List[str]) -> List[List[float]] | List[float]:
        """
        Internal async method to get embeddings from the model.

        This is an abstract method that must be implemented by subclasses.
        It should call the actual embedding API and return the embedding vectors.

        Args:
            input_text: Single text string or list of text strings to embed

        Returns:
            Single embedding vector (List[float]) if input is str,
            or list of embedding vectors (List[List[float]]) if input is List[str]

        Raises:
            Exception: Any exception raised by the underlying embedding API
        """
        raise NotImplementedError

    def _get_embeddings_sync(self, input_text: str | List[str]) -> List[List[float]] | List[float]:
        """
        Internal sync method to get embeddings from the model.

        This is an abstract method that must be implemented by subclasses.
        It should call the actual embedding API and return the embedding vectors.

        Args:
            input_text: Single text string or list of text strings to embed

        Returns:
            Single embedding vector (List[float]) if input is str,
            or list of embedding vectors (List[List[float]]) if input is List[str]

        Raises:
            Exception: Any exception raised by the underlying embedding API
        """
        raise NotImplementedError

    async def get_embeddings(self, input_text: str | List[str]) -> List[List[float]] | List[float]:
        """
        Async get embeddings with automatic retry logic and error handling.

        This method wraps _get_embeddings with retry logic. It will
        automatically retry on failures up to max_retries times.
        For lists larger than max_batch_size, it automatically batches the requests.

        Args:
            input_text: Single text string or list of text strings to embed

        Returns:
            Single embedding vector (List[float]) if input is str,
            list of embedding vectors (List[List[float]]) if input is List[str],
            or empty list [] if all retries fail and raise_exception is False

        Raises:
            Exception: Any exception from the embedding API if raise_exception
                      is True and all retries are exhausted
        """
        # Handle single string - no batching needed
        if isinstance(input_text, str):
            for i in range(self.max_retries):
                try:
                    return await self._get_embeddings(input_text)

                except Exception as e:
                    logger.exception(f"embedding model name={self.model_name} encounter error with e={e.args}")

                    if i == self.max_retries - 1:
                        if self.raise_exception:
                            raise e
                        return []

                    # Exponential backoff between retries
                    await asyncio.sleep(i + 1)

            return []
        
        # Handle list - batch if necessary
        elif isinstance(input_text, list):
            # If list is within max_batch_size, process directly
            if len(input_text) <= self.max_batch_size:
                for i in range(self.max_retries):
                    try:
                        return await self._get_embeddings(input_text)

                    except Exception as e:
                        logger.exception(f"embedding model name={self.model_name} encounter error with e={e.args}")

                        if i == self.max_retries - 1:
                            if self.raise_exception:
                                raise e
                            return []

                        # Exponential backoff between retries
                        await asyncio.sleep(i + 1)

                return []
            
            # If list exceeds max_batch_size, batch the requests sequentially
            else:
                embeddings = []
                for i in range(0, len(input_text), self.max_batch_size):
                    batch = input_text[i: i + self.max_batch_size]
                    batch_embeddings = await self.get_embeddings(batch)
                    if batch_embeddings:
                        embeddings.extend(batch_embeddings)
                
                return embeddings
        
        else:
            raise TypeError(f"unsupported type={type(input_text)}")

    def get_embeddings_sync(self, input_text: str | List[str]) -> List[List[float]] | List[float]:
        """
        Get embeddings with automatic retry logic and error handling.

        This method wraps _get_embeddings_sync with retry logic. It will
        automatically retry on failures up to max_retries times.
        For lists larger than max_batch_size, it automatically batches the requests.

        Args:
            input_text: Single text string or list of text strings to embed

        Returns:
            Single embedding vector (List[float]) if input is str,
            list of embedding vectors (List[List[float]]) if input is List[str],
            or empty list [] if all retries fail and raise_exception is False

        Raises:
            Exception: Any exception from the embedding API if raise_exception
                      is True and all retries are exhausted
        """
        # Handle single string - no batching needed
        if isinstance(input_text, str):
            for i in range(self.max_retries):
                try:
                    return self._get_embeddings_sync(input_text)

                except Exception as e:
                    logger.exception(f"embedding model name={self.model_name} encounter error with e={e.args}")
                    
                    if i == self.max_retries - 1:
                        if self.raise_exception:
                            raise e
                        return []

            return []
        
        # Handle list - batch if necessary
        elif isinstance(input_text, list):
            # If list is within max_batch_size, process directly
            if len(input_text) <= self.max_batch_size:
                for i in range(self.max_retries):
                    try:
                        return self._get_embeddings_sync(input_text)

                    except Exception as e:
                        logger.exception(f"embedding model name={self.model_name} encounter error with e={e.args}")
                        
                        if i == self.max_retries - 1:
                            if self.raise_exception:
                                raise e
                            return []

                return []
            
            # If list exceeds max_batch_size, batch the requests sequentially
            else:
                embeddings = []
                for i in range(0, len(input_text), self.max_batch_size):
                    batch = input_text[i: i + self.max_batch_size]
                    batch_embeddings = self.get_embeddings_sync(batch)
                    if batch_embeddings:
                        embeddings.extend(batch_embeddings)
                
                return embeddings
        
        else:
            raise TypeError(f"unsupported type={type(input_text)}")

    async def get_node_embeddings(self, nodes: VectorNode | List[VectorNode]) -> VectorNode | List[VectorNode]:
        """
        Async get embeddings for VectorNode(s) and populate their vector field.

        This method handles both single VectorNode and lists of VectorNodes.
        For lists, it automatically batches requests according to max_batch_size
        and processes batches concurrently for optimal performance.

        Args:
            nodes: Single VectorNode or list of VectorNodes to generate embeddings for

        Returns:
            The same node(s) with vector field populated

        Raises:
            TypeError: If nodes is not a VectorNode or list of VectorNodes
            Exception: Any exception from get_embeddings if raise_exception is True
        """
        if isinstance(nodes, VectorNode):
            # Handle single node
            nodes.vector = await self.get_embeddings(nodes.content)
            return nodes

        elif isinstance(nodes, list):
            # Process nodes in batches sequentially
            embeddings = []
            for i in range(0, len(nodes), self.max_batch_size):
                batch_nodes = nodes[i: i + self.max_batch_size]
                batch_content = [node.content for node in batch_nodes]
                batch_embeddings = await self.get_embeddings(batch_content)
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)

            # Validate that we got the expected number of embeddings
            if len(embeddings) != len(nodes):
                logger.warning(f"embeddings.size={len(embeddings)} <> nodes.size={len(nodes)}")
            else:
                # Assign embeddings to corresponding nodes
                for node, embedding in zip(nodes, embeddings):
                    node.vector = embedding

            return nodes

        else:
            raise TypeError(f"unsupported type={type(nodes)}")

    def get_node_embeddings_sync(self, nodes: VectorNode | List[VectorNode]) -> VectorNode | List[VectorNode]:
        """
        Sync get embeddings for VectorNode(s) and populate their vector field.

        This method handles both single VectorNode and lists of VectorNodes.
        For lists, it automatically batches requests according to max_batch_size
        to optimize API calls and respect rate limits.

        Args:
            nodes: Single VectorNode or list of VectorNodes to generate embeddings for

        Returns:
            The same node(s) with vector field populated

        Raises:
            TypeError: If nodes is not a VectorNode or list of VectorNodes
            Exception: Any exception from get_embeddings_sync if raise_exception is True
        """
        if isinstance(nodes, VectorNode):
            # Handle single node
            nodes.vector = self.get_embeddings_sync(nodes.content)
            return nodes

        elif isinstance(nodes, list):
            # Process nodes in batches to respect max_batch_size limits
            embeddings = []
            for i in range(0, len(nodes), self.max_batch_size):
                batch_nodes = nodes[i : i + self.max_batch_size]
                batch_content = [node.content for node in batch_nodes]
                batch_embeddings = self.get_embeddings_sync(input_text=batch_content)
                
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)

            # Validate that we got the expected number of embeddings
            if len(embeddings) != len(nodes):
                logger.warning(f"embeddings.size={len(embeddings)} <> nodes.size={len(nodes)}")
            else:
                # Assign embeddings to corresponding nodes
                for node, embedding in zip(nodes, embeddings):
                    node.vector = embedding
            
            return nodes

        else:
            raise TypeError(f"unsupported type={type(nodes)}")

    def close_sync(self):
        """
        Close the client connection or clean up resources synchronously.

        This method should be called when the embedding model instance is no longer needed
        to properly release any held resources (e.g., HTTP connections, file handles).
        Subclasses should override this method if they need to perform cleanup.
        """

    async def close(self):
        """
        Asynchronously close the client connection or clean up resources.

        This async method should be called when the embedding model instance is no longer needed
        to properly release any held resources (e.g., HTTP connections, file handles).
        Subclasses should override this method if they need to perform async cleanup.
        """
