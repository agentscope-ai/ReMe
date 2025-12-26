"""Base class for asynchronous operations.

This module provides the BaseAsyncOp class, which extends BaseOp with
asynchronous execution capabilities. It supports async cache operations,
async task submission and joining, and full async execution lifecycle.
"""

import asyncio
from abc import ABCMeta, abstractmethod
from typing import Any, Callable

from loguru import logger

from .base_op import BaseOp
from ..context import FlowContext, C


class BaseAsyncOp(BaseOp, metaclass=ABCMeta):
    """Base class for asynchronous operations.

    This class extends BaseOp to provide asynchronous execution capabilities.
    All operations created from this class run in async mode by default.

    Example:
        ```python
        class MyAsyncOp(BaseAsyncOp):
            async def async_execute(self):
                return await some_async_function()

        op = MyAsyncOp()
        result = await op.async_call()
        ```
    """

    def __init__(self, **kwargs):
        """Initialize the async operation.

        Automatically sets async_mode to True if not explicitly set.

        Args:
            **kwargs: Arguments passed to BaseOp.__init__
        """
        kwargs.setdefault("async_mode", True)
        super().__init__(**kwargs)

    def execute(self):
        """Placeholder for synchronous execute method.

        This method is not used in async operations. Subclasses should implement
        `async_execute()` instead. This method exists only to satisfy the abstract
        method requirement from BaseOp.
        """

    async def async_save_load_cache(self, key: str, fn: Callable, **kwargs):
        """Save or load from cache asynchronously.

        If caching is enabled, checks cache first. If not found, executes the
        function (async or sync) and saves the result. Supports both coroutine
        functions and regular functions (executed in thread pool).

        Args:
            key: Cache key for storing/retrieving the result
            fn: Function to execute if cache miss (can be async or sync)
            **kwargs: Additional arguments for cache load operation

        Returns:
            Cached result if available, otherwise result from function execution
        """
        if self.enable_cache:
            result = self.cache.load(key, **kwargs)
            if result is None:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn()
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(C.thread_pool, fn)

                self.cache.save(key, result, expire_hours=self.cache_expire_hours)
            else:
                logger.info(f"load {key} from cache")
        else:
            if asyncio.iscoroutinefunction(fn):
                result = await fn()
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(C.thread_pool, fn)

        return result

    async def async_before_execute(self):
        """Hook method called before async_execute(). Override in subclasses.

        This method is called automatically by `async_call()` before executing
        the main `async_execute()` method. Use this to perform any setup,
        validation, or preprocessing needed before execution.

        Example:
            ```python
            async def async_before_execute(self):
                # Validate inputs
                if not self.context.get("input"):
                    raise ValueError("Input is required")
        ```
        """

    async def async_after_execute(self):
        """Hook method called after async_execute(). Override in subclasses.

        This method is called automatically by `async_call()` after successfully
        executing the main `async_execute()` method. Use this to perform any
        cleanup, post-processing, or result transformation.

        Example:
            ```python
            async def async_after_execute(self):
                # Post-process results
                if self.context.response:
                    self.context.response.answer = self.context.response.answer.upper()
        ```
        """

    @abstractmethod
    async def async_execute(self):
        """Main async execution method. Must be implemented in subclasses.

        Returns:
            Execution result
        """

    async def async_default_execute(self, e: Exception = None, **kwargs):
        """Default async execution method when main execution fails. Override in subclasses.

        This method is called when `async_execute()` fails and `raise_exception`
        is False. It provides a fallback mechanism to return a default result
        instead of raising an exception.

        Args:
            e: The exception that was raised during execution (if any)
            **kwargs: Additional keyword arguments

        Returns:
            Default execution result

        Example:
            ```python
            async def async_default_execute(self, e: Exception = None, **kwargs):
                logger.warning(f"Execution failed: {e}, returning default result")
                return {"status": "error", "message": str(e)}
            ```
        """

    async def async_call(self, context: FlowContext = None, **kwargs) -> Any:
        """Execute the operation asynchronously.

        This method handles the full async execution lifecycle including retries,
        error handling, and context management. It automatically calls
        `async_before_execute()`, `async_execute()`, and `async_after_execute()`
        in sequence.

        Args:
            context: Flow context for this execution. If None, a new context
                will be created.
            **kwargs: Additional context updates to merge into the context

        Returns:
            Execution result from `async_execute()`, context response if result
            is None, or None if both are None

        Raises:
            Exception: If execution fails and `raise_exception` is True and
                `max_retries` is exhausted
        """
        self.context = self.build_context(context, **kwargs)
        with self.timer:
            result = None
            if self.max_retries == 1 and self.raise_exception:
                await self.async_before_execute()
                result = await self.async_execute()
                await self.async_after_execute()

            else:
                for i in range(self.max_retries):
                    try:
                        await self.async_before_execute()
                        result = await self.async_execute()
                        await self.async_after_execute()
                        break

                    except Exception as e:
                        logger.exception(f"op={self.name} async execute failed, error={e.args}")

                        if self.raise_exception and i == self.max_retries - 1:
                            raise e

                        result = await self.async_default_execute(e)

        if result is not None:
            return result

        elif self.context is not None and self.context.response is not None:
            return self.context.response

        else:
            return None

    def submit_async_task(self, fn: Callable, *args, **kwargs):
        """Submit an async task for execution.

        Creates an asyncio task and adds it to the task list for later joining.
        Tasks can be collected using `join_async_task()`.

        Args:
            fn: Coroutine function to execute
            *args: Positional arguments for the coroutine
            **kwargs: Keyword arguments for the coroutine

        Note:
            Only coroutine functions are supported. Non-coroutine functions
            will trigger a warning and be ignored.

        Example:
            ```python
            async def my_task(x):
                return x * 2

            self.submit_async_task(my_task, 5)
            results = await self.join_async_task()
            ```
        """
        loop = asyncio.get_running_loop()
        if asyncio.iscoroutinefunction(fn):
            task = loop.create_task(fn(*args, **kwargs))
            self.task_list.append(task)
        else:
            logger.warning("submit_async_task failed, fn is not a coroutine function!")

    async def join_async_task(self, timeout: float = None, return_exceptions: bool = True):
        """Wait for all submitted async tasks to complete and collect results.

        Collects results from all tasks, handling exceptions and timeouts.
        On timeout or exception, all remaining tasks are cancelled.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)
            return_exceptions: Whether to return exceptions as results

        Returns:
            List of task results (exceptions included if return_exceptions=True)

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            Exception: If any task raises an exception and return_exceptions=False
        """
        result = []

        if not self.task_list:
            return result

        try:
            if timeout is not None:
                gather_task = asyncio.gather(*self.task_list, return_exceptions=return_exceptions)
                task_results = await asyncio.wait_for(gather_task, timeout=timeout)
            else:
                task_results = await asyncio.gather(*self.task_list, return_exceptions=return_exceptions)

            for t_result in task_results:
                if return_exceptions and isinstance(t_result, Exception):
                    logger.opt(exception=t_result).error("Task failed with exception")
                    continue

                if t_result:
                    if isinstance(t_result, list):
                        result.extend(t_result)
                    else:
                        result.append(t_result)

        except asyncio.TimeoutError:
            logger.exception(f"join_async_task timeout after {timeout}s, cancelling {len(self.task_list)} tasks...")
            for task in self.task_list:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*self.task_list, return_exceptions=True)
            self.task_list.clear()
            raise

        except Exception as e:
            logger.exception(f"join_async_task failed with {type(e).__name__}, cancelling remaining tasks...")
            for task in self.task_list:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*self.task_list, return_exceptions=True)
            self.task_list.clear()
            raise

        finally:
            self.task_list.clear()

        return result
