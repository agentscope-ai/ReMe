"""Timer utilities for measuring execution time.

This module provides a Timer class and a timer decorator for measuring execution time
of code blocks and functions. Supports both synchronous and asynchronous operations.
"""

import functools
import inspect
import time
from typing import Optional, TypeVar, Callable, Union, Awaitable

from loguru import logger

T = TypeVar("T")


class Timer:
    """A context manager and timer utility for measuring execution time.

    Supports both synchronous and asynchronous context managers.

    Usage as synchronous context manager:
        with Timer("my_operation"):
            # your code here

    Usage as asynchronous context manager:
        async with Timer("my_operation"):
            # your code here
            await some_async_function()

    Usage with decorator:
        @timer("my_function")
        def my_function():
            # your code here

        @timer("my_async_function")
        async def my_async_function():
            # your async code here
            await some_async_function()
    """

    def __init__(self, name: str, use_ms: bool = False, stack_level: int = 2):
        """Initialize a Timer instance.

        Args:
            name: Name of the timer, used in log messages
            use_ms: If True, display time in milliseconds, otherwise in seconds
            stack_level: Stack level for loguru logger (default 2)
        """
        self.name: str = name
        self.use_ms: bool = use_ms
        self.stack_level: int = stack_level

        self.time_start: float = 0.0
        self.time_cost: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.time_start = time.time()
        logger.info(f"========== timer.{self.name} start ==========", stacklevel=self.stack_level)
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Stop the timer and log the elapsed time.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise
        """
        time_end = time.time()
        self.time_cost = time_end - self.time_start
        if self.use_ms:
            time_str = f"{self.time_cost * 1000:.2f}ms"
        else:
            time_str = f"{self.time_cost:.3f}s"

        logger.info(f"========== timer.{self.name} end, time_cost={time_str} ==========", stacklevel=self.stack_level)

    async def __aenter__(self) -> "Timer":
        """Start the timer (async version)."""
        self.time_start = time.time()
        logger.info(f"========== timer.{self.name} start ==========", stacklevel=self.stack_level)
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Stop the timer and log the elapsed time (async version).

        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise
        """
        time_end = time.time()
        self.time_cost = time_end - self.time_start
        if self.use_ms:
            time_str = f"{self.time_cost * 1000:.2f}ms"
        else:
            time_str = f"{self.time_cost:.3f}s"

        logger.info(f"========== timer.{self.name} end, time_cost={time_str} ==========", stacklevel=self.stack_level)


def timer(name: Optional[str] = None, use_ms: bool = False, stack_level: int = 2) -> Callable[
    [Callable[..., Union[T, Awaitable[T]]]],
    Callable[..., Union[T, Awaitable[T]]],
]:
    """A decorator factory for timing function execution.

    Automatically detects if the decorated function is async and handles it accordingly.
    Supports both synchronous and asynchronous functions.

    Args:
        name: Optional name for the timer. If None, uses the function's name
        use_ms: If True, display time in milliseconds, otherwise in seconds
        stack_level: Stack level for loguru logger (default 2, +1 is added for the decorator wrapper)

    Returns:
        A decorator function that can be applied to any function (sync or async)

    Usage:
        @timer("my_function")
        def my_function():
            pass

        @timer("my_async_function")
        async def my_async_function():
            await some_async_function()

        @timer(use_ms=True)
        def another_function():
            pass
    """

    def decorator(func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Union[T, Awaitable[T]]]:
        if inspect.iscoroutinefunction(func):
            # Async function
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                async with Timer(name=name or func.__name__, use_ms=use_ms, stack_level=stack_level + 1):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            # Sync function
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                with Timer(name=name or func.__name__, use_ms=use_ms, stack_level=stack_level + 1):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
