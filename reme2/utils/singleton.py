"""Singleton pattern implementation using a class decorator.

Provides a thread-safe singleton decorator that ensures only one instance
of a decorated class exists throughout the application lifecycle.
"""

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    """A class decorator that ensures only one instance of a class exists.

    Thread-safe implementation using a lock to prevent race conditions
    during instance creation.

    Args:
        cls: The class to decorate with singleton behavior.

    Returns:
        A wrapper function that returns the single instance.
    """
    _instance: dict[type[T], T] = {}
    _lock = threading.Lock()

    def _singleton(*args: Any, **kwargs: Any) -> T:
        """Return the existing instance or create a new one if it doesn't exist.

        Args:
            *args: Positional arguments passed to the class constructor.
            **kwargs: Keyword arguments passed to the class constructor.

        Returns:
            The single instance of the decorated class.
        """
        with _lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
