"""Module providing a decorator to implement the Singleton design pattern."""

import threading


def singleton(cls):
    """A class decorator that ensures only one instance of a class exists."""

    # Dictionary to cache the single instance of the class
    _instance = {}
    _lock = threading.Lock()

    def _singleton(*args, **kwargs):
        """Return the existing instance or create a new one if it doesn't exist."""
        with _lock:
            if cls not in _instance:
                # Create and store the instance if it's the first call
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
