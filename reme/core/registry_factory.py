"""Module providing a registry class for managing class-to-name mappings via decorators."""

import inspect
from typing import Callable, TypeVar

from .base_dict import BaseDict
from .utils import singleton

T = TypeVar("T")


class Registry(BaseDict):
    """A registry container that uses decorators to map and store class references."""

    def register(self, name: str | type = "") -> Callable[[type[T]], type[T]] | type[T]:
        """Return a decorator that registers a class under a specific name in the registry."""
        if inspect.isclass(name):
            self[name.__name__] = name
            return name

        else:

            def decorator(cls):
                key: str = name if isinstance(name, str) and name else cls.__name__
                self[key] = cls
                return cls

            return decorator


@singleton
class RegistryFactory:
    """A factory class for creating registries."""

    def __init__(self):
        self.llms = Registry()
        self.embedding_models = Registry()
        self.vector_stores = Registry()
        self.file_stores = Registry()
        self.ops = Registry()
        self.flows = Registry()
        self.services = Registry()
        self.token_counters = Registry()
        self.file_watchers = Registry()


R = RegistryFactory()
