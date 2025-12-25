from .base_context import BaseContext


class Registry(BaseContext):
    def register(self, name: str = "", add_cls: bool = True):
        def decorator(cls):
            if add_cls:
                key = name or cls.__name__
                self[key] = cls
            return cls

        return decorator
