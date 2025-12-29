from typing import Generic, TypeVar

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class BaseContext(dict, Generic[_KT, _VT]):
    """A dict subclass that supports attribute-style access and pickling."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)

    def __reduce__(self):
        return self.__class__, (), self.__getstate__()
