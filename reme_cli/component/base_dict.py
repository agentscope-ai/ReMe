"""Module providing a dictionary subclass with attribute-style access and pickling support."""

from typing import Generic, TypeVar

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class BaseDict(dict, Generic[_KT, _VT]):
    """A dictionary subclass that enables accessing and modifying keys as attributes."""

    def __getattr__(self, name: str) -> _VT:
        """Retrieve a dictionary item as an attribute."""
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

    def __setattr__(self, name: str, value: _VT) -> None:
        """Assign a value to a dictionary item using attribute syntax."""
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Remove a dictionary item using attribute syntax."""
        try:
            # Delete item from dict via key
            del self[name]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

    def __getstate__(self) -> dict:
        """Return the dictionary representation for pickling."""
        return dict(self)

    def __setstate__(self, state: dict) -> None:
        """Restore the dictionary state from a pickled object."""
        self.update(state)

    def __reduce__(self):
        """Define the reconstruction logic for pickling processes."""
        return self.__class__, (), self.__getstate__()
