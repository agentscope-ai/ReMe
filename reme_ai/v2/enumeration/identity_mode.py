"""Identity mode enumeration for identity memory operations."""

from enum import Enum


class IdentityMode(str, Enum):
    """Operation modes for identity memory management."""
    NEW = "new"
    GET = "get"
    UPDATE = "update"

