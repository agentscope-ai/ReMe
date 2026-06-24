"""Dream bucket enumeration module."""

from enum import Enum


class DreamBucketEnum(str, Enum):
    """Enumeration of digest memory buckets used by dream integration."""

    PROCEDURE = "procedure"
    PERSONAL = "personal"
    WIKI = "wiki"
