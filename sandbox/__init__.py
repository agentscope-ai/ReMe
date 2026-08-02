"""Docker sandbox infrastructure for isolated ReMe benchmark cases.

This package intentionally lives outside :mod:`reme`: benchmark isolation is
infrastructure, not part of ReMe's runtime contract.
"""

from .candidate import ImageCandidate, SourceCandidate, SourceSnapshot
from .docker_sandbox import DockerReMeSandbox, DockerReMeSandboxFactory
from .models import JobResult

__all__ = [
    "DockerReMeSandbox",
    "DockerReMeSandboxFactory",
    "ImageCandidate",
    "JobResult",
    "SourceCandidate",
    "SourceSnapshot",
]
