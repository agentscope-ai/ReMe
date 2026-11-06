"""Inspect an agent session without waiting behind generation."""

from ...common.session import _SessionStep
from ....components import R


@R.register("session_status_step")
class SessionStatusStep(_SessionStep):
    """Inspect session activity without blocking behind generation."""

    operation = "get_session_status"
