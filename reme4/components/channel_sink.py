"""``ChannelSink`` — push ``notifications/claude/channel`` frames to a bound MCP session.

Server side of the Claude Code "channel" mechanism: once a
``ServerSession`` is bound (via ``ClaimChannelStep``), ``emit`` forwards
tagged events to that session, which Claude Code renders as
``<channel source="<server-name>" key1="v1" ...>content</channel>`` and
reacts on per the server's ``instructions``.

Lossy by design:

* Not bound → no-op (no one elected a recipient yet).
* ``send_message`` raises → log warning, swallow (failed notifications
  must not surface as ingest failures). Notifications aren't acked.

We use ``ServerSession.send_message`` (low-level raw frame) instead of
``send_notification`` because the latter validates against a closed
``ServerNotification`` RootModel union that does not include
``notifications/claude/channel`` — Pydantic rejects custom methods.

Meta keys are filtered to ``[A-Za-z0-9_]+``: Claude Code silently drops
keys with hyphens / other chars when projecting onto ``<channel>`` attrs.
"""

import re
from typing import TYPE_CHECKING

from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from ..utils import get_logger

if TYPE_CHECKING:
    from mcp.server.session import ServerSession


_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")
_CHANNEL_METHOD = "notifications/claude/channel"


class ChannelSink:
    """Hold a bound MCP ``ServerSession`` and forward channel events to it."""

    def __init__(self) -> None:
        self._session: "ServerSession | None" = None
        self._logger = get_logger()

    def bind(self, session: "ServerSession") -> None:
        """Set ``session`` as the recipient for subsequent ``emit`` calls (last-claim-wins)."""
        self._session = session

    def unbind(self) -> None:
        """Drop the bound session; future ``emit`` calls become no-ops until rebind."""
        self._session = None

    async def emit(self, content: str, meta: dict[str, str] | None = None) -> None:
        """Send one channel notification; no-op if unbound, log+swallow on transport failure."""
        session = self._session
        if session is None:
            return

        clean_meta = {k: str(v) for k, v in (meta or {}).items() if _IDENT_RE.match(k)}
        message = SessionMessage(
            JSONRPCMessage(
                JSONRPCNotification(
                    jsonrpc="2.0",
                    method=_CHANNEL_METHOD,
                    params={"content": content, "meta": clean_meta},
                ),
            ),
        )

        try:
            await session.send_message(message)
        except Exception as exc:
            self._logger.warning(f"ChannelSink: send_message failed ({type(exc).__name__}: {exc})")
