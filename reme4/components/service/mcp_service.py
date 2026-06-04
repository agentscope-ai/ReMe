"""MCP service: expose jobs as MCP tools.

Channel binding (the `<channel source="reme" kind="vault_change" ...>`
push from background steps to a specific Claude Code window) is uniform
across transports: a single `ChannelSink` lives on
`ApplicationContext.metadata["channel_sink"]`, unbound at startup, and any
client calling the `claim_channel` MCP tool binds itself as the recipient
via `fastmcp.server.dependencies.get_context().session`. Last-claim-wins.

Under stdio (one client per server process) the client should claim once
after init; until then channel events drop silently. Under shared
streamable-http / sse the human picks which window receives events.

``ChannelSink`` is colocated here because it is the runtime mechanism
behind this service's channel feature — pushes ``notifications/claude/channel``
frames to the bound MCP session. Lossy by design: not bound → no-op;
``send_message`` raises → log warning, swallow (failed notifications must
not surface as ingest failures). Uses ``ServerSession.send_message``
(low-level raw frame) instead of ``send_notification`` because the latter
validates against a closed ``ServerNotification`` RootModel union that
does not include ``notifications/claude/channel`` — Pydantic rejects
custom methods. Meta keys are filtered to ``[A-Za-z0-9_]+``: Claude Code
silently drops keys with hyphens / other chars when projecting onto
``<channel>`` attrs.
"""

import re
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from fastmcp.server.server import Transport
from fastmcp.tools import FunctionTool
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from .base_service import BaseService
from ..component_registry import R
from ..job import BaseJob, StreamJob
from ...constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from ...utils import get_logger

if TYPE_CHECKING:
    from mcp.server.session import ServerSession

    from ...application import Application


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


_CHANNEL_INSTRUCTIONS = (
    "Events from the reme channel arrive as\n"
    '  <channel source="reme" kind="vault_change" count="N">\n'
    "    added|modified|deleted: <vault-relative path>\n"
    "    ...\n"
    "  </channel>\n"
    "The vault watcher fires one event per debounced batch of live changes "
    "under daily/, digest/, and resource/ (initial-scan diffs at startup are "
    "intentionally NOT replayed).\n"
    "\n"
    "Events are delivered ONLY to the MCP session that called the "
    "`claim_channel` tool last. Call it once per Claude Code window that "
    "should receive vault-change notifications.\n"
    "\n"
    "When new files appear under daily/ or resource/, treat it as a suggestion "
    "to run `/dream <path>` on each new path -- unless the user is mid-task and "
    "would be interrupted, in which case acknowledge in one line and continue. "
    "For changes under digest/ (which /dream itself writes), just acknowledge; "
    "do not re-dream them. For deletes or modifies elsewhere, just acknowledge."
)


@R.register("mcp")
class MCPService(BaseService):
    """Expose non-stream jobs as MCP tools over stdio, SSE, or streamable-http."""

    def __init__(
        self,
        transport: Transport = "sse",
        host: str = REME_DEFAULT_HOST,
        port: int = REME_DEFAULT_PORT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transport: Transport = transport
        self.host: str = host
        self.port: int = port

    # ----- BaseService contract ------------------------------------------

    def build_service(self, app: "Application") -> None:
        """Construct the FastMCP server and publish an unbound ChannelSink."""
        app.context.metadata["channel_sink"] = ChannelSink()
        self.service = FastMCP(
            name=app.config.app_name,
            instructions=_CHANNEL_INSTRUCTIONS,
            lifespan=self._lifespan(app, self.host, self.port),
        )

    def add_job(self, job: BaseJob) -> None:
        """Register a non-stream job as an MCP tool; StreamJobs are unsupported."""
        if isinstance(job, StreamJob):
            return

        async def execute_tool(**kwargs):
            response = await job(**kwargs)
            return response.answer

        self.service.add_tool(
            FunctionTool(
                name=job.name,
                description=job.description,
                fn=execute_tool,
                parameters=job.parameters or None,
            ),
        )

    def start_service(self, app: "Application") -> None:
        """Run the MCP server; bind host/port only for network transports."""
        transport_kwargs: dict = {}
        if self.transport != "stdio":
            transport_kwargs["host"] = self.host
            transport_kwargs["port"] = self.port
        self.service.run(transport=self.transport, show_banner=False, **transport_kwargs)
