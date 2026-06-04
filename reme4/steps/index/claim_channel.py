"""``claim_channel_step`` — let an MCP client elect itself as the ``<channel>`` recipient.

The single bind path for every transport (stdio, sse, streamable-http):
this step uses ``fastmcp.server.dependencies.get_context()`` to grab the
current request's ``ServerSession`` and ``ChannelSink.bind`` it.

Semantics:

* **Last-claim-wins.** A second client calling ``claim_channel`` silently
  replaces the previous binding; the prior leader stops receiving events.
* **Lossy on leader loss.** If the bound session goes away, the next
  ``send_message`` raises and ``ChannelSink`` swallows it as a warning.
  Events drop until another client claims.
* **stdio = trivially the one client.** Under stdio there is exactly one
  session ever; calling ``claim_channel`` once after init binds it for
  the rest of the server's life. Until then, channel events drop.
"""

from ..base_step import BaseStep
from ...components import R


@R.register("claim_channel_step")
class ClaimChannelStep(BaseStep):
    """Bind the current MCP session as the ``<channel>`` recipient."""

    async def execute(self):
        assert self.context is not None

        try:
            from fastmcp.server.dependencies import get_context

            ctx = get_context()
            session = ctx.session
            assert session is not None, "FastMCP context has no ServerSession"
            assert self.app_context is not None, "claim_channel requires an application context"
            sink = self.app_context.metadata.get("channel_sink")
            assert sink is not None, "channel_sink not configured on application context metadata"
        except Exception as e:
            self.context.response.answer = {"claimed": False, "reason": f"{type(e).__name__}: {e}"}
            self.context.response.metadata["claimed"] = False
            return self.context.response

        sink.bind(session)
        session_id = ctx.session_id or "<unknown>"
        self.logger.info(f"[claim_channel] channel bound to session={session_id}")
        self.context.response.answer = {
            "claimed": True,
            "session_id": session_id,
            "note": (
                "this session now receives <channel source='reme'> notifications. "
                "last-claim-wins: another call to claim_channel takes over."
            ),
        }
        self.context.response.metadata["claimed"] = True
        return self.context.response
