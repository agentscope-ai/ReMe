"""Long-running DingTalk Stream bridge for the cookbook application."""

import asyncio
import hashlib
import json
import time
from typing import Any
from urllib.parse import quote_plus

from ...base_step import BaseStep
from ....components import R
from ....components.agent_wrapper import handle_session_command


def _session_key(message: Any) -> str:
    """Return the per-sender, per-conversation Claude session key."""
    parts = (
        message.conversation_type,
        message.conversation_id,
        message.sender_staff_id,
    )
    if not all(parts):
        raise ValueError("DingTalk message requires conversationType, conversationId, and senderStaffId")
    return ":".join(parts)


def _session_ref(key: str) -> str:
    """Return a stable log correlation id without exposing DingTalk identifiers."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@R.register("dingtalk_wait_step")
class DingTalkWaitStep(BaseStep):
    """Receive DingTalk messages and send final Agent responses as Markdown."""

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        robot_code: str = "",
        worker_count: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_key = app_key
        self.app_secret = app_secret
        self.robot_code = robot_code
        self.worker_count = max(1, worker_count)

    async def execute(self):
        assert self.context is not None
        if self.context.stop_event is None or self.app_context is None:
            raise RuntimeError("dingtalk_wait_step requires an ApplicationContext and background stop_event")
        if self.agent_wrapper is None:
            raise RuntimeError("dingtalk_wait_step requires an agent_wrapper")
        if not self.app_key or not self.app_secret or not self.robot_code:
            raise RuntimeError("dingtalk_wait_step requires app_key, app_secret, and robot_code")

        import dingtalk_stream  # pylint: disable=import-outside-toplevel

        queue: asyncio.Queue = asyncio.Queue()
        handler = self._make_handler(dingtalk_stream, queue)
        client = dingtalk_stream.DingTalkStreamClient(
            dingtalk_stream.Credential(self.app_key, self.app_secret),
        )
        client.register_callback_handler(dingtalk_stream.ChatbotMessage.TOPIC, handler)
        sessions = self.app_context.metadata.setdefault("dingtalk_agent_sessions", {})
        locks: dict[str, asyncio.Lock] = {}
        self.logger.info(
            f"[{self.name}] starting DingTalk Stream bridge workers={self.worker_count}",
        )
        workers = [asyncio.create_task(self._worker(queue, locks, sessions, handler)) for _ in range(self.worker_count)]
        try:
            await self._run_client(client, self.context.stop_event)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self.logger.info(f"[{self.name}] DingTalk Stream bridge stopped")
        return self.context.response

    @staticmethod
    def _make_handler(dingtalk_stream, queue: asyncio.Queue):
        class QueueHandler(dingtalk_stream.ChatbotHandler):
            """Acknowledge callbacks after placing them on the worker queue."""

            async def process(self, callback):
                """Enqueue one callback and immediately acknowledge it."""
                queue.put_nowait(dingtalk_stream.ChatbotMessage.from_dict(callback.data))
                return dingtalk_stream.AckMessage.STATUS_OK, "OK"

        return QueueHandler()

    async def _worker(self, queue, locks, sessions, handler) -> None:
        while True:
            message = await queue.get()
            try:
                key = _session_key(message)
                async with locks.setdefault(key, asyncio.Lock()):
                    await self._handle_message(message, key, sessions, handler)
            except Exception as exc:  # A bad message must not disconnect the Stream client.
                self.logger.exception(f"Failed to handle DingTalk message: {exc}")
                await asyncio.to_thread(handler.reply_text, f"处理失败：{exc}", message)
            finally:
                queue.task_done()

    async def _handle_message(self, message, key, sessions, handler) -> None:
        session_ref = _session_ref(key)
        self.logger.info(
            f"[{self.name}] handling DingTalk callback session={session_ref} "
            f"conversation_type={message.conversation_type!r} conversation_id={message.conversation_id!r} "
            f"sender_staff_id={message.sender_staff_id!r}",
        )
        if self.robot_code and getattr(message, "robot_code", "") != self.robot_code:
            self.logger.warning(
                f"[{self.name}] rejected DingTalk callback session={session_ref} reason=robot_code_mismatch",
            )
            raise ValueError("DingTalk callback robotCode does not match configured robot_code")
        text = (message.text.content if message.text else "").strip()
        if not text:
            self.logger.info(f"[{self.name}] ignored non-text DingTalk message session={session_ref}")
            await asyncio.to_thread(handler.reply_text, "暂时只支持文本消息。", message)
            return
        command = await handle_session_command(self.agent_wrapper, text, sessions.get(key))
        if command is not None:
            if command.session_id is None:
                sessions.pop(key, None)
            else:
                sessions[key] = command.session_id
            self.logger.info(f"[{self.name}] handled session command session={session_ref} command={text}")
            await asyncio.to_thread(handler.reply_text, command.answer, message)
            return

        resumed = key in sessions
        self.logger.info(
            f"[{self.name}] received DingTalk text session={session_ref} chars={len(text)} resume={resumed}",
        )
        kwargs = {"resume": sessions[key]} if key in sessions else {}
        await self._handle_reply(message, key, sessions, handler, text, kwargs, session_ref)

    async def _handle_reply(self, message, key, sessions, handler, text, kwargs, session_ref) -> None:
        """Wait for the final Agent response and send one DingTalk Markdown reply."""
        started_at = time.monotonic()
        try:
            result = await self.agent_wrapper.reply(text, **kwargs)
            if not isinstance(result, dict):
                raise TypeError("Agent reply must be a dictionary")
            if session_id := result.get("session_id"):
                sessions[key] = session_id

            last_message = result.get("last_message")
            if isinstance(last_message, dict) and last_message.get("is_error"):
                raise RuntimeError("Agent 执行失败")

            answer = result.get("result")
            if not isinstance(answer, str) or not (answer := answer.strip()):
                raise ValueError("Agent 返回了空回复")
            response = await asyncio.to_thread(handler.reply_markdown, "ReMe Agent", answer, message)
            if response is None:
                raise RuntimeError("发送钉钉 Markdown 回复失败")
            self.logger.info(
                f"[{self.name}] completed DingTalk reply session={session_ref} success=True "
                f"chars={len(answer)} elapsed={time.monotonic() - started_at:.2f}s",
            )
        except Exception:
            self.logger.warning(
                f"[{self.name}] DingTalk reply failed session={session_ref} "
                f"elapsed={time.monotonic() - started_at:.2f}s",
            )
            raise

    @staticmethod
    async def _run_client(client, stop_event: asyncio.Event) -> None:
        """Run one cancellable WebSocket connection; BackgroundJob owns retries."""
        import websockets  # pylint: disable=import-outside-toplevel

        client.pre_start()
        connection = await asyncio.to_thread(client.open_connection)
        if not connection:
            raise ConnectionError("DingTalk open connection failed")
        uri = f'{connection["endpoint"]}?ticket={quote_plus(connection["ticket"])}'
        async with websockets.connect(uri) as websocket:
            client.websocket = websocket
            keepalive = asyncio.create_task(client.keepalive(websocket))

            async def close_when_stopped() -> None:
                await stop_event.wait()
                await websocket.close()

            stopper = asyncio.create_task(close_when_stopped())
            try:
                async for raw_message in websocket:
                    if await client.route_message(json.loads(raw_message)) == client.TAG_DISCONNECT:
                        await websocket.close()
            finally:
                for task in (stopper, keepalive):
                    task.cancel()
                await asyncio.gather(stopper, keepalive, return_exceptions=True)
            if not stop_event.is_set():
                raise ConnectionError("DingTalk WebSocket closed")
