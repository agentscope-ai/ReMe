"""Long-running DingTalk Stream bridge for the cookbook application."""

import asyncio
import contextlib
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

from ...base_step import BaseStep
from ....components import R
from ....enumeration import ChunkEnum

_VISIBLE_CHUNKS = {ChunkEnum.THINK, ChunkEnum.TOOL_CALL, ChunkEnum.TOOL_RESULT, ChunkEnum.CONTENT, ChunkEnum.ERROR}
_CODE_CHUNKS = {ChunkEnum.TOOL_CALL, ChunkEnum.TOOL_RESULT}
_BLOCK_CHAR_LIMIT = 100


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


def _payload_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return text.replace("```", "'''")


def _session_ref(key: str) -> str:
    """Return a stable log correlation id without exposing DingTalk identifiers."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class _Block:
    chunk_type: ChunkEnum
    key: str
    name: str = ""
    parts: list[Any] = field(default_factory=list)
    chars: int = 0
    truncated: bool = False


class _CardRenderer:
    """Render append-only, size-limited DingTalk blocks."""

    def __init__(self):
        self.started_at = time.monotonic()
        self.active: _Block | None = None
        self.has_output = False
        self.error = False

    def feed(self, chunk) -> str:
        """Buffer one block and return completed Markdown blocks."""
        if chunk.chunk_type not in _VISIBLE_CHUNKS:
            return ""
        if chunk.chunk_type == ChunkEnum.ERROR:
            self.error = True

        key = chunk.block_id or chunk.tool_call_id or chunk.chunk_type.value
        current = (chunk.chunk_type, key)
        if not chunk.chunk:
            return self._complete_active() if self._active_key() == current else ""

        delta = ""
        if self._active_key() != current:
            delta += self._complete_active()
            self.active = _Block(chunk.chunk_type, key, chunk.tool_call_name or "")
        assert self.active is not None
        self.has_output = True
        if chunk.chunk_type in _CODE_CHUNKS:
            self.active.parts.append(chunk.chunk)
            return delta

        if self.active.chars == 0:
            delta += self._text_title(chunk.chunk_type)
        text = _payload_text(chunk.chunk)
        if chunk.chunk_type == ChunkEnum.CONTENT:
            self.active.chars += len(text)
            return delta + text
        remaining = _BLOCK_CHAR_LIMIT - self.active.chars
        delta += text[:remaining]
        self.active.chars += min(len(text), remaining)
        if len(text) > remaining and not self.active.truncated:
            delta += "..."
            self.active.truncated = True
        return delta

    def fail(self, message: str) -> str:
        """Return an error block for an unexpected local failure."""
        delta = self._complete_active()
        self.error = True
        self.has_output = True
        text = _payload_text(message)
        body = text if len(text) <= _BLOCK_CHAR_LIMIT else f"{text[:_BLOCK_CHAR_LIMIT]}..."
        return f"{delta}### ⚠️ Error\n\n{body}\n\n"

    def _active_key(self) -> tuple[ChunkEnum, str] | None:
        if self.active is None:
            return None
        return self.active.chunk_type, self.active.key

    def _complete_active(self) -> str:
        if self.active is None:
            return ""
        block, self.active = self.active, None
        if block.chunk_type not in _CODE_CHUNKS:
            return "\n\n"
        if block.chunk_type == ChunkEnum.TOOL_CALL and len(block.parts) == 1 and self._is_call_metadata(block.parts[0]):
            return ""

        body = self._block_body(block)
        body = body if len(body) <= _BLOCK_CHAR_LIMIT else f"{body[:_BLOCK_CHAR_LIMIT]}..."
        if block.chunk_type == ChunkEnum.TOOL_CALL:
            name = block.name or self._metadata_name(block) or "tool"
            safe_name = name.replace("`", "'")
            title = f"### 🔧 Tool Call · `{safe_name}`"
        else:
            title = "### 📦 Tool Result"

        body = "\n".join(f"> {self._visible_indent(line)}" for line in body.splitlines())
        return f"{title}\n\n{body}\n\n"

    @staticmethod
    def _text_title(chunk_type: ChunkEnum) -> str:
        if chunk_type == ChunkEnum.THINK:
            return "### 🧠 Think\n\n"
        if chunk_type == ChunkEnum.ERROR:
            return "### ⚠️ Error\n\n"
        return "### 💬 Content\n\n"

    def _block_body(self, block: _Block) -> str:
        if block.chunk_type not in _CODE_CHUNKS:
            return "".join(_payload_text(part) for part in block.parts)
        payload, is_json = self._tool_payload(block)
        if is_json:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return "".join(_payload_text(part) for part in block.parts)

    @staticmethod
    def _visible_indent(line: str) -> str:
        """Keep JSON indentation after DingTalk collapses ordinary spaces."""
        spaces = len(line) - len(line.lstrip(" "))
        return "　" * (spaces // 2) + line[spaces:]

    def _tool_payload(self, block: _Block) -> tuple[Any, bool]:
        parts = block.parts
        if block.chunk_type == ChunkEnum.TOOL_CALL and len(parts) > 1 and self._is_call_metadata(parts[0]):
            parts = parts[1:]
        if len(parts) == 1 and not isinstance(parts[0], str):
            return self._expand_nested_json(parts[0]), True
        if parts and all(isinstance(part, str) for part in parts):
            try:
                value = json.loads("".join(parts))
            except json.JSONDecodeError:
                pass
            else:
                return self._expand_nested_json(value), True
        return None, False

    def _metadata_name(self, block: _Block) -> str:
        if not block.parts or not self._is_call_metadata(block.parts[0]):
            return ""
        return json.loads(block.parts[0]).get("name") or ""

    @staticmethod
    def _is_call_metadata(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict) and bool(parsed) and set(parsed) <= {"id", "name"}

    @classmethod
    def _expand_nested_json(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._expand_nested_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._expand_nested_json(item) for item in value]
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                return cls._expand_nested_json(json.loads(value))
            except json.JSONDecodeError:
                pass
        return value

    def finish(self) -> str:
        """Close the active block and append elapsed time."""
        delta = self._complete_active()
        if not delta and self.has_output:
            delta = "\n\n"
        return f"{delta}{time.monotonic() - self.started_at:.1f}s"


@R.register("dingtalk_wait_step")
class DingTalkWaitStep(BaseStep):
    """Receive DingTalk messages and reply through Markdown or streaming AI cards."""

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        robot_code: str = "",
        stream: bool = False,
        card_update_interval: float = 1.0,
        worker_count: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_key = app_key
        self.app_secret = app_secret
        self.robot_code = robot_code
        self.stream = stream
        self.card_update_interval = max(0.05, card_update_interval)
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
            f"[{self.name}] starting DingTalk Stream bridge "
            f"workers={self.worker_count} reply_mode={'stream' if self.stream else 'non-stream'} "
            f"card_update_interval={self.card_update_interval:.2f}s",
        )
        workers = [
            asyncio.create_task(self._worker(queue, locks, sessions, handler, dingtalk_stream))
            for _ in range(self.worker_count)
        ]
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

    async def _worker(self, queue, locks, sessions, handler, dingtalk_stream) -> None:
        while True:
            message = await queue.get()
            try:
                key = _session_key(message)
                async with locks.setdefault(key, asyncio.Lock()):
                    await self._handle_message(message, key, sessions, handler, dingtalk_stream)
            except Exception as exc:  # A bad message must not disconnect the Stream client.
                self.logger.exception(f"Failed to handle DingTalk message: {exc}")
                await asyncio.to_thread(handler.reply_text, f"处理失败：{exc}", message)
            finally:
                queue.task_done()

    async def _handle_message(self, message, key, sessions, handler, dingtalk_stream) -> None:
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
        if text == "/clear":
            cleared = sessions.pop(key, None) is not None
            self.logger.info(f"[{self.name}] cleared DingTalk session session={session_ref} existed={cleared}")
            await asyncio.to_thread(
                handler.reply_text,
                "✅ Conversation cleared. The next message will start a new session.",
                message,
            )
            return

        resumed = key in sessions
        self.logger.info(
            f"[{self.name}] received DingTalk text session={session_ref} chars={len(text)} resume={resumed}",
        )
        kwargs = {"resume": sessions[key]} if key in sessions else {}
        if not self.stream:
            await self._handle_non_stream_reply(message, key, sessions, handler, text, kwargs, session_ref)
            return
        await self._handle_stream_reply(message, key, sessions, handler, dingtalk_stream, text, kwargs, session_ref)

    async def _handle_non_stream_reply(self, message, key, sessions, handler, text, kwargs, session_ref) -> None:
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
                f"mode=non-stream chars={len(answer)} elapsed={time.monotonic() - started_at:.2f}s",
            )
        except Exception:
            self.logger.warning(
                f"[{self.name}] DingTalk reply failed session={session_ref} "
                f"mode=non-stream elapsed={time.monotonic() - started_at:.2f}s",
            )
            raise

    async def _handle_stream_reply(
        self,
        message,
        key,
        sessions,
        handler,
        dingtalk_stream,
        text,
        kwargs,
        session_ref,
    ) -> None:
        """Stream Agent chunks into one DingTalk AI card."""
        renderer = _CardRenderer()
        card = dingtalk_stream.AIMarkdownCardInstance(handler.dingtalk_client, message)
        card_id = await card.async_create_and_send_card(
            card.card_template_id,
            card.get_card_data(flow_status=dingtalk_stream.AICardStatus.PROCESSING),
            at_sender=True,
        )
        if not card_id:
            raise RuntimeError("创建钉钉 AI 卡片失败")
        self.logger.debug(f"[{self.name}] started DingTalk AI card session={session_ref}")

        last_update = 0.0
        pending = ""
        chunk_count = 0
        update_count = 0
        streamed_chars = 0
        finalizing = False
        try:
            async for chunk in self.agent_wrapper.reply_stream(text, **kwargs):
                chunk_count += 1
                if chunk.session_id:
                    sessions[key] = chunk.session_id
                pending += renderer.feed(chunk)
                if pending and time.monotonic() - last_update >= self.card_update_interval:
                    delta = pending
                    await card.async_streaming(
                        card_id,
                        "msgContent",
                        delta,
                        append=True,
                        finished=False,
                        failed=False,
                    )
                    pending = ""
                    streamed_chars += len(delta)
                    update_count += 1
                    last_update = time.monotonic()

            pending += renderer.finish()
            finalizing = True
            await card.async_streaming(
                card_id,
                "msgContent",
                pending,
                append=True,
                finished=True,
                failed=renderer.error,
            )
            streamed_chars += len(pending)
            pending = ""
            log = self.logger.warning if renderer.error else self.logger.info
            log(
                f"[{self.name}] completed DingTalk reply session={session_ref} success={not renderer.error} "
                f"mode=stream chunks={chunk_count} card_updates={update_count} card_chars={streamed_chars} "
                f"elapsed={time.monotonic() - renderer.started_at:.2f}s",
            )
        except Exception:
            if not finalizing:
                if not renderer.error:
                    pending += renderer.fail("Agent 执行失败")
                pending += renderer.finish()
            failure_delta, pending = pending, ""
            with contextlib.suppress(Exception):
                await card.async_streaming(
                    card_id,
                    "msgContent",
                    failure_delta,
                    append=True,
                    finished=True,
                    failed=True,
                )
            streamed_chars += len(failure_delta)
            self.logger.warning(
                f"[{self.name}] DingTalk reply failed session={session_ref} "
                f"mode=stream chunks={chunk_count} card_updates={update_count} card_chars={streamed_chars} "
                f"elapsed={time.monotonic() - renderer.started_at:.2f}s",
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
