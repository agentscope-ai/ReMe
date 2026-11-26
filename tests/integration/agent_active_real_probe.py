"""Opt-in real-model verification of the same production session contract for all backends.

    python -m tests.integration.agent_active_real_probe --run-real --backend agentscope \
        --env-file .env --model qwen3.8-max --output /tmp/as-active-real.json

All storage is temporary. A local relay delays a real provider response so
fork and append run during generation; no model output is synthesized.
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

from aiohttp import web
import httpx

from reme.core.context import ApplicationContext
from reme.components.agent_wrapper import AsAgentWrapper, CcAgentWrapper, CodexAgentWrapper
from reme.components.codex_app_server import SdkCodexAppServer
from reme.enumeration import ComponentEnum
from reme.utils.env_utils import parse_env_file


def progress(event, **facts):
    """Log progress without headers, credentials or model input."""
    print(json.dumps({"event": event, **facts}), flush=True)


class Relay:
    """Forward actual provider traffic while retaining only synthetic probe inputs."""

    def __init__(self, upstream):
        self.upstream = upstream.rstrip("/")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = []
        self.client = None
        self.base_url = ""

    async def handle(self, request):
        """Delay the first inference response, preserving upstream body bytes."""
        body = await request.read()
        payload = json.loads(body) if body else {}
        inference = request.method == "POST" and "count_tokens" not in request.path
        record = None
        if inference:
            record = {"model": payload.get("model"), "input": payload.get("messages", payload.get("input", []))}
            self.requests.append(record)
            progress("provider_request", index=len(self.requests), model=record["model"])
        ignored = {"host", "content-length", "connection", "accept-encoding"}
        headers = {key: value for key, value in request.headers.items() if key.lower() not in ignored}
        target = self.upstream + str(request.rel_url)
        upstream = await self.client.send(
            self.client.build_request(request.method, target, content=body, headers=headers),
            stream=True,
        )
        try:
            if record is not None:
                record["status"] = upstream.status_code
                progress("provider_response", index=self.requests.index(record) + 1, status=upstream.status_code)
                if record is self.requests[0] and upstream.is_success:
                    self.entered.set()
                    await asyncio.wait_for(self.release.wait(), 60)
            ignored = {"content-length", "transfer-encoding", "connection", "content-encoding"}
            response = web.StreamResponse(
                status=upstream.status_code,
                headers={key: value for key, value in upstream.headers.items() if key.lower() not in ignored},
            )
            await response.prepare(request)
            try:
                async for chunk in upstream.aiter_bytes():
                    await response.write(chunk)
                await response.write_eof()
            except ConnectionResetError:
                # Streaming SDKs may close immediately after their terminal SSE event.
                pass
            return response
        finally:
            await upstream.aclose()

    @asynccontextmanager
    async def serve(self):
        """Own the relay and HTTP client for precisely one probe."""
        async with httpx.AsyncClient(timeout=90) as client:
            self.client = client
            app = web.Application()
            app.router.add_route("*", "/{path:.*}", self.handle)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self.base_url = (
                f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # pylint: disable=protected-access
            )
            try:
                yield self
            finally:
                self.release.set()
                await runner.cleanup()


@asynccontextmanager
async def production_wrapper(backend, root, env, model_name, base_url):
    """Only provider construction differs; every probe below uses BaseAgentWrapper APIs."""
    context = ApplicationContext(workspace_dir=str(root))
    if backend == "codex":
        server = SdkCodexAppServer(
            app_context=context,
            auth_mode="api_key",
            api_key=env["LLM_API_KEY"],
            base_url=base_url,
            config_overrides=[
                "project_doc_max_bytes=0",
                "check_for_update_on_startup=false",
                'model_reasoning_effort="none"',
            ],
        )
        context.components = {ComponentEnum.CODEX_APP_SERVER: {"default": server}}
        async with server, CodexAgentWrapper(app_context=context, model=model_name) as wrapper:
            yield wrapper
    elif backend == "claude":
        async with CcAgentWrapper(
            app_context=context,
            model=model_name,
            api_key=env["CLAUDE_CODE_API_KEY"],
            base_url=base_url,
            tools=[],
            thinking={"type": "disabled"},
            max_turns=1,
            env={
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "256",
                "API_TIMEOUT_MS": "60000",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "NO_PROXY": "127.0.0.1,localhost",
            },
        ) as wrapper:
            yield wrapper
    else:
        from agentscope.credential import OpenAICredential
        from agentscope.model import OpenAIChatModel

        model = OpenAIChatModel(
            credential=OpenAICredential(api_key=env["LLM_API_KEY"], base_url=base_url),
            model=model_name,
            parameters=OpenAIChatModel.Parameters(max_tokens=256),
            extra_body={"enable_thinking": False},
            stream=True,
            max_retries=0,
            client_kwargs={"timeout": 90},
        )
        wrapper = AsAgentWrapper(app_context=context, as_llm="", session_retention_days=0)
        wrapper.as_llm = SimpleNamespace(model=model)
        try:
            async with wrapper:
                yield wrapper
        finally:
            await model.client.close()


async def probe(backend, root, env, model_name):
    """Verify live fork, both append roles, retry, isolation and restart using one flow."""
    upstream = env["CLAUDE_CODE_BASE_URL"] if backend == "claude" else env["LLM_BASE_URL"]
    async with Relay(upstream).serve() as relay:
        source_label, user_label, assistant_label = [
            f"{prefix}_{uuid4().hex[:12]}" for prefix in ("source", "user", "assistant")
        ]
        options = {
            "system_prompt": "Follow exact output requests. Do not use tools.",
            "sandbox": "read-only",
            "approval_mode": "deny_all",
        }
        async with production_wrapper(backend, root, env, model_name, relay.base_url) as wrapper:
            source = await wrapper.create_session(**options)
            running = asyncio.create_task(
                wrapper.reply(
                    f"The fictional source label is {source_label}. Please briefly acknowledge it.",
                    resume=source,
                    **options,
                ),
            )
            observers = []
            first_outcome = "persisted"
            try:
                await asyncio.wait_for(relay.entered.wait(), 45)
                assert not running.done()
                anchor = (await wrapper.read_session(source)).last_turn_id
                child = await asyncio.wait_for(wrapper.create_session(resume=source, fork_session=True, **options), 15)
                progress("fork_created", backend=backend)
                assert child != source and not running.done()
                inherited = await wrapper.read_session(child)
                assert source_label in inherited.model_dump_json(), "Fork did not include current source input"
                child_result = await wrapper.reply(
                    "What is the fictional source label? Return it exactly.",
                    resume=child,
                    **options,
                )
                assert child_result["status"] == "completed" and source_label in child_result["result"], (
                    child_result.get("error") or child_result["result"]
                )
                assert not running.done(), "Fork changed the paused source execution"
                progress("fork_recalled_source", backend=backend)
                assistant_content = f"The fictional assistant label is {assistant_label}."
                user_content = f"The fictional user label is {user_label}."
                try:
                    await wrapper.append_session(
                        source,
                        assistant_content,
                        message_id="probe-assistant",
                        delivery="immediate",
                        timeout=0.5,
                    )
                except TimeoutError:
                    first_outcome = "timeout_with_pending_receipt"
                observers = [
                    asyncio.create_task(
                        wrapper.append_session(
                            source,
                            assistant_content,
                            message_id="probe-assistant",
                            delivery="immediate",
                            expected_revision="stale",
                            timeout=75,
                        ),
                    ),
                    asyncio.create_task(
                        wrapper.append_session(
                            source,
                            user_content,
                            message_id="probe-user",
                            role="user",
                            delivery="immediate",
                            timeout=75,
                        ),
                    ),
                ]
                await asyncio.sleep(0.05)
                assert not running.done()
                relay.release.set()
                original, assistant, user = await asyncio.wait_for(asyncio.gather(running, *observers), 90)
                assert original["status"] == "completed", original.get("error")
                snapshot = await wrapper.read_session(source)
                assert snapshot.last_turn_id == anchor, "Appending started a new user turn"
                for receipt in (assistant, user):
                    assert sum(message.id == receipt.id for message in snapshot.messages) == 1
                before = len(relay.requests)
                assert (
                    await wrapper.append_session(
                        source,
                        assistant_content,
                        message_id="probe-assistant",
                        expected_turn_id="stale",
                    )
                    == assistant
                )
                assert (
                    await wrapper.append_session(
                        source,
                        user_content,
                        message_id="probe-user",
                        role="user",
                        expected_revision="stale",
                    )
                    == user
                )
                assert len(relay.requests) == before
                detached = await wrapper.read_session(child)
                assert all(label not in detached.model_dump_json() for label in (user_label, assistant_label))
                recall = await wrapper.reply(
                    "What are the fictional user label and assistant label? Return both exact labels.",
                    resume=source,
                    **options,
                )
                assert all(label in recall["result"] for label in (user_label, assistant_label)), (
                    recall.get("error") or recall["result"]
                )
                progress("live_recall_passed", backend=backend)
            finally:
                relay.release.set()
                for task in (running, *observers):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(running, *observers, return_exceptions=True)
        async with production_wrapper(backend, root, env, model_name, relay.base_url) as restarted:
            assert await restarted.append_session(source, assistant_content, message_id="probe-assistant") == assistant
            result = await restarted.reply(
                "What are the fictional user label and assistant label? Return both exact labels.",
                resume=source,
                **options,
            )
            assert all(label in result["result"] for label in (user_label, assistant_label)), (
                result.get("error") or result["result"]
            )
        assert all(request["model"] == model_name for request in relay.requests)
        return {
            "implementation": "production wrapper",
            "active_fork": True,
            "source_continued": True,
            "source_and_child_isolated": True,
            "both_roles_persisted_once": True,
            "first_append_outcome": first_outcome,
            "same_turn_anchor": True,
            "retry_started_no_inference": True,
            "live_recall": True,
            "restart_recall": True,
            "model_requests": len(relay.requests),
            "usage": recall["usage"],
        }


async def main(arguments):
    """Read credentials in memory and redact them from any failure report."""
    env = parse_env_file(arguments.env_file)
    report = {
        "backend": arguments.backend,
        "model": arguments.model,
        "real_provider": True,
        "sdk_version": version(
            {"agentscope": "agentscope", "codex": "openai-codex", "claude": "claude-agent-sdk"}[arguments.backend],
        ),
    }
    if arguments.claude_use_dashscope:
        env["CLAUDE_CODE_BASE_URL"] = "https://dashscope.aliyuncs.com/apps/anthropic"
        env["CLAUDE_CODE_API_KEY"] = env["LLM_API_KEY"]
        report["credential_source"] = "LLM_API_KEY with DashScope Anthropic-compatible endpoint"
    try:
        with TemporaryDirectory(prefix=f"reme-{arguments.backend}-real-active-") as directory:
            async with asyncio.timeout(240):
                report["result"] = await probe(arguments.backend, Path(directory), env, arguments.model)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for name, value in env.items():
            if value and any(word in name for word in ("KEY", "TOKEN", "SECRET")):
                error = error.replace(value, "[REDACTED]")
        report["error"] = error
    print(json.dumps(report, ensure_ascii=False), flush=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return "error" in report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real", action="store_true", required=True)
    parser.add_argument("--backend", choices=("agentscope", "codex", "claude"), required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument(
        "--claude-use-dashscope",
        action="store_true",
        help="Use LLM_API_KEY with the repository's DashScope Anthropic default",
    )
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
