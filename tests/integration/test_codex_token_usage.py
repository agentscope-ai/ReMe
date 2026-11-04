"""Opt-in Qwen token estimates and Codex per-turn accounting with synthetic inputs.

REME_CODEX_INTEGRATION=1 pytest tests/integration/test_codex_token_usage.py -v -s
Credentials come from LLM_API_KEY / LLM_BASE_URL in the environment or repository .env.
"""

# pylint: disable=redefined-outer-name

import asyncio
import json
import math
import os
from pathlib import Path
import random
import string
import tempfile

import httpx
import pytest

from reme.application import Application
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum
from reme.utils import global_counter_get_all, parse_env_file

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to authorize live Qwen token tests",
)

_TEXT = {
    "zh": (
        "请阅读以下合成工作记录，只回复“收到”。\n",
        "今天完成了项目配置检查，确认服务启动正常，文件归用户所有。"
        "下一步核对上下文压缩、工具调用和会话恢复，记录输入与输出的统计结果。\n",
    ),
    "en": (
        "Read the synthetic work log below and reply only OK.\n",
        "The configuration review is complete and the service starts correctly. Workspace files belong to the user. "
        "Next, verify context compaction, tool calls, and session recovery, recording input and output token counts.\n",
    ),
    "mixed": (
        "阅读以下 synthetic work log，只回复 OK。\n",
        "今天完成 configuration review，service starts correctly。Workspace 文件归用户所有。"
        "下一步检查 context compaction、tool calls 和 session recovery，记录 input/output token counts。\n",
    ),
}

_RANDOM = random.Random(20261105)
_IDENTIFIERS = ["".join(_RANDOM.choices(string.ascii_letters + string.digits, k=96)) for _ in range(1024)]
_TEXT.update(
    {
        "zh_rare": (
            "以下是罕见汉字压力样本，只回复“收到”。\n",
            "龘靐齉齾爩鱻麤飝灪纞虋麷灩韽爨灥麣鸞鬣𠮷𠮟𪚥𰻞𰻝。\n",
        ),
        "en_ids": (
            "Read these synthetic identifiers and reply only OK.\n",
            "\n".join(f"trace_id={value} status=ok" for value in _IDENTIFIERS),
        ),
        "mixed_logs": (
            "阅读以下合成 JSON 日志，只回复 OK。\n",
            "\n".join(
                json.dumps({"trace_id": value, "状态": "成功", "message": "正在验证 token usage"}, ensure_ascii=False)
                for value in _IDENTIFIERS
            ),
        ),
    },
)

_QUERIES = {
    "zh": (
        "必须按顺序调用两次加法工具：先计算二十一加二十一，等待工具返回后，"
        "再把返回值加八。不要使用其他工具。最后只输出最终的阿拉伯数字结果。",
        "继续使用同一会话，把上一轮最终结果通过加法工具加五。最后只输出阿拉伯数字结果。",
    ),
    "en": (
        "Call the add tool twice, strictly sequentially: first add twenty-one and twenty-one, "
        "wait for the tool result, then add eight to that returned value. Use no other tools. "
        "Finally output only the numeric result.",
        "Continue this conversation. Use the add tool to add five to the previous final result. "
        "Output only the numeric result.",
    ),
    "mixed": (
        "请先调用 add tool 计算 21 + 21，wait for its result，再把返回值加 8。"
        "Call the tool sequentially twice. 不要使用其他工具。最后只输出 numeric result。",
        "Continue 同一会话，使用 add tool 把上一轮最终结果加 5。只输出 numeric result。",
    ),
}


@pytest.fixture
def live_config(monkeypatch):
    """Load the real proactive settings without displaying credentials."""
    path = Path(__file__).resolve().parents[2] / ".env"
    environment = {**(parse_env_file(path) if path.is_file() else {}), **os.environ}
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "CODEX_MODEL_NAME"):
        if value := environment.get(name):
            monkeypatch.setenv(name, value)
    assert os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL"), "LLM credentials are required"
    return resolve_app_config(config="proactive", log_config=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("language", _TEXT)
@pytest.mark.parametrize("size", [4096, 32768, 131072])
async def test_live_qwen_text_token_estimate(live_config, language, size):
    """Compare matched UTF-8 byte sizes against provider input usage, not an assumed tokenizer."""
    server = live_config["components"]["codex_app_server"]["default"]
    model = live_config["components"]["agent_wrapper"]["default"]["model"]
    prefix, sentence = _TEXT[language]
    remaining = size - len(prefix.encode())
    body = (sentence * (remaining // len(sentence.encode()) + 1)).encode()[:remaining].decode(errors="ignore")
    text = prefix + body
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            server["base_url"].rstrip("/") + "/responses",
            headers={"Authorization": "Bearer " + server["api_key"]},
            json={
                "model": model,
                "input": text,
                "reasoning": {"effort": "none"},
                "max_output_tokens": 32,
                "stream": False,
            },
        )
    assert response.status_code == 200, f"Responses request returned HTTP {response.status_code}"
    usage = response.json()["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    estimate = math.ceil(len(text.encode()) / 4)
    print(
        json.dumps(
            {
                "language": language,
                "bytes": len(text.encode()),
                "byte_div4": estimate,
                "input_tokens": usage["input_tokens"],
                "actual_over_estimate": usage["input_tokens"] / estimate,
            },
        ),
    )


def _reported_turns(home):
    """Independently sum distinct native reports after the Codex process has flushed its rollout."""
    turns = {}
    for path in (home / "sessions").rglob("*.jsonl"):
        turn_id = None
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            payload = entry.get("payload", {})
            if entry.get("type") != "event_msg":
                continue
            if payload.get("type") == "task_started":
                turn_id = payload["turn_id"]
                turns.setdefault(turn_id, [])
            if payload.get("type") != "token_count" or not payload.get("info"):
                continue
            info = payload["info"]
            last, total = info["last_token_usage"], info["total_token_usage"]
            identity = (total["input_tokens"], total["output_tokens"])
            if turn_id and identity not in seen and last["input_tokens"] + last["output_tokens"] > 0:
                turns[turn_id].append({key: last[key] for key in ("input_tokens", "output_tokens", "total_tokens")})
            seen.add(identity)
    return list(turns.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("language", _QUERIES)
async def test_live_codex_turn_usage_and_resume(live_config, language, tmp_path):
    """Compare non-streaming and resumed streaming totals against actual multi-call provider reports."""
    with tempfile.TemporaryDirectory(prefix="reme-codex-token-") as workspace:
        config = {
            "workspace_dir": workspace,
            "enable_logo": False,
            "log_to_console": False,
            "log_to_file": False,
            "service": {"backend": "cli"},
            "components": {key: live_config["components"][key] for key in ("codex_app_server", "agent_wrapper")},
            "jobs": {
                "add": {
                    "backend": "base",
                    "description": "Return a + b. Use this tool for all addition.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                    "steps": [{"backend": "add_step"}],
                },
            },
        }
        app = Application(**config)
        server = app.context.components["codex_app_server"]["default"]
        first_query, second_query = _QUERIES[language]
        async with app:
            wrapper = app.context.components["agent_wrapper"]["default"]
            async with asyncio.timeout(180):
                original = await wrapper.reply(first_query, job_tools=["add"])
                assert "50" in original["result"]
                chunks = [
                    chunk
                    async for chunk in wrapper.reply_stream(
                        second_query,
                        session_id=original["session_id"],
                        job_tools=["add"],
                    )
                ]
            assert "55" in "".join(chunk.chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.CONTENT)
            streamed = [chunk.chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.USAGE]
            assert streamed, "provider usage was not reported"
            usages = [original["usage"], streamed[-1]]
            counters = global_counter_get_all(app.context.metadata, ["__token_counter", "default"])["children"]
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                assert counters[key]["value"] == sum(usage[key] for usage in usages)
        turns = _reported_turns(server.session_path)
        assert len(turns) == 2 and len(turns[0]) >= 3 and len(turns[1]) >= 2
        rows = []
        for calls, usage in zip(turns, usages):
            expected = {key: sum(call[key] for call in calls) for key in usage}
            assert usage == expected
            rows.append({"model_calls": len(calls), "last_call_tokens": calls[-1]["total_tokens"], "usage": usage})
        report = {"language": language, "turns": rows}
        (tmp_path / "usage-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
