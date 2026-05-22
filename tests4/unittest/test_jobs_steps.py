"""Tests for job steps: digester (skip paths) + synchronizer (skip-on-empty).

Both jobs ultimately drive a ReAct agent against an LLM, so the LLM-driven
paths require an as_llm component and are exercised via the full E2E
tests. These unit tests cover the LLM-free branches:

  * Digester with empty daily_paths   → skipped
  * Digester with no LLM configured   → skipped with error message
  * DistillResult defaults
  * Synchronizer with empty messages   → skipped early-return
"""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile
import warnings

from reme4.components.file_store import LocalFileStore
from reme4.steps.jobs import digester as digester_step
from reme4.steps.jobs import synchronizer as synchronizer_step

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


async def _make_store() -> LocalFileStore:
    store = LocalFileStore(store_name="t", embedding_model="")
    await store.start()
    return store


def _answer(step) -> dict:
    return json.loads(step.context.response.answer)


# -- Digester: skip paths -----------------------------------------------


def test_digester_skipped_when_no_dailies():
    """Empty / missing daily_paths → DistillResult(skipped=True), no LLM use."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = digester_step.Digester(file_store=store)
            await step(daily_paths=[])
            payload = _answer(step)
            assert payload["used_llm"] is False
            assert payload["skipped"] is True
            assert payload["error"] == ""
            assert step.context.response.success is True

            # Same outcome when daily_paths is omitted entirely.
            step2 = digester_step.Digester(file_store=store)
            await step2()
            payload2 = _answer(step2)
            assert payload2["skipped"] is True
            assert payload2["error"] == ""
            assert step2.context.response.success is True

            await store.close()
        print("✓ test_digester_skipped_when_no_dailies passed")

    asyncio.run(run())


def test_digester_skipped_when_no_llm():
    """Daily paths provided but no app_context → skipped with recorded error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = digester_step.Digester(file_store=store)
            await step(daily_paths=["daily/2026-05-17/foo/foo.md"])
            payload = _answer(step)
            assert payload["used_llm"] is False
            assert payload["skipped"] is True
            assert "no as_llm" in payload["error"]
            assert step.context.response.success is False
            await store.close()
        print("✓ test_digester_skipped_when_no_llm passed")

    asyncio.run(run())


def test_distill_result_default_fields():
    """DistillResult defaults: simplified shape — no audit-derived buckets."""
    r = digester_step.DistillResult(used_llm=False)
    assert r.used_llm is False
    assert r.skipped is False
    assert r.daily_read == []
    assert r.summary == ""
    assert r.error == ""
    print("✓ test_distill_result_default_fields passed")


def test_pack_daily_handles_missing_path():
    """_pack_daily renders a graceful placeholder when the daily doesn't exist."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            blob = digester_step._pack_daily(store, "daily/2026-05-17/nope/nope.md")
            assert "daily/2026-05-17/nope/nope.md" in blob
            assert "does not exist" in blob
            await store.close()
        print("✓ test_pack_daily_handles_missing_path passed")

    asyncio.run(run())


def test_pack_daily_reads_summary_and_lists_siblings():
    """_pack_daily includes the summary note body plus a sibling listing."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            # Create a fake daily folder with summary + materials.
            from pathlib import Path

            folder = Path(tmp) / "daily/2026-05-17/foo"
            folder.mkdir(parents=True)
            (folder / "foo.md").write_text("# summary\nHello world", encoding="utf-8")
            (folder / "ref.pdf").write_text("dummy", encoding="utf-8")
            (folder / "notes.md").write_text("notes body", encoding="utf-8")

            blob = digester_step._pack_daily(store, "daily/2026-05-17/foo")
            assert "Hello world" in blob
            assert "Sibling materials" in blob
            assert "- ref.pdf" in blob
            assert "- notes.md" in blob
            # The summary note itself should NOT appear in the sibling list.
            assert "- foo.md" not in blob

            await store.close()
        print("✓ test_pack_daily_reads_summary_and_lists_siblings passed")

    asyncio.run(run())


# -- Digester: protocol override interface ------------------------------


def test_digester_loads_default_protocol_when_unspecified():
    """No override → falls back to the protocol.md shipped next to digester.py."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = digester_step.Digester(file_store=store)
            # The shipped reference protocol has its title in the first line.
            assert "Memory Protocol" in step._protocol
            await store.close()
        print("✓ test_digester_loads_default_protocol_when_unspecified passed")

    asyncio.run(run())


def test_digester_protocol_inline_override_wins():
    """``protocol=<str>`` replaces the protocol document wholesale."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = digester_step.Digester(
                file_store=store,
                protocol="MY CUSTOM SCHEMA",
            )
            assert step._protocol == "MY CUSTOM SCHEMA"
            assert "Memory Protocol" not in step._protocol
            await store.close()
        print("✓ test_digester_protocol_inline_override_wins passed")

    asyncio.run(run())


def test_digester_protocol_path_override_reads_file():
    """``protocol_path=<path>`` loads from disk when ``protocol`` not given."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            from pathlib import Path

            custom = Path(tmp) / "service-protocol.md"
            custom.write_text("# Service-layer schema\nrole: agent", encoding="utf-8")

            step = digester_step.Digester(
                file_store=store,
                protocol_path=str(custom),
            )
            assert "Service-layer schema" in step._protocol
            await store.close()
        print("✓ test_digester_protocol_path_override_reads_file passed")

    asyncio.run(run())


def test_digester_inline_protocol_beats_path():
    """When both are given, inline ``protocol`` wins over ``protocol_path``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            from pathlib import Path

            custom = Path(tmp) / "ignored.md"
            custom.write_text("FROM FILE", encoding="utf-8")

            step = digester_step.Digester(
                file_store=store,
                protocol="FROM INLINE",
                protocol_path=str(custom),
            )
            assert step._protocol == "FROM INLINE"
            await store.close()
        print("✓ test_digester_inline_protocol_beats_path passed")

    asyncio.run(run())


def test_digester_protocol_path_missing_falls_back_to_default():
    """A bad ``protocol_path`` silently falls back to the shipped default."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = digester_step.Digester(
                file_store=store,
                protocol_path="/nonexistent/path/to/protocol.md",
            )
            # Falls back to bundled default rather than empty string.
            assert "Memory Protocol" in step._protocol
            await store.close()
        print("✓ test_digester_protocol_path_missing_falls_back_to_default passed")

    asyncio.run(run())


# -- Synchronizer: skip-on-empty -----------------------------------------


def test_synchronizer_skips_on_empty_messages():
    """Empty messages list → SynchronizerResult(skipped=True), no LLM use."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = synchronizer_step.Synchronizer(file_store=store)
            await step(messages=[])
            payload = _answer(step)
            assert payload["used_llm"] is False
            assert payload["skipped"] is True
            assert step.context.response.success is True
            await store.close()
        print("✓ test_synchronizer_skips_on_empty_messages passed")

    asyncio.run(run())


def test_synchronizer_skips_when_messages_omitted():
    """Missing `messages` key behaves the same as empty list."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = synchronizer_step.Synchronizer(file_store=store)
            await step()  # no messages kwarg at all
            payload = _answer(step)
            assert payload["skipped"] is True
            await store.close()
        print("✓ test_synchronizer_skips_when_messages_omitted passed")

    asyncio.run(run())


def test_synchronizer_result_default_fields():
    """SynchronizerResult defaults: simplified shape — no audit-derived buckets."""
    r = synchronizer_step.SynchronizerResult()
    assert r.used_llm is False
    assert r.skipped is False
    assert r.actions == ""
    assert r.workspace is None
    assert r.summary is None
    print("✓ test_synchronizer_result_default_fields passed")


if __name__ == "__main__":
    print("\n=== jobs step tests ===")
    test_digester_skipped_when_no_dailies()
    test_digester_skipped_when_no_llm()
    test_distill_result_default_fields()
    test_pack_daily_handles_missing_path()
    test_pack_daily_reads_summary_and_lists_siblings()
    test_digester_loads_default_protocol_when_unspecified()
    test_digester_protocol_inline_override_wins()
    test_digester_protocol_path_override_reads_file()
    test_digester_inline_protocol_beats_path()
    test_digester_protocol_path_missing_falls_back_to_default()
    test_synchronizer_skips_on_empty_messages()
    test_synchronizer_skips_when_messages_omitted()
    test_synchronizer_result_default_fields()
    print("\n所有测试通过!")
