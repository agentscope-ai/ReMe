"""Tests for the resource ingest path: ``IngestStep`` + helpers.

``ingest`` is the **passive** ingest entry point — external channels
push assets into ``resource/<YYYY-MM-DD>/``, where each call appends a
:class:`FileNode` row to ``meta.json`` (provenance on
``front_matter``) and regenerates the day's ``<date>.md`` view from
the updated meta. These tests exercise that contract end-to-end on a
temp vault, plus the pure ``_assemble_day_md`` helper in isolation.

The bucket file name is always derived: ``<channel>__<HHMMSS>__<basename>``.
Duplicates surface as errors (no silent suffixing).
"""

# pylint: disable=protected-access

import asyncio
import datetime
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileFrontMatter, FileNode
from reme4.steps.transfer import ingest as crud_ingest
from reme4.steps.transfer.ingest import _assemble_day_md

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
    """Minimal LocalFileStore (embedding disabled). vault_path resolves to CWD."""
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    return store


def _metadata(step) -> dict:
    return step.context.response.metadata


def _meta(tmp: str, date: str) -> list[dict]:
    return json.loads((Path(tmp) / "resource" / date / "meta.json").read_text(encoding="utf-8"))


def _today() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")


# Matches the canonical derived filename shape.
_NAME_RE = re.compile(r"^([a-z0-9][a-z0-9-]*)__(\d{6})__(.+)$")


# -- _assemble_day_md ----------------------------------------------------


def _entry(name: str, **fm_fields) -> FileNode:
    """Build a FileNode resource entry: path = resource/<date>/<name>, all
    provenance fields go onto front_matter (extras allowed)."""
    return FileNode(
        path=f"resource/2026-05-22/{name}",
        st_mtime=0.0,
        front_matter=FileFrontMatter(**fm_fields),
    )


def test_assemble_day_md_renders_entries():
    """The derived view lists entries with channel / source / time / description."""
    entries = [
        _entry(
            "wechat__143000__report.pdf",
            description="Q1 report",
            channel="wechat",
            source="design-group",
            received_at="2026-05-22T14:30:00",
        ),
        _entry("browser__095501__bare.png", channel="browser"),
    ]
    md = _assemble_day_md(entries, "2026-05-22")
    assert "name: 2026-05-22" in md
    assert "assets: [wechat__143000__report.pdf, browser__095501__bare.png]" in md
    assert (
        "- [[resource/2026-05-22/wechat__143000__report.pdf]] — wechat from `design-group` at 14:30 — Q1 report" in md
    )
    assert "- [[resource/2026-05-22/browser__095501__bare.png]] — browser" in md
    print("✓ test_assemble_day_md_renders_entries passed")


def test_assemble_day_md_empty_bucket():
    """An empty bucket still produces a well-formed frontmatter + header."""
    md = _assemble_day_md([], "2026-05-22")
    assert "assets: []" in md
    assert "# 2026-05-22 resources" in md
    print("✓ test_assemble_day_md_empty_bucket passed")


# -- _validate_basename (direct, pure) -----------------------------------


def test_validate_basename_rejects_path_separators():
    """Path-separator basenames are rejected even if the public API can no
    longer reach this code path (Path(...).name strips them) — defense in depth."""
    for bad in ("evil/payload.pdf", "..\\winpath.pdf", "../escape.pdf"):
        err = crud_ingest._validate_basename(bad)
        assert "path separators" in err or "reserved" in err, (bad, err)
    print("✓ test_validate_basename_rejects_path_separators passed")


def test_validate_basename_rejects_dot_segments():
    """`.` and `..` are explicitly reserved."""
    for bad in (".", ".."):
        err = crud_ingest._validate_basename(bad)
        assert "reserved" in err or "start with '.'" in err, (bad, err)
    print("✓ test_validate_basename_rejects_dot_segments passed")


# -- _validate_channel (direct, pure) ------------------------------------


def test_validate_channel_accepts_safe_identifiers():
    """Lowercase letters / digits / dashes, starting alnum — all accepted."""
    for ok in ("wechat", "email", "api", "browser", "slack-1", "ch1"):
        assert crud_ingest._validate_channel(ok) == "", ok
    print("✓ test_validate_channel_accepts_safe_identifiers passed")


def test_validate_channel_rejects_unsafe_identifiers():
    """Uppercase, underscores, leading dash, empty, special chars — rejected."""
    for bad in ("", "WeChat", "we_chat", "-leading", "we chat", "we/chat", "我"):
        err = crud_ingest._validate_channel(bad)
        assert err, bad
    print("✓ test_validate_channel_rejects_unsafe_identifiers passed")


# -- IngestStep end-to-end --------------------------------------


def test_upload_first_call_creates_bucket():
    """First upload creates resource/<date>/, copies the asset under the derived name."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "incoming.pdf"
            src.write_bytes(b"%PDF-fake")

            step = crud_ingest.IngestStep(file_store=store)
            await step(
                path=str(src),
                channel="wechat",
                description="Q1 report",
                metadata={"source": "design-group"},
            )
            payload = _metadata(step)
            assert "error" not in payload, payload
            date = payload["date"]
            assert date == _today()

            m = _NAME_RE.match(payload["name"])
            assert m is not None, payload["name"]
            assert m.group(1) == "wechat"
            assert m.group(3) == "incoming.pdf"
            assert payload["path"] == f"resource/{date}/{payload['name']}"

            bucket = Path(tmp) / "resource" / date
            assert (bucket / payload["name"]).read_bytes() == b"%PDF-fake"

            meta = _meta(tmp, date)
            assert len(meta) == 1
            assert Path(meta[0]["path"]).name == payload["name"]
            fm = meta[0]["front_matter"]
            assert fm["channel"] == "wechat"
            assert fm["source"] == "design-group"
            assert fm["description"] == "Q1 report"

            day_md = (bucket / f"{date}.md").read_text(encoding="utf-8")
            assert f"name: {date}" in day_md
            assert payload["name"] in day_md
            await store.close()
        print("✓ test_upload_first_call_creates_bucket passed")

    asyncio.run(run())


def test_upload_metadata_optional():
    """metadata is optional — minimal call is just path + channel + description."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "small.txt"
            src.write_text("x")
            step = crud_ingest.IngestStep(file_store=store)
            await step(path=str(src), channel="api", description="minimal")
            payload = _metadata(step)
            assert "error" not in payload, payload
            row = _meta(tmp, payload["date"])[0]
            fm = row["front_matter"]
            assert fm["channel"] == "api"
            assert fm.get("source", "") == ""
            await store.close()
        print("✓ test_upload_metadata_optional passed")

    asyncio.run(run())


def test_upload_appends_to_existing_meta():
    """Subsequent uploads append to meta.json and regenerate <date>.md."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            names = []
            for i, suffix in enumerate(("first", "second"), start=1):
                src = Path(tmp) / f"{suffix}.txt"
                src.write_text(f"payload-{i}")
                step = crud_ingest.IngestStep(file_store=store)
                await step(path=str(src), channel="email", description=f"item {i}")
                payload = _metadata(step)
                assert "error" not in payload, payload
                names.append(payload["name"])

            date = _today()
            meta = _meta(tmp, date)
            assert [Path(row["path"]).name for row in meta] == names

            day_md = (Path(tmp) / "resource" / date / f"{date}.md").read_text(encoding="utf-8")
            assert f"assets: [{', '.join(names)}]" in day_md
            for name in names:
                assert f"[[resource/{date}/{name}]]" in day_md
            await store.close()
        print("✓ test_upload_appends_to_existing_meta passed")

    asyncio.run(run())


def test_upload_errors_on_duplicate_same_second(monkeypatch):
    """Two uploads of the same (channel, second, basename) → second one errors out."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            # Pin time so both calls land in the same HHMMSS slot deterministically.
            fixed = datetime.datetime(2026, 5, 22, 15, 30, 22)

            class _FrozenDT(datetime.datetime):
                @classmethod
                def now(cls, tz=None):  # pylint: disable=unused-argument
                    return fixed

            monkeypatch.setattr(crud_ingest.datetime, "datetime", _FrozenDT)

            for i, body in enumerate((b"alpha", b"beta")):
                src = Path(tmp) / "incoming.pdf"
                src.write_bytes(body)
                step = crud_ingest.IngestStep(file_store=store)
                await step(path=str(src), channel="wechat", description="dup test")
                payload = _metadata(step)
                if i == 0:
                    assert "error" not in payload, payload
                else:
                    assert "duplicate" in payload.get("error", "").lower(), payload

            bucket = Path(tmp) / "resource" / "2026-05-22"
            # Only the first upload's file should be on disk.
            payloads = [p for p in bucket.iterdir() if p.is_file() and p.name.endswith(".pdf")]
            assert len(payloads) == 1
            assert payloads[0].read_bytes() == b"alpha"
            meta = _meta(tmp, "2026-05-22")
            assert len(meta) == 1
            await store.close()
        print("✓ test_upload_errors_on_duplicate_same_second passed")

    asyncio.run(run())


def test_upload_errors_on_duplicate_against_on_disk_stray(monkeypatch):
    """A stray file on disk (no meta row) still counts as a collision → error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            fixed = datetime.datetime(2026, 5, 22, 15, 30, 22)

            class _FrozenDT(datetime.datetime):
                @classmethod
                def now(cls, tz=None):  # pylint: disable=unused-argument
                    return fixed

            monkeypatch.setattr(crud_ingest.datetime, "datetime", _FrozenDT)

            bucket = Path(tmp) / "resource" / "2026-05-22"
            bucket.mkdir(parents=True)
            stray = bucket / "api__153022__report.pdf"
            stray.write_bytes(b"orphan")

            src = Path(tmp) / "report.pdf"
            src.write_bytes(b"fresh")
            step = crud_ingest.IngestStep(file_store=store)
            await step(path=str(src), channel="api", description="fresh copy")
            payload = _metadata(step)
            assert "duplicate" in payload.get("error", "").lower(), payload
            # Stray untouched.
            assert stray.read_bytes() == b"orphan"
            await store.close()
        print("✓ test_upload_errors_on_duplicate_against_on_disk_stray passed")

    asyncio.run(run())


def test_upload_rejects_missing_source():
    """Missing local file → error, no bucket created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = crud_ingest.IngestStep(file_store=store)
            await step(
                path=str(Path(tmp) / "ghost.txt"),
                channel="email",
                description="x",
            )
            payload = _metadata(step)
            assert "not found" in payload.get("error", "")
            assert not (Path(tmp) / "resource").exists()
            await store.close()
        print("✓ test_upload_rejects_missing_source passed")

    asyncio.run(run())


def test_upload_requires_channel():
    """Missing / blank / malformed channel → error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "x.txt"
            src.write_text("x")
            for bad in ("   ", "WeChat", "we_chat"):
                step = crud_ingest.IngestStep(file_store=store)
                await step(path=str(src), channel=bad, description="x")
                payload = _metadata(step)
                assert "channel" in payload.get("error", ""), (bad, payload)
            await store.close()
        print("✓ test_upload_requires_channel passed")

    asyncio.run(run())


def test_upload_requires_description():
    """Empty / blank description → error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "x.txt"
            src.write_text("x")
            step = crud_ingest.IngestStep(file_store=store)
            await step(path=str(src), channel="api", description="   ")
            payload = _metadata(step)
            assert "description" in payload.get("error", "")
            await store.close()
        print("✓ test_upload_requires_description passed")

    asyncio.run(run())


def test_upload_rejects_non_dict_metadata():
    """Passing a non-dict in `metadata=` → error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "x.txt"
            src.write_text("x")
            step = crud_ingest.IngestStep(file_store=store)
            await step(
                path=str(src),
                channel="api",
                description="x",
                metadata="source=foo",
            )
            payload = _metadata(step)
            assert "metadata" in payload.get("error", "")
            await store.close()
        print("✓ test_upload_rejects_non_dict_metadata passed")

    asyncio.run(run())


def test_upload_rejects_reserved_metadata_keys():
    """Step-managed keys (`name`, `channel`, `received_at`, `description`)
    can't be smuggled in via metadata."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "x.txt"
            src.write_text("x")
            for bad in ("name", "channel", "received_at", "description"):
                step = crud_ingest.IngestStep(file_store=store)
                await step(
                    path=str(src),
                    channel="api",
                    description="x",
                    metadata={bad: "evil"},
                )
                payload = _metadata(step)
                assert "reserved" in payload.get("error", ""), (bad, payload)
            await store.close()
        print("✓ test_upload_rejects_reserved_metadata_keys passed")

    asyncio.run(run())


def test_upload_preserves_extra_metadata_keys():
    """Arbitrary keys in `metadata` land on the meta.json row verbatim."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "x.txt"
            src.write_text("x")
            step = crud_ingest.IngestStep(file_store=store)
            await step(
                path=str(src),
                channel="api",
                description="tagged",
                metadata={
                    "source": "https://example.com",
                    "tag": "design",
                    "priority": 3,
                },
            )
            payload = _metadata(step)
            assert "error" not in payload, payload
            row = _meta(tmp, payload["date"])[0]
            fm = row["front_matter"]
            assert fm["tag"] == "design"
            assert fm["priority"] == 3
            assert fm["source"] == "https://example.com"
            await store.close()
        print("✓ test_upload_preserves_extra_metadata_keys passed")

    asyncio.run(run())


# -- basename-derivation safety ------------------------------------------


def test_upload_rejects_dotfile_source():
    """A source file whose basename starts with '.' → error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            for bad in (".hidden", ".lock", ".env"):
                src = Path(tmp) / bad
                src.write_text("x")
                step = crud_ingest.IngestStep(file_store=store)
                await step(
                    path=str(src),
                    channel="api",
                    description="x",
                )
                payload = _metadata(step)
                assert "start with '.'" in payload.get("error", ""), payload
                src.unlink()
            await store.close()
        print("✓ test_upload_rejects_dotfile_source passed")

    asyncio.run(run())


# -- payload-shape sanity ------------------------------------------------


def test_upload_records_received_at_internally():
    """`received_at` is not a caller param but the step stamps it from the
    system clock so the day's <date>.md HH:MM column renders."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "doc.pdf"
            src.write_bytes(b"%PDF")
            step = crud_ingest.IngestStep(file_store=store)
            await step(path=str(src), channel="api", description="x")
            payload = _metadata(step)
            assert "error" not in payload
            meta = _meta(tmp, payload["date"])
            assert len(meta) == 1
            stamped = meta[0]["front_matter"]["received_at"]
            parsed = datetime.datetime.fromisoformat(stamped)
            assert parsed.strftime("%Y-%m-%d") == payload["date"]
            # The HHMMSS slot in the name matches the stamped time.
            m = _NAME_RE.match(payload["name"])
            assert m is not None
            assert m.group(2) == parsed.strftime("%H%M%S")
            await store.close()
        print("✓ test_upload_records_received_at_internally passed")

    asyncio.run(run())


def test_upload_preserves_description_verbatim_in_meta():
    """meta.json carries the verbatim multi-line description (downstream agents
    rely on it for analysis hints); only the day.md bullet flattens for display."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            src = Path(tmp) / "doc.pdf"
            src.write_bytes(b"%PDF")
            multi = "wechat group screenshot\nfrom design-group at 14:30\nlikely a Q1 KPI table — extract numbers"
            step = crud_ingest.IngestStep(file_store=store)
            await step(
                path=str(src),
                channel="api",
                description=multi,
            )
            payload = _metadata(step)
            assert "error" not in payload, payload

            # meta.json preserves the original — downstream dreamer sees full hint.
            meta = _meta(tmp, payload["date"])
            assert meta[0]["front_matter"]["description"] == multi

            # day.md bullet is flattened (single line, no embedded newlines).
            day_md = (Path(tmp) / "resource" / payload["date"] / f"{payload['date']}.md").read_text(encoding="utf-8")
            flat = " ".join(multi.split())
            assert flat in day_md
            # The bullet line itself must contain the flattened text — and no embedded newline.
            bullet_prefix = f"- [[resource/{payload['date']}/{payload['name']}]]"
            bullets = [line for line in day_md.splitlines() if line.startswith(bullet_prefix)]
            assert len(bullets) == 1, day_md
            assert flat in bullets[0]
            await store.close()
        print("✓ test_upload_preserves_description_verbatim_in_meta passed")

    asyncio.run(run())


def test_assemble_day_md_flattens_multiline_description():
    """Pure helper: a multi-line description on an entry renders as a single
    bullet line with newlines collapsed."""
    entries = [
        _entry(
            "api__120000__doc.pdf",
            description="line one\nline two\n  line three",
            channel="api",
            received_at="2026-05-22T12:00:00",
        ),
    ]
    md = _assemble_day_md(entries, "2026-05-22")
    assert "line one line two line three" in md
    # The bullet line must contain the flattened text on a single line.
    bullets = [line for line in md.splitlines() if line.startswith("- [[resource/2026-05-22/api__120000__doc.pdf]]")]
    assert len(bullets) == 1, md
    assert "line one line two line three" in bullets[0]
    print("✓ test_assemble_day_md_flattens_multiline_description passed")


if __name__ == "__main__":
    print("\n=== resource step tests ===")
    test_assemble_day_md_renders_entries()
    test_assemble_day_md_empty_bucket()
    test_validate_basename_rejects_path_separators()
    test_validate_basename_rejects_dot_segments()
    test_validate_channel_accepts_safe_identifiers()
    test_validate_channel_rejects_unsafe_identifiers()
    test_upload_first_call_creates_bucket()
    test_upload_metadata_optional()
    test_upload_appends_to_existing_meta()
    test_upload_rejects_missing_source()
    test_upload_requires_channel()
    test_upload_requires_description()
    test_upload_rejects_non_dict_metadata()
    test_upload_rejects_reserved_metadata_keys()
    test_upload_preserves_extra_metadata_keys()
    test_upload_rejects_dotfile_source()
    test_upload_records_received_at_internally()
    test_upload_preserves_description_verbatim_in_meta()
    test_assemble_day_md_flattens_multiline_description()
    print("\n所有测试通过!")
