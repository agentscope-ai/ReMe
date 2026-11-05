"""Date-range and pagination contracts for the read-only daily_list job."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from reme.components import ApplicationContext
from reme.components.file_store import LocalFileStore
from reme.steps.file_io.daily_list import DailyListStep


@pytest_asyncio.fixture(name="step")
async def daily_step(tmp_path, monkeypatch):
    """Use a real file store in an isolated workspace, without model services."""
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(name="daily", embedding_store="")
    await store.start()
    context = ApplicationContext(workspace_dir=str(tmp_path), timezone="UTC")
    try:
        yield DailyListStep(file_store=store, app_context=context)
    finally:
        await store.close()


def note(root: Path, day: str, name: str, *, daily_dir="daily", metadata="") -> Path:
    """Create a note whose full frontmatter must survive projection."""
    path = root / daily_dir / day / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\n{metadata}---\nbody\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_range_is_inclusive_newest_first_and_keeps_full_frontmatter(step, tmp_path):
    """Range boundaries and page order must not overwrite user metadata."""
    for day, name in [("2024-02-28", "old"), ("2024-02-29", "b"), ("2024-02-29", "a"), ("2024-03-01", "new")]:
        note(tmp_path, day, name, metadata="date: custom\nlabels: [one, two]\n")
    await step(start_date="2024-02-29", end_date="2024-03-01", page_size=2)
    response = step.context.response
    assert response.success
    assert response.metadata == {
        "start_date": "2024-02-29",
        "end_date": "2024-03-01",
        "page": 1,
        "page_size": 2,
        "total_notes": 3,
        "total_pages": 2,
        "count": 2,
        "notes": [
            {"path": "daily/2024-03-01/new.md", "name": "new", "date": "custom", "labels": ["one", "two"]},
            {"path": "daily/2024-02-29/a.md", "name": "a", "date": "custom", "labels": ["one", "two"]},
        ],
    }
    assert "daily/2024-03-01/new.md" in response.answer
    assert "daily/2024-02-29/b.md" not in response.answer


@pytest.mark.asyncio
async def test_opt_in_pagination_preserves_unpaged_single_day_shape(step, tmp_path):
    """Pagination is optional, stable and explicit about pages beyond the end."""
    for name in ("a", "b", "c"):
        note(tmp_path, "2024-03-01", name)
    await step(date="2024-03-01")
    assert set(step.context.response.metadata) == {"date", "count", "notes"}
    assert step.context.response.metadata["count"] == 3
    await step(date="2024-03-01", page=2, page_size=2)
    result = step.context.response.metadata
    assert result["count"] == 1 and result["total_notes"] == 3
    assert result["notes"][0]["name"] == "c"
    await step(date="2024-03-01", page=3, page_size=2)
    assert step.context.response.success
    assert step.context.response.metadata["notes"] == []
    assert step.context.response.metadata["total_pages"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"start_date": "2024-03-01"},
        {"end_date": "2024-03-01"},
        {"start_date": None, "end_date": "2024-03-01"},
        {"start_date": "", "end_date": "2024-03-01"},
        {"start_date": "2024-W09-4", "end_date": "2024-03-01"},
        {"start_date": "2024-03-02", "end_date": "2024-03-01"},
        {"start_date": "2024-02-30", "end_date": "2024-03-01"},
        {"start_date": "2024-03-01T00:00:00", "end_date": "2024-03-02"},
        {"date": "2024-03-01", "start_date": "2024-03-01", "end_date": "2024-03-02"},
        {"page": 0},
        {"page_size": 1001},
        {"page_size": 0},
        {"page": "2"},
        {"page": None},
        {"page": True},
        {"page_size": False},
        {"page_size": 1.5},
    ],
)
async def test_invalid_requests_fail_instead_of_falling_back_to_today(step, params):
    """Malformed or conflicting queries must not quietly list today's notes."""
    await step(**params)
    assert not step.context.response.success
    assert "Error:" in step.context.response.answer


@pytest.mark.asyncio
async def test_default_range_page_is_bounded_but_legacy_calls_return_every_note(step, tmp_path):
    """Omitted pagination must not truncate existing single-day callers."""
    for index in range(101):
        note(tmp_path, "2024-03-01", f"note-{index:03}")
    await step(date="2024-03-01")
    assert step.context.response.metadata["count"] == 101
    assert set(step.context.response.metadata) == {"date", "count", "notes"}
    await step(start_date="2024-03-01", end_date="2024-03-01")
    result = step.context.response.metadata
    assert result["count"] == result["page_size"] == 100
    assert result["total_notes"] == 101 and result["total_pages"] == 2
    await step(start_date="2024-03-01", end_date="2024-03-01", page=2)
    assert step.context.response.metadata["notes"][0]["name"] == "note-100"


@pytest.mark.asyncio
async def test_empty_range_is_read_only_and_has_no_pages(step, tmp_path):
    """A query must not create missing date directories or indexes."""
    await step(start_date="0001-01-01", end_date="9999-12-31")
    assert step.context.response.success
    assert step.context.response.metadata["total_notes"] == 0
    assert step.context.response.metadata["total_pages"] == 0
    assert step.context.response.metadata["notes"] == []
    assert not (tmp_path / "daily").exists()


@pytest.mark.asyncio
async def test_range_uses_custom_directory_and_skips_indexes_invalid_dates_and_nested_files(step, tmp_path):
    """Scan existing valid date directories, never every date or nested content."""
    step.app_context.app_config.daily_dir = "journal/days"
    note(tmp_path, "2024-03-01", "kept", daily_dir="journal/days")
    note(tmp_path, "not-a-date", "ignored", daily_dir="journal/days")
    note(tmp_path, "2024-W09-5", "week-date", daily_dir="journal/days")
    note(tmp_path, "2024-03-01/nested", "ignored", daily_dir="journal/days")
    note(tmp_path, "2024-03-01", "other-root")
    index = tmp_path / "journal/days/2024-03-01.md"
    index.write_text("day index", encoding="utf-8")
    (tmp_path / "journal/days/2024-03-02").write_text("not a directory", encoding="utf-8")
    await step(start_date="0001-01-01", end_date="9999-12-31")
    assert step.context.response.metadata["notes"] == [{"name": "kept", "path": "journal/days/2024-03-01/kept.md"}]


@pytest.mark.asyncio
async def test_permissions_filter_before_reading_and_before_counting(step, tmp_path, monkeypatch):
    """An exact-file scope must hide other dates and sibling frontmatter."""
    allowed = note(tmp_path, "2024-03-01", "allowed")
    denied = {note(tmp_path, "2024-03-01", "sibling"), note(tmp_path, "2024-02-29", "other-day")}
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path in denied:
            pytest.fail("permission-denied note was read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    for params in ({"date": "2024-03-01"}, {"start_date": "2024-02-29", "end_date": "2024-03-01"}):
        await step(**params, _allowed_paths=[str(allowed.relative_to(tmp_path))])
        result = step.context.response.metadata
        assert result["count"] == 1
        assert result["notes"] == [{"name": "allowed", "path": "daily/2024-03-01/allowed.md"}]
        assert result.get("total_notes", 1) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [[], {}, 42, ["../outside"], ["daily", "../outside"]])
async def test_invalid_permission_constraints_fail_closed(step, tmp_path, scope):
    """An invalid scope cannot become an unrestricted listing."""
    note(tmp_path, "2024-03-01", "hidden")
    await step(start_date="2024-03-01", end_date="2024-03-01", _allowed_paths=scope)
    assert step.context.response.metadata["notes"] == []
    assert step.context.response.metadata["total_notes"] == 0


@pytest.mark.asyncio
async def test_directory_scope_counts_only_authorized_notes(step, tmp_path):
    """Filtering happens before pagination and totals, including directory scopes."""
    note(tmp_path, "2024-03-01", "a")
    note(tmp_path, "2024-03-01", "b")
    note(tmp_path, "2024-02-29", "hidden")
    await step(start_date="2024-02-29", end_date="2024-03-01", page_size=1, _allowed_paths=["daily/2024-03-01"])
    result = step.context.response.metadata
    assert result["total_notes"] == result["total_pages"] == 2
    assert result["notes"][0]["name"] == "a"


@pytest.mark.asyncio
async def test_symlink_escapes_and_loops_do_not_leak_notes(step, tmp_path):
    """Both date-directory links and note links retain workspace containment."""
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.md").write_text("---\nname: secret\n---\nprivate", encoding="utf-8")
    kept = note(tmp_path, "2024-03-01", "kept")
    try:
        (kept.parent / "escape.md").symlink_to(outside / "secret.md")
        (tmp_path / "daily/2024-02-29").symlink_to(outside, target_is_directory=True)
        (tmp_path / "daily/2024-03-02").symlink_to("2024-03-02", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    await step(start_date="2024-02-29", end_date="2024-03-02")
    assert step.context.response.metadata["notes"] == [{"name": "kept", "path": "daily/2024-03-01/kept.md"}]


@pytest.mark.asyncio
async def test_queries_preserve_note_and_index_contents_and_mtimes(step, tmp_path):
    """Listing never reindexes, rewrites frontmatter, or touches user files."""
    note(tmp_path, "2024-03-01", "a")
    note(tmp_path, "2024-02-29", "b")
    index = tmp_path / "daily/2024-03-01.md"
    index.write_text("handwritten index\n", encoding="utf-8")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "daily").rglob("*") if p.is_file()}
    await step(start_date="2024-02-29", end_date="2024-03-01", page_size=1)
    await step(date="2024-03-01", page=2, page_size=1)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "daily").rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.asyncio
async def test_invalid_or_unreadable_notes_do_not_count_toward_pages(step, tmp_path, monkeypatch):
    """Match the existing scanner's skip behavior for malformed and unreadable notes."""
    good = note(tmp_path, "2024-03-01", "good")
    malformed = note(tmp_path, "2024-03-01", "bad")
    malformed.write_text("---\nlabels: [\n---\n", encoding="utf-8")
    unreadable = note(tmp_path, "2024-03-01", "unreadable")
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == unreadable:
            raise PermissionError("unreadable note")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    await step(start_date="2024-03-01", end_date="2024-03-01", page_size=1)
    assert step.context.response.metadata["total_notes"] == 1
    assert step.context.response.metadata["notes"][0]["path"] == good.relative_to(tmp_path).as_posix()


@pytest.mark.asyncio
async def test_scanning_stays_off_the_event_loop(step, tmp_path, monkeypatch):
    """File parsing uses the event loop's managed executor, not the caller's thread."""
    from threading import current_thread
    import frontmatter

    note(tmp_path, "2024-03-01", "a")
    original = frontmatter.loads
    caller = current_thread().ident
    threads = []

    def capture_thread(*args, **kwargs):
        threads.append(current_thread().ident)
        return original(*args, **kwargs)

    monkeypatch.setattr(frontmatter, "loads", capture_thread)
    await step(start_date="2024-03-01", end_date="2024-03-01")
    assert threads and all(thread != caller for thread in threads)


@pytest.mark.asyncio
async def test_registered_job_and_http_endpoint_use_the_updated_schema(step, tmp_path):
    """Exercise the configured Job and HTTP boundary rather than only a Step helper."""
    import httpx
    import yaml

    from reme.components.job import BaseJob
    from reme.components.service import HttpService, MCPService
    from reme.enumeration import ComponentEnum

    default = yaml.safe_load((Path(__file__).parents[2] / "reme/config/default.yaml").read_text(encoding="utf-8"))
    spec = default["jobs"]["daily_list"]
    properties = spec["parameters"]["properties"]
    assert {"date", "start_date", "end_date", "page", "page_size"} == set(properties)
    assert "default" not in properties["page"] and "default" not in properties["page_size"]
    assert properties["page_size"]["maximum"] == 1000
    note(tmp_path, "2024-02-29", "old")
    note(tmp_path, "2024-03-01", "new")
    context = step.app_context
    context.components[ComponentEnum.FILE_STORE] = {"default": step.file_store}
    job = BaseJob(name="daily_list", app_context=context, **spec)
    await job.start()
    try:
        response = await job(start_date="2024-02-29", end_date="2024-03-01", page_size=1)
        assert response.success and response.metadata["notes"][0]["name"] == "new"
        service = HttpService(mcp_enabled=False, web_enabled=False)
        service.build_service(SimpleNamespace(config=context.app_config))
        service.add_job(job)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.service),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/daily_list",
                json={"start_date": "2024-02-29", "end_date": "2024-03-01", "page_size": 1},
            )
            assert response.status_code == 200
            assert response.json()["metadata"]["total_notes"] == 2
            response = await client.post("/daily_list", json={"date": "2024-03-01"})
            assert set(response.json()["metadata"]) == {"date", "count", "notes"}
            response = await client.post("/daily_list", json={"start_date": "2024-03-01"})
            assert not response.json()["success"]
        mcp = MCPService()
        mcp.build_service(SimpleNamespace(config=context.app_config))
        mcp.add_job(job)
        tool = await mcp.service.get_tool("daily_list")
        result = await tool.run({"start_date": "2024-02-29", "end_date": "2024-03-01", "page_size": 1})
        assert "total_notes: 2" in str(result.content)
        assert "daily/2024-03-01/new.md" in str(result.content)
        assert "daily/2024-02-29/old.md" not in str(result.content)
        legacy = await tool.run({"date": "2024-03-01"})
        assert "total_notes:" not in str(legacy.content)
    finally:
        await job.close()
