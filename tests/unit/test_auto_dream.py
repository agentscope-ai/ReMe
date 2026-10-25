"""Unit tests for the refactored dream package."""

# pylint: disable=protected-access

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import frontmatter
import pytest
import yaml

from reme.components.application_context import ApplicationContext
from reme.components.as_llm import BaseAsLLM
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.file_catalog import BaseFileCatalog
from reme.components.file_store import BaseFileStore
from reme.components.job import BaseJob
from reme.components.runtime_context import RuntimeContext
from reme.components.tag_index import LocalTagIndex
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.schema import DreamState, FileNode
from reme.steps.evolve.auto_tag import AutoTagStep
from reme.steps.evolve.dream.extract import DreamExtractStep
from reme.steps.evolve.dream.finish import DreamFinishStep
from reme.steps.evolve.dream.integrate import DreamIntegrateStep, _snapshot_digest
from reme.steps.evolve.dream.utils import parse_structured_reply, recent_dates, scan_day_files
from reme.steps.file_io.frontmatter_update import FrontmatterUpdateStep

_MALFORMED_EXTRACT_REPLY = (
    "units:\n- name: checkout\n  bucket: procedure\n"
    "  summary: CI must add `with: submodules: recursive`\n"
    "  paths: [daily/2026-05-28/session.md]\n"
)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _Catalog(BaseFileCatalog):
    def __init__(self):
        super().__init__()
        self.upserts = []
        self.dumps = 0

    async def upsert(self, nodes):
        self.upserts.extend(nodes)

    async def delete(self, path):
        return None

    async def get_nodes(self, paths=None):
        return []

    async def dump(self):
        self.dumps += 1


class _FileStore(BaseFileStore):
    def __init__(self, workspace: Path):
        super().__init__()
        self._workspace_path = workspace

    @property
    def workspace_path(self) -> Path:
        return self._workspace_path

    async def upsert(self, files):
        return None

    async def delete(self, path):
        return None

    async def clear(self):
        return None

    async def get_nodes(self, paths=None):
        return []

    async def get_outlinks(self, path, scope=None):
        return []

    async def get_inlinks(self, path, scope=None):
        return []

    async def vector_search(self, query, limit, search_filter):
        return []

    async def keyword_search(self, query, limit, search_filter):
        return []


class _ReplyAgent(BaseAgentWrapper):
    """Small agent double that can return, mutate files, or raise."""

    def __init__(self, result=None, *, on_reply=None, error=None):
        super().__init__()
        self.result = result or {"result": "{}"}
        self.on_reply = on_reply
        self.error = error
        self.calls = 0

    async def reply(self, _message, **_kwargs):
        self.calls += 1
        if self.on_reply:
            self.on_reply()
        if self.error:
            raise self.error
        return self.result


class _SequenceAgent(BaseAgentWrapper):
    """Return a fixed sequence of replies or exceptions."""

    def __init__(self, *outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests = []

    async def reply(self, _message, **_kwargs):
        self.calls += 1
        self.requests.append((_message, _kwargs))
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_scan_day_files_includes_only_markdown_day_files():
    """Scan day indexes and nested Markdown notes without including YAML products."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        _touch(workspace / "daily" / "2026-05-28.md")
        _touch(workspace / "daily" / "2026-05-28" / "session.md")
        _touch(workspace / "daily" / "2026-05-28" / "auth-refactor" / "notes.md")
        _touch(workspace / "daily" / "2026-05-28" / "interests.yaml")

        assert scan_day_files(workspace, "2026-05-28", "daily") == [
            "daily/2026-05-28.md",
            "daily/2026-05-28/auth-refactor/notes.md",
            "daily/2026-05-28/session.md",
        ]


def test_dream_extract_matches_posix_catalog_paths(tmp_path):
    """Unchanged nested files retain their POSIX catalog entries on every platform."""

    class Catalog(_Catalog):
        """Catalog seeded with POSIX paths and recording deletions."""

        def __init__(self, nodes):
            super().__init__()
            self.nodes = nodes
            self.deleted = []

        async def delete(self, path):
            self.deleted.extend(path if isinstance(path, list) else [path])

        async def get_nodes(self, paths=None):
            return self.nodes

    async def run():
        note = _touch(tmp_path / "daily" / "2026-05-28" / "nested" / "session.md")
        rel_path = note.relative_to(tmp_path).as_posix()
        catalog = Catalog([FileNode(path=rel_path, st_mtime=note.stat().st_mtime)])
        step = DreamExtractStep(scan_days=1, app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        with patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}):
            response = await step(
                RuntimeContext(date="2026-05-28", file_catalog=catalog, file_store=_FileStore(tmp_path)),
            )

        dream = response.metadata["dream"]
        assert response.success is True
        assert dream["unchanged_paths"] == [rel_path]
        assert dream["changed_paths"] == []
        assert dream["deleted_paths"] == []
        assert not catalog.deleted

    asyncio.run(run())


def test_dream_extract_removes_all_legacy_interests_entries_from_catalog(tmp_path):
    """Auto Dream removes historical interests watermarks without touching exposure files."""

    class Catalog(_Catalog):
        """Catalog seeded with a legacy interests entry."""

        def __init__(self, nodes):
            super().__init__()
            self.nodes = nodes
            self.deleted = []

        async def delete(self, path):
            self.deleted.extend(path if isinstance(path, list) else [path])

        async def get_nodes(self, paths=None):
            return self.nodes

    async def run():
        interests = _touch(tmp_path / "daily" / "2026-05-28" / "interests.yaml", "topics: []\n")
        old_interests = _touch(tmp_path / "daily" / "2026-01-01" / "interests.yaml", "topics: []\n")
        rel_path = interests.relative_to(tmp_path).as_posix()
        old_rel_path = old_interests.relative_to(tmp_path).as_posix()
        catalog = Catalog(
            [
                FileNode(path=rel_path, st_mtime=interests.stat().st_mtime),
                FileNode(path=old_rel_path, st_mtime=old_interests.stat().st_mtime),
            ],
        )
        step = DreamExtractStep(scan_days=1, app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        with patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}):
            response = await step(
                RuntimeContext(date="2026-05-28", file_catalog=catalog, file_store=_FileStore(tmp_path)),
            )

        assert response.success is True
        assert response.metadata["dream"]["deleted_paths"] == [old_rel_path, rel_path]
        assert catalog.deleted == [old_rel_path, rel_path]
        assert interests.read_text(encoding="utf-8") == "topics: []\n"
        assert old_interests.read_text(encoding="utf-8") == "topics: []\n"

    asyncio.run(run())


def test_recent_dates_includes_anchor_and_previous_days():
    """Recent date window is inclusive and chronological."""
    assert recent_dates("2026-05-28", 3) == ["2026-05-26", "2026-05-27", "2026-05-28"]
    assert recent_dates("2026-05-28", 1) == ["2026-05-28"]


def test_parse_structured_reply_handles_fenced_yaml_and_scalar_fallback():
    """Parse a JSON/YAML object from an agent reply, including fenced blocks."""
    data = parse_structured_reply(
        "```yaml\n"
        "action: REFINE\n"
        "target_path: digest/personal/no-trailing-summary.md\n"
        "note: Extended node. Core rule unchanged: answer directly and stop.\n"
        "```",
    )
    assert data["action"] == "REFINE"
    assert data["target_path"] == "digest/personal/no-trailing-summary.md"
    assert data["note"].startswith("Extended node")


def test_extract_parser_preserves_yaml_diagnostics_and_accepts_quoted_colons():
    """Strict extraction rejects the reported YAML while legacy parsing stays compatible."""
    assert parse_structured_reply(_MALFORMED_EXTRACT_REPLY) == {}
    with pytest.raises(ValueError, match="ScannerError") as exc:
        parse_structured_reply(_MALFORMED_EXTRACT_REPLY, strict=True)
    assert "line 4" in str(exc.value)
    assert "submodules: recursive" in str(exc.value)
    quoted = _MALFORMED_EXTRACT_REPLY.replace(
        "summary: CI must add `with: submodules: recursive`",
        'summary: "CI must add `with: submodules: recursive`"',
    )
    assert len(parse_structured_reply(quoted, strict=True)["units"]) == 1
    assert len(parse_structured_reply(f"```yaml\n{quoted}```", strict=True)["units"]) == 1
    assert parse_structured_reply("{}", strict=True) == {}
    for reply in ("", "not an object", "[]"):
        with pytest.raises(ValueError, match="expected a JSON/YAML object"):
            parse_structured_reply(reply, strict=True)


def test_integrate_prompts_require_wikilinks_in_contextual_prose():
    """Every digest bucket rejects standalone relation fields and bare links."""
    prompt_path = Path(__file__).parents[2] / "reme" / "steps" / "evolve" / "dream" / "integrate.yaml"
    prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

    for bucket in ("procedure", "personal", "wiki"):
        english = prompts[f"integrate_system_prompt_{bucket}"]
        chinese = prompts[f"integrate_system_prompt_{bucket}_zh"]
        assert "Never emit standalone relation fields or bare-link lines" in english
        assert "Digest links belong inside the sentence" in english
        assert "Before returning, read the target" in english
        assert "禁止生成 `relates_to:: [[...]]`" in chinese
        assert "Digest 链接必须融入解释关系的句子" in chinese
        assert "返回前读取目标文件" in chinese


def test_extract_clean_output_respects_max_units():
    """Extract cleaning caps valid units at max_units."""
    state = DreamState(changed_paths=["daily/a.md"])
    meta = {
        "units": [
            {"name": f"unit-{i}", "bucket": "wiki", "summary": f"summary {i}", "paths": ["daily/a.md"]}
            for i in range(7)
        ],
    }

    DreamExtractStep().clean_output(state, meta, max_units=5)

    assert len(state.units) == 5
    assert [unit["name"] for unit in state.units] == [f"unit-{i}" for i in range(5)]


@pytest.mark.parametrize("field", ["units", "memory_units"])
@pytest.mark.parametrize(
    ("raw_bucket", "expected_bucket"),
    [
        (" procedure ", "procedure"),
        ("\tpersonal\n", "personal"),
        (" wiki ", "wiki"),
        (" unknown ", "wiki"),
        (" \t\n ", "wiki"),
    ],
)
def test_extract_validation_strips_bucket_whitespace_before_fallback(field, raw_bucket, expected_bucket):
    """Padded known buckets retain their destination; unknown and blank values still use wiki."""
    state = DreamState(changed_paths=["daily/a.md"])
    unit = {"name": "unit", "bucket": raw_bucket, "summary": "memory", "paths": ["daily/a.md"]}

    DreamExtractStep().clean_output(state, {field: [unit]})

    assert state.units == [{**unit, "bucket": expected_bucket}]


def test_extract_validation_preserves_legacy_alias_and_unknown_bucket_fallback():
    """Validation retains the documented bucket fallback and source-path cleanup."""
    state = DreamState(changed_paths=["daily/a.md"])
    DreamExtractStep().clean_output(
        state,
        {
            "memory_units": [
                {
                    "name": " unit ",
                    "bucket": "unknown",
                    "summary": " memory ",
                    "paths": ["daily/a.md", "daily/a.md", "outside.md"],
                },
            ],
        },
    )
    assert state.units == [{"name": "unit", "bucket": "wiki", "summary": "memory", "paths": ["daily/a.md"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("job_name", ["auto_dream", "dream_cron"])
@pytest.mark.parametrize(
    ("reply", "succeeds"),
    [
        (_MALFORMED_EXTRACT_REPLY, False),
        ("{}", False),
        ('{"units": null}', False),
        ('{"units": {}}', False),
        ('{"units": [42]}', False),
        ('{"units": [{"name": "missing-fields"}]}', False),
        (
            '{"units": [{"name": "x", "summary": " ", "bucket": "wiki", "paths": ["daily/2026-05-28/session.md"]}]}',
            False,
        ),
        ('{"units": [{"name": "x", "summary": "x", "bucket": "wiki", "paths": ["outside.md"]}]}', False),
        (
            '{"units": [{"name": "valid", "summary": "x", "bucket": "wiki", '
            '"paths": ["daily/2026-05-28/session.md"]}, '
            '{"name": "invalid", "summary": "x", "bucket": "wiki", "paths": []}]}',
            False,
        ),
        ('{"units": []}', True),
        ("units: []", True),
        ("memory_units: []", True),
    ],
)
async def test_dream_extraction_failure_preserves_changed_inputs(tmp_path, monkeypatch, job_name, reply, succeeds):
    """Invalid receipts fail both configured pipelines without checkpointing notes or day indexes."""

    class Catalog(_Catalog):
        """Expose versions checkpointed by Finish to the next job invocation."""

        async def get_nodes(self, paths=None):
            return self.upserts

    note = _touch(tmp_path / "daily/2026-05-28/session.md", "CI needs submodules: recursive")
    _touch(tmp_path / "daily/2026-05-28.md", "day index")
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    catalog, agent = Catalog(), _ReplyAgent({"result": reply})
    config = resolve_app_config(config="default", log_config=False)["jobs"][job_name]
    job = BaseJob(name=job_name, steps=config["steps"], app_context=app_context)
    monkeypatch.setattr("reme.steps.evolve.dream.extract.refresh_day_index", AsyncMock(return_value={}))
    monkeypatch.setattr("reme.steps.evolve.dream.extract.llm_available", lambda _step: True)
    await job.start()
    try:
        inputs = {
            "date": "2026-05-28",
            "scan_days": 1,
            "agent_wrapper": agent,
            "file_catalog": catalog,
            "file_store": _FileStore(tmp_path),
        }
        response = await job(**inputs)
        dream = response.metadata["dream"]
        assert response.success is succeeds
        assert response.answer.startswith("AutoDream completed" if succeeds else "AutoDream failed")
        assert dream["units"] == []
        assert dream["files_changed"] == 2
        assert dream["failed_paths"] == ([] if succeeds else dream["changed_paths"])
        assert bool(dream["errors"]) is not succeeds
        assert set(dream["checkpoint_paths"]) == (set(dream["changed_paths"]) if succeeds else set())
        assert catalog.dumps == int(succeeds)
        assert agent.calls == (1 if succeeds else 2)
        assert note.read_text(encoding="utf-8") == "CI needs submodules: recursive"

        agent.result = {"result": '{"units": []}'}
        response = await job(**inputs)
        assert response.success is True
        assert response.metadata["dream"]["files_changed"] == (0 if succeeds else 2)
        assert agent.calls == (1 if succeeds else 3)
        assert catalog.dumps == (2 if succeeds else 1)
    finally:
        await job.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize(
    ("first_reply", "error"),
    [(_MALFORMED_EXTRACT_REPLY, "ScannerError"), ('{"units": [{"name": "missing-fields"}]}', "ValidationError")],
)
async def test_extract_retry_includes_parser_error_and_recovers_json(tmp_path, language, first_reply, error):
    """Both localized prompts recover code-like strings using JSON and concrete error feedback."""
    source = "daily/2026-05-28/session.md"
    _touch(tmp_path / source)
    unit = {
        "name": "checkout",
        "bucket": "procedure",
        "summary": 'CI needs `submodules: recursive`, "quoted" text, and \\paths.\nKeep dependencies.',
        "paths": [source],
    }
    agent = _SequenceAgent({"result": first_reply}, {"result": json.dumps({"units": [unit]})})
    step = DreamExtractStep(
        scan_days=1,
        language=language,
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        agent_wrapper=agent,
    )
    with (
        patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}),
        patch("reme.steps.evolve.dream.extract.llm_available", return_value=True),
    ):
        response = await step(
            RuntimeContext(date="2026-05-28", file_catalog=_Catalog(), file_store=_FileStore(tmp_path)),
        )

    assert response.success is True
    assert response.metadata["dream"]["units"] == [unit]
    assert not response.metadata["dream"]["errors"]
    first_message, first_kwargs = agent.requests[0]
    retry_message, retry_kwargs = agent.requests[1]
    assert retry_message.startswith(first_message)
    assert error in retry_message
    if error == "ScannerError":
        assert "line 4" in retry_message
    else:
        assert "Field required" in retry_message
    assert "key: value" in retry_message
    assert retry_kwargs["system_prompt"] == first_kwargs["system_prompt"]
    system_prompt = first_kwargs["system_prompt"]
    assert "JSON schema" in system_prompt
    assert "YAML or JSON" not in system_prompt
    example = json.loads(system_prompt.splitlines()[-1])
    assert "submodules: recursive" in example["units"][0]["summary"]


@pytest.mark.asyncio
async def test_extract_retries_one_unusable_receipt(tmp_path):
    """A transient malformed extraction receipt gets one bounded retry."""

    _touch(tmp_path / "daily" / "2026-05-28" / "session.md")
    agent = _SequenceAgent({"result": "{}"}, {"result": '{"units": []}'})
    step = DreamExtractStep(
        scan_days=1,
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        agent_wrapper=agent,
    )

    with (
        patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}),
        patch("reme.steps.evolve.dream.extract.llm_available", return_value=True),
    ):
        response = await step(
            RuntimeContext(date="2026-05-28", file_catalog=_Catalog(), file_store=_FileStore(tmp_path)),
        )

    assert response.success is True
    assert response.metadata["dream"]["warnings"] == []
    assert agent.calls == 2


@pytest.mark.parametrize(
    "receipt",
    ["{}", "not a receipt", '{"action": "CREATE", "target_path": "digest/wiki/missing.md"}'],
)
@pytest.mark.asyncio
async def test_integrate_invalid_receipt_without_file_change_fails(tmp_path, receipt):
    """An unusable integration receipt cannot acknowledge unprocessed source files."""

    unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
    state = DreamState(units=[unit])
    step = DreamIntegrateStep()
    step.agent_wrapper = _ReplyAgent({"result": receipt})

    await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

    assert state.integrate_results == []
    assert state.skipped_units == []
    assert len(state.failed_units) == 1
    assert state.failed_units[0]["name"] == unit["name"]
    assert state.failed_paths == unit["paths"]
    assert len(state.errors) == 1
    assert state.warnings == []
    assert step.agent_wrapper.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_source", [False, True])
async def test_integrate_failure_only_blocks_affected_sources(tmp_path, monkeypatch, shared_source):
    """A failed unit prevents checkpointing all its sources, even one used by a successful unit."""
    good, bad, target = "daily/good.md", "daily/bad.md", "digest/wiki/good.md"
    observed = {rel: _touch(tmp_path / rel).stat().st_mtime for rel in (good, bad)}
    failed_paths = [bad, good] if shared_source else [bad]
    units = [
        {"name": "good", "bucket": "wiki", "summary": "good memory", "paths": [good]},
        {"name": "bad", "bucket": "wiki", "summary": "bad memory", "paths": failed_paths},
    ]

    def integrate_good():
        _touch(tmp_path / target, "integrated memory")
        return {"result": json.dumps({"action": "CREATE", "target_path": target})}

    agent = _SequenceAgent(integrate_good, {"result": "{}"}, {"result": "{}"})
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    state = DreamState(units=units, changed_paths=[good, bad], existing=observed, workspace=str(tmp_path))
    catalog = _Catalog()
    context = RuntimeContext(dream=state.model_dump(), file_catalog=catalog, file_store=_FileStore(tmp_path))
    monkeypatch.setattr("reme.steps.evolve.dream.integrate.llm_available", lambda _step: True)

    response = await DreamIntegrateStep(app_context=app_context, agent_wrapper=agent)(context)
    assert response.success is False
    response = await DreamFinishStep(app_context=app_context)(context)

    dream = response.metadata["dream"]
    assert response.success is False
    assert len(dream["integrate_results"]) == 1
    assert len(dream["failed_units"]) == 1
    assert set(dream["failed_paths"]) == set(failed_paths)
    assert dream["checkpoint_paths"] == ([] if shared_source else [good])
    assert {node.path for node in catalog.upserts} == (set() if shared_source else {good})
    assert (tmp_path / target).read_text(encoding="utf-8") == "integrated memory"
    assert agent.calls == 3


@pytest.mark.asyncio
async def test_integrate_retries_one_invalid_receipt(tmp_path):
    """A transient invalid integration receipt gets one bounded retry."""

    unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
    state = DreamState(units=[unit])
    target = _touch(tmp_path / "digest" / "wiki" / "unit.md", "integrated")
    agent = _SequenceAgent(
        {"result": "{}"},
        {
            "result": ('{"action": "REFINE", "target_path": "digest/wiki/unit.md", ' '"note": "updated after retry"}'),
        },
    )
    step = DreamIntegrateStep(agent_wrapper=agent)

    with patch(
        "reme.steps.evolve.dream.integrate._snapshot_digest",
        wraps=_snapshot_digest,
    ) as snapshot:
        await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

    assert target.is_file()
    assert state.integrate_results[0]["target_path"] == "digest/wiki/unit.md"
    assert state.skipped_units == []
    assert agent.calls == 2
    assert snapshot.call_count == 3


@pytest.mark.asyncio
async def test_integrate_invalid_receipt_recovers_one_changed_digest_file(tmp_path):
    """The file-native side effect wins when the agent's final receipt is malformed."""

    unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
    state = DreamState(units=[unit])
    target = tmp_path / "digest" / "wiki" / "unit.md"

    def write_digest():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("integrated", encoding="utf-8")

    step = DreamIntegrateStep()
    step.agent_wrapper = _ReplyAgent(on_reply=write_digest)

    await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

    assert state.skipped_units == []
    assert state.failed_units == []
    assert state.integrate_results[0]["action"] == "CREATE"
    assert state.integrate_results[0]["target_path"] == "digest/wiki/unit.md"
    assert state.nodes_created == ["digest/wiki/unit.md"]
    assert state.modified_paths == ["digest/wiki/unit.md"]
    assert len(state.warnings) == 1


@pytest.mark.asyncio
async def test_integrate_recovers_an_update_to_another_bucket(tmp_path):
    """A malformed receipt can recover an update to an existing cross-bucket node."""

    unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
    state = DreamState(units=[unit])
    target = tmp_path / "digest" / "procedure" / "unit.md"
    _touch(target, "existing procedure")

    def write_digest():
        target.write_text("updated procedure", encoding="utf-8")

    step = DreamIntegrateStep()
    step.agent_wrapper = _ReplyAgent(on_reply=write_digest)

    await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

    assert state.skipped_units == []
    assert state.integrate_results[0]["action"] == "UPDATED"
    assert state.integrate_results[0]["target_path"] == "digest/procedure/unit.md"
    assert state.nodes_updated == ["digest/procedure/unit.md"]
    assert state.modified_paths == ["digest/procedure/unit.md"]
    assert step.agent_wrapper.calls == 1


@pytest.mark.asyncio
async def test_integrate_does_not_recover_a_create_in_another_bucket(tmp_path):
    """A malformed receipt cannot recover a cross-bucket create."""

    unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
    state = DreamState(units=[unit])
    target = tmp_path / "digest" / "procedure" / "unit.md"

    def write_digest():
        _touch(target, "created procedure")

    step = DreamIntegrateStep(agent_wrapper=_ReplyAgent(on_reply=write_digest))

    await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

    assert state.integrate_results == []
    assert state.skipped_units == []
    assert len(state.failed_units) == 1
    assert state.failed_paths == unit["paths"]
    assert state.errors
    assert state.nodes_created == []
    assert state.modified_paths == ["digest/procedure/unit.md"]


@pytest.mark.asyncio
async def test_integrate_accepts_cross_bucket_update_receipt(tmp_path):
    """An UPDATE receipt may target an existing node in any supported bucket."""

    unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
    state = DreamState(units=[unit])
    _touch(tmp_path / "digest" / "procedure" / "unit.md", "updated procedure")
    agent = _ReplyAgent(
        result={
            "result": (
                '{"action": "REFINE", "target_path": "digest/procedure/unit.md", '
                '"note": "updated existing procedure"}'
            ),
        },
    )
    step = DreamIntegrateStep(agent_wrapper=agent)

    with patch(
        "reme.steps.evolve.dream.integrate._snapshot_digest",
        wraps=_snapshot_digest,
    ) as snapshot:
        await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

    assert state.skipped_units == []
    assert state.integrate_results[0]["action"] == "REFINE"
    assert state.nodes_updated == ["digest/procedure/unit.md"]
    assert state.modified_paths == []
    assert snapshot.call_count == 2


def test_integrate_uses_one_application_wide_lock(tmp_path):
    """Concurrent AutoDream jobs in one Application serialize integration."""
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    first = DreamIntegrateStep(app_context=app_context)
    second = DreamIntegrateStep(app_context=app_context)

    assert first._integration_lock() is second._integration_lock()  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_integrate_emits_actual_changes_once_across_units(tmp_path, monkeypatch):
    """Creation followed by updates stays added; receipt-only updates emit nothing."""
    created = "digest/wiki/created.md"
    updated = "digest/procedure/updated.md"
    untouched = "digest/personal/untouched.md"
    _touch(tmp_path / updated, "old")
    _touch(tmp_path / untouched, "unchanged")

    def create():
        _touch(tmp_path / created, "new")
        return {"result": json.dumps({"action": "CREATE", "target_path": created})}

    def update():
        _touch(tmp_path / created, "new with more evidence")
        _touch(tmp_path / updated, "updated existing memory")
        return {"result": json.dumps({"action": "REFINE", "target_path": created})}

    agent = _SequenceAgent(
        create,
        update,
        {"result": json.dumps({"action": "CORROBORATE", "target_path": untouched})},
    )
    state = DreamState(
        units=[{"name": str(i), "bucket": "wiki", "paths": ["daily/source.md"]} for i in range(3)],
    )
    context = RuntimeContext(dream=state.model_dump(), file_store=_FileStore(tmp_path))
    step = DreamIntegrateStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), agent_wrapper=agent)
    monkeypatch.setattr("reme.steps.evolve.dream.integrate.llm_available", lambda _step: True)

    await step(context)

    assert context.response.success is True
    assert context["changes"] == [
        {"change": "modified", "path": updated},
        {"change": "added", "path": created},
    ]
    assert untouched in context.response.metadata["dream"]["nodes_updated"]
    assert untouched not in context.response.metadata["dream"]["modified_paths"]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_error", [False, True])
async def test_integrate_keeps_added_changes_across_retry_recovery(tmp_path, monkeypatch, agent_error):
    """Files from both attempts remain eligible for tagging after receipt recovery."""
    first, second = "digest/wiki/first.md", "digest/wiki/second.md"

    def attempt_one():
        _touch(tmp_path / first, "first")
        _touch(tmp_path / second, "second")
        return RuntimeError("agent failed") if agent_error else {"result": "{}"}

    def attempt_two():
        _touch(tmp_path / first, "first with more evidence")
        return RuntimeError("agent failed") if agent_error else {"result": "{}"}

    agent = _SequenceAgent(attempt_one, attempt_two)
    state = DreamState(units=[{"name": "unit", "bucket": "wiki", "paths": ["daily/source.md"]}])
    context = RuntimeContext(dream=state.model_dump(), file_store=_FileStore(tmp_path))
    step = DreamIntegrateStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), agent_wrapper=agent)
    monkeypatch.setattr("reme.steps.evolve.dream.integrate.llm_available", lambda _step: True)

    await step(context)

    assert agent.calls == 2
    assert context.response.success is True
    assert context.response.metadata["dream"]["warnings"]
    assert context["changes"] == [{"change": "added", "path": path} for path in (first, second)]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_units", [False, True])
async def test_integrate_clears_changes_when_skipping_or_missing_llm(tmp_path, monkeypatch, with_units):
    """Early returns cannot pass caller-supplied paths into AutoTag."""
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    state = DreamState(units=[{"name": "unit", "paths": ["daily/source.md"]}] if with_units else [])
    agent = _ReplyAgent()
    context = RuntimeContext(dream=state.model_dump(), changes=[{"change": "added", "path": "daily/source.md"}])
    monkeypatch.setattr("reme.steps.evolve.dream.integrate.llm_available", lambda _step: False)

    await DreamIntegrateStep(app_context=app_context, agent_wrapper=agent)(context)
    response = await AutoTagStep(app_context=app_context, agent_wrapper=agent)(context)

    assert context["changes"] == []
    assert response.success is not with_units
    assert response.metadata["auto_tag"]["processed"] == 0
    assert agent.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("job_name", ["auto_dream", "dream_cron"])
@pytest.mark.parametrize("tag_fails", [False, True])
@pytest.mark.parametrize("integration_failure", [None, "exception", "invalid_receipt"])
async def test_dream_jobs_tag_outputs_and_preserve_checkpoint_results(
    tmp_path,
    monkeypatch,
    job_name,
    tag_fails,
    integration_failure,
):
    """Both configured pipelines tag durable outputs without changing dream success."""
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    catalog, store = _Catalog(), _FileStore(tmp_path)
    store.tag_index = LocalTagIndex()
    source = "daily/2026-09-11/source.md"
    targets = ["digest/wiki/first.md", "digest/wiki/second.md"]
    _touch(tmp_path / source, "Source about ReMe")
    unit = {"name": "unit", "bucket": "wiki", "summary": "ReMe memory", "paths": [source]}

    async def reply(_message, **kwargs):
        if kwargs["job_tools"] == ["read"]:
            return {"result": json.dumps({"units": [unit]})}
        if "list_tags" in kwargs["job_tools"]:
            injected = kwargs["injected_job_kwargs"]
            path = injected["_allowed_paths"][0]
            if tag_fails and path == targets[0]:
                raise RuntimeError("tagging unavailable")
            update_context = RuntimeContext(
                path=path,
                metadata={"memory_tags": ["ReMe"]},
                **injected,
            )
            await FrontmatterUpdateStep(file_store=store)(update_context)
            assert update_context.response.success
            return {"result": "Tagged ReMe"}
        for path in targets:
            previous = (tmp_path / path).read_text(encoding="utf-8") if (tmp_path / path).exists() else ""
            _touch(tmp_path / path, previous + "ReMe evidence\n")
        if integration_failure == "exception":
            raise RuntimeError("integration unavailable")
        if integration_failure == "invalid_receipt":
            return {"result": "{}"}
        return {"result": json.dumps({"action": "CREATE", "target_path": targets[0]})}

    agent = _ReplyAgent()
    monkeypatch.setattr(agent, "reply", AsyncMock(side_effect=reply))
    monkeypatch.setattr("reme.steps.evolve.dream.extract.llm_available", lambda _step: True)
    monkeypatch.setattr("reme.steps.evolve.dream.integrate.llm_available", lambda _step: True)
    config = resolve_app_config(config="default", log_config=False)["jobs"][job_name]
    # Execute the cron's configured steps once without starting a scheduler.
    job = BaseJob(name=job_name, steps=config["steps"], app_context=app_context)
    await job.start()
    try:
        response = await job(
            date="2026-09-11",
            scan_days=1,
            agent_wrapper=agent,
            file_store=store,
            file_catalog=catalog,
        )
    finally:
        await job.close()

    integrate_fails = bool(integration_failure)
    assert response.success is not integrate_fails
    assert response.answer.startswith("AutoDream failed" if integrate_fails else "AutoDream completed")
    assert response.metadata["modified"] is True
    dream, tagging = response.metadata["dream"], response.metadata["auto_tag"]
    assert (source in dream["checkpoint_paths"]) is not integrate_fails
    assert (source in dream["failed_paths"]) is integrate_fails
    assert tagging["processed"] == 2
    assert tagging["failed"] == int(tag_fails)
    assert tagging["succeeded"] == 2 - int(tag_fails)
    assert [{"change": item["change"], "path": item["path"]} for item in tagging["results"]] == [
        {"change": "added", "path": path} for path in targets
    ]
    for path in targets:
        post = frontmatter.loads((tmp_path / path).read_text(encoding="utf-8"))
        assert post.metadata.get("memory_tags") == (None if tag_fails and path == targets[0] else ["ReMe"])
    assert (tmp_path / source).read_text(encoding="utf-8") == "Source about ReMe"


@pytest.mark.asyncio
async def test_extract_without_llm_marks_changed_paths_failed(tmp_path):
    """A missing LLM must not let finish checkpoint unprocessed source files."""

    note = _touch(tmp_path / "daily" / "2026-05-28" / "session.md")
    step = DreamExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    response = await step(
        RuntimeContext(date="2026-05-28", file_catalog=_Catalog(), file_store=_FileStore(tmp_path)),
    )

    dream = response.metadata["dream"]
    assert response.success is False
    assert note.relative_to(tmp_path).as_posix() in dream["changed_paths"]
    assert dream["failed_paths"] == dream["changed_paths"]


@pytest.mark.asyncio
async def test_finish_does_not_checkpoint_failed_changed_paths():
    """Finish does not checkpoint failed changed paths."""

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        ok = _touch(workspace / "daily" / "2026-05-28" / "ok.md")
        failed = _touch(workspace / "daily" / "2026-05-28" / "failed.md")
        day_index = _touch(workspace / "daily" / "2026-05-28.md")
        state = DreamState(
            date="2026-05-28",
            dates=["2026-05-26", "2026-05-27", "2026-05-28"],
            workspace=str(workspace),
            daily_dir="daily",
            changed_paths=[ok.relative_to(workspace).as_posix(), failed.relative_to(workspace).as_posix()],
            existing={path.relative_to(workspace).as_posix(): path.stat().st_mtime for path in (ok, failed, day_index)},
            failed_paths=[failed.relative_to(workspace).as_posix()],
            modified_paths=["digest/procedure/example.md"],
            integrate_results=[
                {
                    "action": "CREATE",
                    "target_path": "digest/procedure/example.md",
                    "note": "Created a concise procedure node.",
                },
            ],
        )
        step, catalog = DreamFinishStep(), _Catalog()
        resp = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

        upserted = [n.path for n in catalog.upserts]
        assert resp.success is True
        assert resp.answer.startswith("AutoDream completed\n\n")
        assert "action:" not in resp.answer
        assert "topics:" not in resp.answer
        assert "Changes:" in resp.answer
        assert "- [digest/procedure/example.md][CREATE]: Created a concise procedure node." in resp.answer
        assert ok.relative_to(workspace).as_posix() in upserted
        assert failed.relative_to(workspace).as_posix() not in upserted
        assert day_index.relative_to(workspace).as_posix() in upserted
        assert catalog.dumps == 1
        assert resp.metadata["modified"] is True


@pytest.mark.asyncio
async def test_finish_does_not_readd_a_failed_day_index(tmp_path):
    """The generated day-index list cannot bypass the failed-path filter."""

    day_index = _touch(tmp_path / "daily" / "2026-05-28.md")
    rel_path = day_index.relative_to(tmp_path).as_posix()
    state = DreamState(
        date="2026-05-28",
        dates=["2026-05-28"],
        workspace=str(tmp_path),
        daily_dir="daily",
        changed_paths=[rel_path],
        existing={rel_path: day_index.stat().st_mtime},
        failed_paths=[rel_path],
        errors=["extract unavailable"],
    )
    step, catalog = DreamFinishStep(), _Catalog()

    response = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

    assert response.success is False
    assert not catalog.upserts
    assert response.metadata["dream"]["checkpoint_paths"] == []
    assert response.metadata["modified"] is False


@pytest.mark.asyncio
async def test_finish_keeps_skipped_agent_output_successful(tmp_path):
    """Best-effort agent skips remain successful and checkpointed after their bounded retry."""

    rel_path = "daily/2026-05-28/source.md"
    source = _touch(tmp_path / rel_path)
    state = DreamState(
        date="2026-05-28",
        dates=["2026-05-28"],
        workspace=str(tmp_path),
        daily_dir="daily",
        changed_paths=[rel_path],
        existing={rel_path: source.stat().st_mtime},
        skipped_units=[{"name": "unit", "reason": "unusable receipt"}],
        warnings=["unit 'unit' skipped: unusable agent receipt"],
    )
    step, catalog = DreamFinishStep(), _Catalog()

    response = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

    assert response.success is True
    assert response.answer.startswith("AutoDream completed with warnings\n\n")
    assert "- Integrated: 0 ok, 1 skipped, 0 failed" in response.answer
    assert response.metadata["dream"]["checkpoint_paths"] == [rel_path]


@pytest.mark.parametrize("stage", ["extract", "integrate"])
@pytest.mark.asyncio
async def test_dream_initializes_started_provider_before_availability_check(tmp_path, stage):
    """Dream's real availability check must initialize a started lazy provider."""

    context = ApplicationContext(workspace_dir=str(tmp_path))
    credential_cls = Mock()
    llm = BaseAsLLM(app_context=context, model="test")
    llm.credential_cls = credential_cls
    source = "daily/2026-05-28/session.md"
    _touch(tmp_path / source, "memory source")
    unit = {"name": "memory", "bucket": "wiki", "summary": "memory summary", "paths": [source]}
    target = "digest/wiki/memory.md"
    if stage == "extract":
        agent = _ReplyAgent({"result": json.dumps({"units": [unit]})})
        step = DreamExtractStep(app_context=context, scan_days=1)
        inputs = {"date": "2026-05-28"}
    else:
        agent = _ReplyAgent(
            {"result": json.dumps({"action": "CREATE", "target_path": target, "note": "created"})},
            on_reply=lambda: _touch(tmp_path / target, "# Memory\n\nmemory summary\n"),
        )
        step = DreamIntegrateStep(app_context=context)
        inputs = {"dream": {"units": [unit], "workspace": str(tmp_path)}}
    context.components[ComponentEnum.AS_LLM] = {"default": llm}
    context.components[ComponentEnum.AGENT_WRAPPER] = {"default": agent}
    await llm.start()
    await agent.start()
    try:
        credential_cls.assert_not_called()
        with patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}):
            response = await step(
                RuntimeContext(file_catalog=_Catalog(), file_store=_FileStore(tmp_path), **inputs),
            )
        assert response.success, response.answer
        assert agent.calls == 1
        credential_cls.assert_called_once_with()
        dream = response.metadata["dream"]
        assert dream["errors"] == []
        assert dream["failed_paths"] == []
        if stage == "extract":
            assert dream["units"][0]["name"] == "memory"
        else:
            assert dream["integrate_results"][0]["target_path"] == target
    finally:
        await agent.close()
        await llm.close()
