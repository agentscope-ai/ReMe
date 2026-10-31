"""Real Job tool invocation by a fake extraction model, including cross-segment revision."""

# pylint: disable=missing-function-docstring

import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_default_long_segments_and_internal_summary_slack(app_factory, tmp_path):
    app, store = await app_factory()
    source = tmp_path / "long.log"
    text = "x" * (96 * 1024 - 1) + "\nlast segment"
    source.write_text(text)
    seen = []

    async def reply(inputs, _options):
        data = json.loads(inputs)
        seen.append(data)
        assert data["token_limits"]["segment"] == 98304
        assert data["token_limits"]["summary"] == 8192
        assert data["token_limits"]["context"] == 262144
        assert data["previous_summary"] == ("" if len(seen) == 1 else "s" * 9011)
        return {"result": json.dumps({"summary": "s" * 9011, "complete": True})}

    app.context.metadata["fake_reply"] = reply
    result = await app.context.jobs["procedure_auto_memory"](file=str(source), filename="long.log", request_id="long")
    assert result.success, result.answer
    assert len(seen) == 2
    assert "".join(data["trajectory_text"] for data in seen) == text
    assert all(0 < store.budget.count(data["trajectory_text"]) <= 98304 for data in seen)
    assert [data["offset"] for data in seen] == [0, len(seen[0]["trajectory_text"])]
    assert store.budget.limits["summary"] == 8192


@pytest.mark.asyncio
async def test_progressive_extraction_and_tool_writes(app_factory, tmp_path):
    app, store = await app_factory(token_limits={"segment": 12})
    source = tmp_path / "run.jsonl"
    source.write_text('{"step":1}\n{"step":2}\n')
    seen = []

    async def reply(inputs, options):
        data = json.loads(inputs)
        seen.append(data)
        assert options["builtin_tools"] == []
        assert set(options["job_tools"]) == {
            "procedure_search",
            "procedure_read",
            "procedure_create",
            "procedure_update",
        }
        injected = options["injected_job_kwargs"]
        search = await app.context.jobs["procedure_search"](query="repair", index="when_to_use_content", **injected)
        assert search.success
        if data["segment"] == 0:
            assert data["previous_summary"] == ""
            result = await app.context.jobs["procedure_create"](
                filename="repair-from-evidence.json",
                description="Repair",
                **{"when-to-use": "When repair fails"},
                content="Inspect before retry",
                operation_key="create",
                **injected,
            )
        else:
            assert data["previous_summary"] == "first summary"
            original = search.metadata["results"][0]
            result = await app.context.jobs["procedure_update"](
                path=original["path"],
                expected_version=original["version"],
                operation_key="update",
                changes={"content": "Rebuild after inspecting evidence"},
                **injected,
            )
        assert result.success
        # Verify mutation is searchable before the agent finishes this segment.
        immediate = await app.context.jobs["procedure_search"](query="repair", index="when_to_use", **injected)
        assert immediate.metadata["results"][0]["version"] == result.metadata["version"]
        return {
            "result": json.dumps(
                {"summary": "first summary" if data["segment"] == 0 else "combined summary", "complete": True},
            ),
        }

    app.context.metadata["fake_reply"] = reply
    result = await app.context.jobs["procedure_auto_memory"](
        file=str(source),
        filename="run.jsonl",
        request_id="stable",
    )
    assert result.success, result.answer
    assert len(seen) == 2
    memory = await store.read_experience("procedures/repair-from-evidence.json")
    assert memory["experience"]["content"] == "Rebuild after inspecting evidence"
    assert memory["experience"]["sources"] == ["session/trajectories/run.jsonl"]
    assert result.metadata["summary"] == "combined summary"
    repeated = await app.context.jobs["procedure_auto_memory"](
        file=str(source),
        filename="run.jsonl",
        request_id="stable",
    )
    assert repeated.success
    assert len(seen) == 2
    assert len(list(store.trajectory_root.glob("run*"))) == 1
    assert not store.scopes


@pytest.mark.asyncio
async def test_zero_and_multiple_experiences(app_factory, tmp_path):
    app, store = await app_factory()
    source = tmp_path / "empty.txt"
    source.write_text("No reusable action")

    async def skip(_inputs, _options):
        return {"result": '{"summary":"Nothing reusable","complete":true}'}

    app.context.metadata["fake_reply"] = skip
    assert (await app.context.jobs["procedure_auto_memory"](file=str(source), filename="empty.txt")).success
    assert not list(store.procedure_root.glob("*.json"))

    async def many(_inputs, options):
        for name in ("inspect-errors", "repair-cache", "verify-results"):
            response = await app.context.jobs["procedure_create"](
                filename=f"{name}.json",
                description=name,
                **{"when-to-use": "On failure"},
                content=name,
                operation_key=name,
                **options["injected_job_kwargs"],
            )
            assert response.success
        return {"result": '{"summary":"Three complementary actions","complete":true}'}

    app.context.metadata["fake_reply"] = many
    assert (await app.context.jobs["procedure_auto_memory"](file=str(source), filename="more.txt")).success
    assert len(list(store.procedure_root.glob("*.json"))) == 3


@pytest.mark.asyncio
async def test_summary_repair_has_no_mutation_tools(app_factory, tmp_path):
    app, _ = await app_factory(token_limits={"summary": 5})
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    calls = []

    async def reply(_inputs, options):
        calls.append(options)
        return {"result": json.dumps({"summary": "too much summary" if len(calls) == 1 else "brief", "complete": True})}

    app.context.metadata["fake_reply"] = reply
    result = await app.context.jobs["procedure_auto_memory"](file=str(source), filename="a.txt")
    assert result.success, result.answer
    assert calls[1]["job_tools"] == []
    assert result.metadata["summary"] == "brief"


@pytest.mark.asyncio
async def test_single_process_serializes_files(app_factory, tmp_path):
    app, _ = await app_factory()
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    active = 0
    maximum = 0

    async def reply(_inputs, _options):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"result": '{"summary":"done","complete":true}'}

    app.context.metadata["fake_reply"] = reply
    results = await asyncio.gather(
        *[app.context.jobs["procedure_auto_memory"](file=str(source), filename="a.txt") for _ in range(2)],
    )
    assert all(result.success for result in results)
    assert maximum == 1
    assert {result.metadata["source"] for result in results} == {
        "session/trajectories/a.txt",
        "session/trajectories/a-2.txt",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["exceed_max_iters", "interrupted", "error"])
async def test_sdk_termination_cannot_advance_segment(app_factory, tmp_path, reason):
    app, store = await app_factory()
    source = tmp_path / "a.txt"
    source.write_text("evidence")

    async def reply(_inputs, _options):
        return {
            "result": '{"summary":"partial","complete":true}',
            "last_message": {"finished_reason": reason},
        }

    app.context.metadata["fake_reply"] = reply
    result = await app.context.jobs["procedure_auto_memory"](file=str(source), filename="a.txt", request_id="limited")
    assert not result.success
    assert result.metadata["next_segment"] == 0
    assert result.metadata["active"]["execution_incomplete"]
    assert not store.scopes
