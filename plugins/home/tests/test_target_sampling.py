"""Targeted formal sampling and per-case replay through real Home Jobs."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import copy
from pathlib import Path

import pytest

from reme_home import sampling
from reme_home.agent.optimizer import verify_evaluation
from reme_home.evaluation import reuse
from reme_home.evaluation.artifacts import read_json, write_json
from reme_home.evaluation.history import rebuild_sampling_state, read_state
from reme_home.preparation.data.models import CaseSpec, GroundTruth, QuerySpec
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    command,
    commit_candidate,
    home_application,
    preparation_inputs,
    prepared_home,
)


def cases():
    return [
        CaseSpec(
            case_id=str(index),
            sessions=[],
            queries=[QuerySpec(query_id=f"fact:{qid}", question="?", groundtruth=GroundTruth()) for qid in range(1, 5)],
        )
        for index in range(1, 5)
    ]


def test_reuse_scan_filters_manifest_first_and_caches_distance(tmp_path, monkeypatch):
    fingerprints = {"dataset": "data", "construction_config": "config"}
    manifest = {
        "commit_sha": "near",
        "mode": "scored",
        "dataset_kind": "training-set",
        "fingerprints": fingerprints,
    }
    summary = {
        "status": "completed",
        "cases": {"1": {"construction": {"status": "success"}, "queries": {"q1": {}}}},
    }
    paths = {}
    for branch, overrides in (
        ("a-far", {"commit_sha": "init", "status": "completed"}),
        ("b-missing-status", {}),
        ("c-near-first", {"status": "completed"}),
        ("d-near-second", {"status": "completed"}),
        ("e-mismatch", {"fingerprints": {}, "status": "completed"}),
        ("f-infra-error", {"status": "completed"}),
    ):
        run = tmp_path / "evaluations" / branch / "prefix/1"
        paths[branch] = run
        write_json(run / "manifest.json", {**manifest, **overrides})
        write_json(
            run / "summary.json",
            {**summary, "query_status_counts": {"infra_error": int(branch == "f-infra-error")}},
        )

    distances = []
    summaries = []

    def git(repo, *args):
        assert repo == tmp_path
        if args == ("rev-parse", "current^1"):
            return "near"
        if args == ("rev-list", "current"):
            return "current\nnear\ninit"
        assert args[:2] == ("rev-list", "--count")
        distances.append(args[-1])
        return {"init..current": "2", "near..current": "1"}[args[-1]]

    def track_summary(path):
        if path.name == "summary.json":
            summaries.append(path.parent.parent.parent.name)
        return read_json(path)

    monkeypatch.setattr(reuse, "verify_candidate", lambda *_: (tmp_path, {"code": {"initial_commit": "init"}}))
    monkeypatch.setattr(reuse, "git", git)
    monkeypatch.setattr(reuse, "read_json", track_summary)

    result = reuse.reuse_context(tmp_path, None, [], fingerprints, "current", "candidate", targeted=True)
    assert summaries == ["a-far", "c-near-first", "d-near-second", "f-infra-error"]
    assert distances == ["init..current", "near..current"]
    assert result["sources"] == [
        (str(paths["c-near-first"]), "near"),
        (str(paths["d-near-second"]), "near"),
        (str(paths["a-far"]), "init"),
    ]
    assert result["candidate_sources"] == {"1": str(paths["c-near-first"])}


def test_target_forms_invalid_ids_and_remaining_capacity(monkeypatch):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 12345)
    state = {"round": 2, "cases": {}, "queries": {}}
    before = copy.deepcopy(state)
    plan = sampling.create_sampling_plan(
        cases(),
        state,
        sampling.SamplingPolicy(max_cases=4, max_queries=12),
        target_queries={"1": [], "2": None, "3": ["fact:1", "fact:1", "missing"], "missing": None},
    )
    assert set(plan["case_ids"]) == {"1", "2", "3", "4"}
    assert sum(map(len, plan["query_ids"].values())) == 12
    assert set(plan["query_ids"]["2"]) == {f"fact:{qid}" for qid in range(1, 5)}
    assert "fact:1" in plan["query_ids"]["3"]
    targets = plan["targets"]
    assert targets["invalid_case_ids"] == ["missing"]
    assert targets["invalid_query_ids"] == {"3": ["missing"]}
    assert targets["retained_query_ids"]["1"] == []
    assert targets["retained_query_ids"]["3"] == ["fact:1"]
    assert all(len(ids) == len(set(ids)) for ids in plan["query_ids"].values())
    assert state == before


@pytest.mark.parametrize("case_limit,query_limit", [(2, 8), (4, 2), (3, 5), (4, None)])
def test_oversized_targets_stay_within_both_limits(monkeypatch, case_limit, query_limit):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 42)
    requested = {str(index): None for index in range(1, 5)}
    policy = sampling.SamplingPolicy(max_cases=case_limit, max_queries=query_limit)
    plan = sampling.create_sampling_plan(cases(), {}, policy, target_queries=requested)
    assert len(plan["case_ids"]) == min(case_limit, query_limit or case_limit)
    assert sum(map(len, plan["query_ids"].values())) == min(query_limit or 16, case_limit * 4)
    assert all(plan["query_ids"].values())
    assert plan == sampling.create_sampling_plan(cases(), {}, policy, target_queries=requested)
    assert set(plan["targets"]["trimmed_case_ids"]) == set(requested) - set(plan["case_ids"])


def test_case_anchor_keeps_a_question_before_other_targets_consume_budget(monkeypatch):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 42)
    plan = sampling.create_sampling_plan(
        cases(),
        {},
        sampling.SamplingPolicy(max_cases=2, max_queries=4),
        target_queries={"1": [], "2": None},
    )
    assert len(plan["query_ids"]["1"]) == 1
    assert len(plan["query_ids"]["2"]) == 3
    assert len(plan["targets"]["trimmed_query_ids"]["2"]) == 1


def test_invalid_questions_keep_case_and_invalid_cases_fall_back(monkeypatch):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 7)
    policy = sampling.SamplingPolicy(max_cases=1, max_queries=2)
    plan = sampling.create_sampling_plan(cases(), {}, policy, target_queries={"2": ["missing"]})
    assert plan["case_ids"] == ["2"]
    assert len(plan["query_ids"]["2"]) == 2
    assert plan["targets"]["retained_query_ids"] == {"2": []}
    fallback = sampling.create_sampling_plan(cases(), {}, policy, target_queries={"missing": []})
    original = sampling.create_sampling_plan(cases(), {}, policy)
    assert {key: value for key, value in fallback.items() if key != "targets"} == original


def test_debug_ignores_targets_and_empty_targets_preserve_exact_mode():
    from reme_home.evaluation.inputs import EvaluationSpec

    assert EvaluationSpec(debug=True, target_queries={"missing": None}).target_queries is None
    assert EvaluationSpec(target_queries={}).target_queries is None


@pytest.fixture
def allow_reuse(monkeypatch):
    from reme_home.agent import candidate_gate, construction_equivalence

    reviews, draws = [], []

    async def review(wrapper, home, context, spec, selected, fingerprints, **kwargs):
        reviews.append((context["source_validation"], [case.case_id for case in selected]))
        return {**context, "decision": "construction_unchanged", "reason": "Answer-only change"}

    def draw():
        draws.append(True)
        return {"probability": 0.6, "draw": 0.1, "selected": True}

    async def admit(wrapper, home, context, budget, **kwargs):
        return {**context, "decision": "accept", "reason": "Fixture candidate"}

    monkeypatch.setattr(construction_equivalence, "review_construction", review)
    monkeypatch.setattr(candidate_gate, "review_candidate", admit)
    monkeypatch.setattr(reuse, "select_query_only", draw)
    return reviews, draws


@pytest.mark.asyncio
async def test_partial_reuse_preserves_targets_and_recovers_frozen_plan(app, prepared, allow_reuse, monkeypatch):
    baseline = await app.run_job("home_evaluate", case_ids=["1"])
    assert baseline.success
    commit_candidate(prepared)
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    failed = await app.run_job(
        "home_evaluate",
        sampling=True,
        sampling_policy={"max_cases": 2, "max_queries": 2},
        target_queries={"1": [], "2": ["missing"], "invalid": None},
        max_evaluations=10,
    )
    assert not failed.success
    assert isinstance(failed.answer, dict), failed.answer
    path = Path(failed.answer["result_dir"]) / "manifest.json"
    old = read_json(path)
    assert old["execution_mode"] == "mixed_validation"
    assert set(old["case_ids"]) == {"1", "2"}
    assert set(old["construction_sources"]) == {"1"}
    assert old["target_selection"]["invalid_case_ids"] == ["invalid"]
    events = FakeFactory.instances[-1].events
    assert ("1", "upload") in events
    assert any(event[:2] == ("2", "build") for event in events)
    monkeypatch.setattr(FakeFactory, "bad_export", False)

    def no_draw(*args, **kwargs):
        pytest.fail("Recovery cannot redraw targets or reuse")

    monkeypatch.setattr(sampling, "create_sampling_plan", no_draw)
    monkeypatch.setattr("reme_home.evaluation.component.create_sampling_plan", no_draw)
    monkeypatch.setattr(reuse, "select_query_only", no_draw)
    write_json(path, {key: value for key, value in old.items() if key != "construction_sources"})
    launches = len(FakeFactory.instances)
    rejected = await app.run_job("home_evaluate", sampling=True)
    assert not rejected.success and "Recovery requires saved per-case construction sources" in str(rejected.answer)
    assert len(FakeFactory.instances) == launches
    write_json(path, old)
    recovered = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries={"invalid-new": None},
        max_evaluations=10,
    )
    assert recovered.success, recovered.answer
    run = Path(recovered.answer["result_dir"])
    manifest = read_json(run / "manifest.json")
    assert manifest["query_ids"] == old["query_ids"]
    assert manifest["construction_sources"] == old["construction_sources"]
    assert manifest["target_selection"] == old["target_selection"]
    assert recovered.answer["selection"]["invalid_case_count"] == 1
    assert recovered.answer["selection"]["invalid_query_count"] == 1
    assert any("Restored" in message for message in recovered.answer["selection"]["notices"])
    assert len(allow_reuse[0]) == len(allow_reuse[1]) == 1
    verify_evaluation(prepared, run, manifest["commit_sha"])
    before = read_state(prepared)
    assert rebuild_sampling_state(prepared) == before
    # Final acceptance must check the actual per-case snapshot, not just the run mode.
    result_path = run / "cases/1/memory_construction/result.json"
    result = read_json(result_path)
    write_json(result_path, {**result, "workspace_sha256": "wrong"})
    with pytest.raises(ValueError, match="provenance"):
        verify_evaluation(prepared, run, manifest["commit_sha"])


@pytest.mark.asyncio
async def test_multiple_sources_reuse_every_available_selected_case(app, prepared, allow_reuse):
    first = await app.run_job("home_evaluate", case_ids=["1"])
    assert first.success
    commit_candidate(prepared)
    second = await app.run_job("home_evaluate", case_ids=["2"])
    assert second.success
    commit_candidate(prepared)
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries={"1": None, "2": []},
        sampling_policy={"max_cases": 2, "max_queries": 2},
    )
    assert response.success, response.answer
    run = Path(response.answer["result_dir"])
    manifest = read_json(run / "manifest.json")
    assert manifest["execution_mode"] == "query_only_replay"
    assert manifest["construction_sources"]["1"]["result_dir"] == first.answer["result_dir"]
    assert manifest["construction_sources"]["2"]["result_dir"] == second.answer["result_dir"]
    assert "construction_source" not in manifest
    decision = response.answer["automatic_construction_reuse"]
    saved = manifest["automatic_construction_reuse"]
    assert "equivalence_report" not in saved
    assert len(saved["source_reviews"]) == 2
    assert saved["source_reviews"][saved["source_validation"]]["equivalence_report"] == decision["equivalence_report"]
    assert len(allow_reuse[0]) == 2 and len(allow_reuse[1]) == 1
    assert all(event[1] != "build" for event in FakeFactory.instances[-1].events)
    verify_evaluation(prepared, run, manifest["commit_sha"])
    saved.pop("probability_selected")
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="successful host probability draw"):
        verify_evaluation(prepared, run, manifest["commit_sha"])
    manifest.pop("construction_sources")
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="host-selected per-case construction sources"):
        verify_evaluation(prepared, run, manifest["commit_sha"])


@pytest.mark.asyncio
@pytest.mark.parametrize("same_commit", [False, True])
async def test_checkout_change_after_prepare_is_rejected_before_replay_scan(
    app,
    prepared,
    allow_reuse,
    monkeypatch,
    same_commit,
):
    component = app.context.components["home.evaluation"]["default"]
    repo = prepared / "code/repo"
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    monkeypatch.setattr(FakeFactory, "bad_export", True)
    failed = await app.run_job("home_evaluate", sampling=True)
    assert not failed.success
    manifest = read_json(Path(failed.answer["result_dir"]) / "manifest.json")
    assert manifest["automatic_construction_reuse"]["selected"]
    if same_commit:
        commit_candidate(prepared)
        command(repo, "branch", "other-candidate")
    else:
        command(repo, "switch", "-c", "next-candidate", "init")
        path = repo / "plugins/beam/src/reme_beam/__init__.py"
        path.write_text(path.read_text() + "\nsibling = True\n")
        commit_candidate(prepared)

    prepare = component._prepare
    launches = len(FakeFactory.instances)

    def switch_after_prepare(*args):
        result = prepare(*args)
        command(repo, "switch", "other-candidate" if same_commit else "test-candidate")
        return result

    def unexpected_read(path):
        pytest.fail(f"Replay read {path} after the checkout changed")

    monkeypatch.setattr(component, "_prepare", switch_after_prepare)
    monkeypatch.setattr(reuse, "read_json", unexpected_read)
    result = await app.run_job("home_evaluate", sampling=True)
    assert not result.success and "Candidate changed during equivalence review" in str(result.answer)
    assert len(FakeFactory.instances) == launches
    assert len(allow_reuse[0]) == len(allow_reuse[1]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_each_init_child_samples_after_review_and_draw(app, prepared, allow_reuse, monkeypatch, selected):
    from reme_home.evaluation import component

    assert (await app.run_job("home_evaluate")).success
    events = []
    review = reuse._review_source
    sample = component.create_sampling_plan

    async def tracked_review(*args, **kwargs):
        events.append("review")
        return await review(*args, **kwargs)

    def draw():
        events.append("draw")
        return {"probability": 0.6, "draw": 0.1 if selected else 0.9, "selected": selected}

    def tracked_sample(*args, **kwargs):
        events.append("sample")
        return sample(*args, **kwargs)

    monkeypatch.setattr(reuse, "_review_source", tracked_review)
    monkeypatch.setattr(reuse, "select_query_only", draw)
    monkeypatch.setattr(component, "create_sampling_plan", tracked_sample)
    for sibling in (False, True):
        if sibling:
            command(prepared / "code/repo", "switch", "-c", "sibling", "init")
            path = prepared / "code/repo/plugins/beam/src/reme_beam/__init__.py"
            path.write_text(path.read_text() + "\nsibling = True\n")
        commit_candidate(prepared)
        events.clear()
        response = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_cases": 1, "max_queries": 1})
        assert response.success, response.answer
        assert events == ["review", "draw", "sample"]
        assert response.answer["case_count"] == response.answer["scored_query_count"] == 1
        assert response.answer["execution_mode"] == ("query_only_replay" if selected else "full_validation")
        run = Path(response.answer["result_dir"])
        verify_evaluation(prepared, run, response.answer["commit_sha"])


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", [None, {}, {"invalid": None}])
async def test_no_targets_replays_exact_source_selection_without_sampling(
    app,
    prepared,
    allow_reuse,
    monkeypatch,
    targets,
):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    source = await app.run_job("home_evaluate", case_ids=["2"], query_ids={"2": ["fact:1"]})
    assert source.success
    commit_candidate(prepared)
    monkeypatch.setattr(
        "reme_home.evaluation.component.create_sampling_plan",
        lambda *args, **kwargs: pytest.fail("A paired replay must not sample"),
    )
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries=targets,
        sampling_policy={"max_cases": 1},
    )
    assert response.success, response.answer
    manifest = read_json(Path(response.answer["result_dir"]) / "manifest.json")
    old = read_json(Path(source.answer["result_dir"]) / "manifest.json")
    assert manifest["case_ids"] == old["case_ids"]
    assert manifest["query_ids"] == old["query_ids"]
    assert manifest["sampling"]["mode"] == "replay"
    assert response.answer["selection"]["invalid_case_count"] == bool(targets)
    verify_evaluation(prepared, Path(response.answer["result_dir"]), manifest["commit_sha"])


@pytest.mark.asyncio
async def test_init_source_does_not_replace_the_direct_parent_check(app, prepared, allow_reuse, monkeypatch):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    commit_candidate(prepared)
    commit_candidate(prepared)
    monkeypatch.setattr(
        "reme_home.evaluation.component.create_sampling_plan",
        lambda *args, **kwargs: pytest.fail("The direct parent is not init, even though the memory source is init"),
    )
    response = await app.run_job("home_evaluate", sampling=True)
    assert response.success, response.answer
    manifest = read_json(Path(response.answer["result_dir"]) / "manifest.json")
    old = read_json(Path(baseline.answer["result_dir"]) / "manifest.json")
    assert manifest["case_ids"] == old["case_ids"]
    assert manifest["query_ids"] == old["query_ids"]
    assert manifest["sampling"]["mode"] == "replay"


@pytest.mark.asyncio
async def test_init_child_sampling_can_select_an_uncovered_case(app, prepared, allow_reuse, monkeypatch):
    from reme_home.evaluation import component

    assert (await app.run_job("home_evaluate", case_ids=["1"])).success
    commit_candidate(prepared)
    sample = component.create_sampling_plan

    def uncovered_sample(available, state, policy, **kwargs):
        assert len(allow_reuse[0]) == len(allow_reuse[1]) == 1
        return sample([case for case in available if case.case_id == "2"], state, policy, **kwargs)

    monkeypatch.setattr(component, "create_sampling_plan", uncovered_sample)
    response = await app.run_job("home_evaluate", sampling=True, sampling_policy={"max_cases": 1, "max_queries": 1})
    assert response.success, response.answer
    decision = response.answer["automatic_construction_reuse"]
    assert decision["probability_selected"] is True and decision["selected"] is False
    assert decision["case_sources"] == {}
    assert decision["reason"] == "no_reusable_snapshot_for_sampled_cases"
    assert response.answer["execution_mode"] == "full_validation"
    manifest = read_json(Path(response.answer["result_dir"]) / "manifest.json")
    assert manifest["case_ids"] == ["2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("target", [[], None, ["fact:1"], ["invalid"]])
async def test_targets_share_one_draw_with_background(
    app,
    prepared,
    allow_reuse,
    monkeypatch,
    selected,
    target,
):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    draws = []

    def draw():
        draws.append(True)
        return {"probability": 0.6, "draw": 0.1 if selected else 0.9, "selected": selected}

    monkeypatch.setattr(reuse, "select_query_only", draw)
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries={"1": target},
        sampling_policy={"max_cases": 2, "max_queries": 2},
    )
    assert response.success, response.answer
    decision = response.answer["automatic_construction_reuse"]
    assert decision["selected"] is selected and decision["probability_selected"] is selected
    assert set(decision["case_sources"]) == ({"1", "2"} if selected else set())
    if selected:
        assert decision["reuse_reasons"]["1"] == "60_percent_policy"
    assert len(draws) == 1
    assert response.answer["execution_mode"] == ("query_only_replay" if selected else "full_validation")
    run = Path(response.answer["result_dir"])
    verify_evaluation(prepared, run, response.answer["commit_sha"])
    if selected:
        path = run / "manifest.json"
        manifest = read_json(path)
        manifest["automatic_construction_reuse"]["target_case_ids"] = []
        write_json(path, manifest)
        with pytest.raises(ValueError, match="matching targets"):
            verify_evaluation(prepared, run, response.answer["commit_sha"])


@pytest.mark.asyncio
@pytest.mark.parametrize("case_limit,query_limit", [(1, 2), (2, 1)])
async def test_oversized_replay_source_falls_back_to_sampling(
    app,
    prepared,
    allow_reuse,
    monkeypatch,
    case_limit,
    query_limit,
):
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    assert (await app.run_job("home_evaluate")).success
    commit_candidate(prepared)
    monkeypatch.setattr(reuse, "select_query_only", lambda: pytest.fail("No compatible source to draw for"))
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        sampling_policy={"max_cases": case_limit, "max_queries": query_limit},
    )
    assert response.success, response.answer
    assert response.answer["case_count"] == 1
    assert response.answer["execution_mode"] == "full_validation"
    assert not allow_reuse[0]


@pytest.mark.parametrize(
    "targets,reusable,limit,preferred",
    [
        ({"1": []}, {"1", "3", "4"}, 3, True),
        ({"1": []}, {"1", "3"}, 4, True),
        ({"1": [], "2": []}, {"1", "3"}, 3, False),
        ({"1": [], "2": []}, {"1", "2", "3", "4"}, 1, True),
    ],
)
def test_background_prefers_memory_only_when_all_retained_targets_are_reusable(
    monkeypatch,
    targets,
    reusable,
    limit,
    preferred,
):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 42)
    plan = sampling.create_sampling_plan(
        cases(),
        {},
        sampling.SamplingPolicy(max_cases=limit, max_queries=limit),
        target_queries=targets,
        reusable_case_ids=reusable,
    )
    assert plan["prefer_reusable_background"] is preferred
    assert len(plan["case_ids"]) == limit
    if preferred:
        assert len(set(plan["case_ids"]) & reusable) == min(len(reusable), limit)
    else:
        ordinary = sampling.create_sampling_plan(
            cases(),
            {},
            sampling.SamplingPolicy(max_cases=limit, max_queries=limit),
            target_queries=targets,
        )
        assert plan["case_ids"] == ordinary["case_ids"]
        assert plan["query_ids"] == ordinary["query_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize("target_id", ["1", "2"])
async def test_target_sampling_limits_review_and_uncovered_cases_rebuild(
    app,
    prepared,
    allow_reuse,
    monkeypatch,
    target_id,
):
    from reme_home.evaluation import component

    assert (await app.run_job("home_evaluate", case_ids=["1"])).success
    commit_candidate(prepared)
    sample = component.create_sampling_plan

    def tracked_sample(*args, **kwargs):
        assert not allow_reuse[0] and not allow_reuse[1]
        assert kwargs["reusable_case_ids"] == {"1"}
        return sample(*args, **kwargs)

    monkeypatch.setattr(component, "create_sampling_plan", tracked_sample)
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries={target_id: ["fact:1"]},
        sampling_policy={"max_cases": 2, "max_queries": 2},
    )
    assert response.success, response.answer
    assert response.answer["execution_mode"] == "mixed_validation"
    run = Path(response.answer["result_dir"])
    manifest = read_json(run / "manifest.json")
    plan = read_json(prepared / manifest["sampling"]["plan_path"])
    assert plan["prefer_reusable_background"] is (target_id == "1")
    assert set(manifest["construction_sources"]) == {"1"}
    events = FakeFactory.instances[-1].events
    assert ("1", "upload") in events
    assert not any(event[:2] == ("1", "build") for event in events)
    assert any(event[:2] == ("2", "build") for event in events)
    verify_evaluation(prepared, run, manifest["commit_sha"])


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", [{"max_cases": 1}, {"max_cases": 2, "max_queries": 1}])
@pytest.mark.parametrize("reuse_selected", [True, False])
async def test_targeted_review_opens_only_budgeted_archives(app, prepared, monkeypatch, limits, reuse_selected):
    from reme_home.agent import construction_equivalence
    from reme_home.evaluation import inputs
    from reme_home.workspace import _initialization_lock

    baseline = await app.run_job("home_evaluate")
    assert baseline.success, baseline.answer
    source = Path(baseline.answer["result_dir"])
    commit_candidate(prepared)
    inspected, exported = set(), set()
    inspect_archive = inputs.inspect_archive

    def inspect(path, max_bytes, destination=None):
        if path.is_relative_to(source):
            case_id = path.relative_to(source).parts[1]
            (exported if destination is not None else inspected).add(case_id)
            with _initialization_lock(prepared / "metadata/workspace.lock"):
                pass  # Archive validation must not block other workspace Jobs.
        return inspect_archive(path, max_bytes, destination)

    async def report(_wrapper, _home, _scope, _prompt, context, **_kwargs):
        return {**context, "decision": "construction_unchanged", "reason": "Answer-only change"}

    monkeypatch.setattr(inputs, "inspect_archive", inspect)
    monkeypatch.setattr(construction_equivalence, "inspect_archive", inspect)
    monkeypatch.setattr(construction_equivalence, "review_report", report)
    monkeypatch.setattr(
        reuse,
        "select_query_only",
        lambda: {"probability": 0.6, "draw": 0.1 if reuse_selected else 0.9, "selected": reuse_selected},
    )
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries={"1": ["fact:1"]},
        sampling_policy=limits,
    )
    assert response.success, response.answer
    run = Path(response.answer["result_dir"])
    manifest = read_json(run / "manifest.json")
    assert manifest["query_ids"] == {"1": ["fact:1"]}
    assert inspected == exported == {"1"}
    plan = read_json(prepared / manifest["sampling"]["plan_path"])
    assert plan["prefer_reusable_background"] is reuse_selected
    assert manifest["execution_mode"] == ("query_only_replay" if reuse_selected else "full_validation")
    verify_evaluation(prepared, run, manifest["commit_sha"])


@pytest.mark.asyncio
async def test_targeted_review_rebuilds_if_nominated_archive_fails_validation(app, prepared, allow_reuse):
    baseline = await app.run_job("home_evaluate")
    assert baseline.success
    commit_candidate(prepared)
    recent = await app.run_job("home_evaluate", case_ids=["1"])
    assert recent.success
    (Path(recent.answer["result_dir"]) / "cases/1/memory_construction/reme_workspace.tar.gz").write_bytes(b"bad")
    commit_candidate(prepared)
    response = await app.run_job(
        "home_evaluate",
        sampling=True,
        target_queries={"1": []},
        sampling_policy={"max_cases": 1},
    )
    assert response.success, response.answer
    assert response.answer["execution_mode"] == "full_validation"
    assert allow_reuse == ([], [])
    assert any(event[:2] == ("1", "build") for event in FakeFactory.instances[-1].events)


def test_discarded_uncovered_target_does_not_disable_memory_preference(monkeypatch):
    monkeypatch.setattr(sampling.time, "time_ns", lambda: 42)
    policy = sampling.SamplingPolicy(max_cases=1, max_queries=1)
    targets = {"1": [], "2": []}
    ordinary = sampling.create_sampling_plan(cases(), {}, policy, target_queries=targets)
    retained = set(ordinary["targets"]["retained_case_ids"])
    plan = sampling.create_sampling_plan(
        cases(),
        {},
        policy,
        target_queries=targets,
        reusable_case_ids=retained,
    )
    assert plan["prefer_reusable_background"] is True
    assert plan["case_ids"] == ordinary["case_ids"]
    assert len(plan["targets"]["trimmed_case_ids"]) == 1
