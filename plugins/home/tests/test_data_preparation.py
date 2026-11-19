"""Debug derivation reads only selected cases and preserves deterministic output."""

from pathlib import Path

import pytest

from reme_home.preparation.data.basic import derive_debug_cases, write_normalized_dataset
from reme_home.preparation.data.models import CaseSpec, GroundTruth, QuerySpec, SessionSpec
from reme_home.preparation.spec import DebugSpec


def test_debug_derivation_reads_only_selected_cases(tmp_path, monkeypatch):
    """Unselected training files need not be loaded to derive the smoke dataset."""
    training = tmp_path / "training"
    output = tmp_path / "debug"
    write_normalized_dataset(
        training,
        "beam",
        "10M",
        (
            CaseSpec(
                case_id=str(index),
                sessions=[
                    SessionSpec(
                        session_id=f"s{session_index}",
                        messages=[{"role": "user", "content": str(message_index)} for message_index in range(10)],
                    )
                    for session_index in range(5)
                ],
                queries=[QuerySpec(query_id="q", question="What?", groundtruth=GroundTruth())],
            )
            for index in range(30)
        ),
    )
    expected_indices = [0, 10, 19, 29]
    expected_paths = [training / "cases" / f"{index:06d}.json" for index in expected_indices]
    reads = []
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.parent == training / "cases":
            assert path in expected_paths, f"Unselected training case read: {path}"
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    debug = DebugSpec()
    manifest = derive_debug_cases(
        training,
        output,
        sessions_per_case=debug.sessions_per_case,
        messages_per_session=debug.messages_per_session,
    )

    assert reads == expected_paths
    assert manifest.case_ids == [f"debug-{index}" for index in expected_indices]
    assert manifest.query_count == 4
    assert not (output / "derivation.json").exists()
    assert "schema_version" not in manifest.model_dump()
    for path in sorted((output / "cases").glob("*.json")):
        case = CaseSpec.model_validate_json(path.read_text())
        assert [session.session_id for session in case.sessions] == ["s0", "s1", "s3", "s4"]
        assert all(
            [message["content"] for message in session.messages] == [str(index) for index in range(8)]
            for session in case.sessions
        )


@pytest.mark.parametrize(
    "case_types,limit,coverage",
    [
        ([list("abcdefgh") * 2 for _ in range(4)], None, 8),
        ([["common", "rare"], ["common"]], 1, 2),
        ([list("abc"), list("def")], 1, 2),
        ([[None] * 10, ["known"]], None, 1),
        ([[], ["known"]], None, 1),
    ],
)
def test_debug_queries_cover_types_with_case_limits(tmp_path, case_types, limit, coverage):
    """Coverage respects case capacity, scarce types, untyped data and short cases."""
    training = tmp_path / "training"
    write_normalized_dataset(
        training,
        "beam",
        "100K",
        [
            CaseSpec(
                case_id=str(case_index),
                sessions=[SessionSpec(session_id="s", messages=[{"role": "user", "content": "hello"}])],
                queries=[
                    QuerySpec(
                        query_id=str(query_index),
                        question="What?",
                        groundtruth=GroundTruth(),
                        metadata={"question_type": question_type} if question_type else {},
                    )
                    for query_index, question_type in enumerate(question_types)
                ],
            )
            for case_index, question_types in enumerate(case_types)
        ],
    )
    options = {} if limit is None else {"queries_per_case": limit}
    manifests = [
        derive_debug_cases(training, tmp_path / name, sessions_per_case=4, messages_per_session=8, **options)
        for name in ("first", "second")
    ]
    assert manifests[0].normalized_fingerprint == manifests[1].normalized_fingerprint
    selected_types = set()
    for path, question_types in zip(sorted((tmp_path / "first/cases").glob("*.json")), case_types):
        case = CaseSpec.model_validate_json(path.read_text())
        assert len(case.queries) == min(limit or DebugSpec().queries_per_case, len(question_types))
        assert len({query.query_id for query in case.queries}) == len(case.queries)
        assert all(query.query_id in {str(index) for index in range(len(question_types))} for query in case.queries)
        selected_types.update(query.metadata["question_type"] for query in case.queries if query.metadata)
    assert len(selected_types) == coverage
