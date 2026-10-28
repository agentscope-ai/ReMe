"""Debug derivation reads only selected cases and preserves deterministic output."""

import json
from pathlib import Path

from reme_home.preparation.data.basic import derive_debug_cases, write_normalized_dataset
from reme_home.preparation.data.models import CaseSpec, QuerySpec, SessionSpec


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
                        messages=[{"role": "user", "content": str(message_index)} for message_index in range(4)],
                    )
                    for session_index in range(3)
                ],
                queries=[QuerySpec(query_id="q", question="What?", golden_answer="answer")],
            )
            for index in range(30)
        ),
    )
    expected_indices = [0, 14, 29]
    expected_paths = [training / "cases" / f"{index:06d}.json" for index in expected_indices]
    reads = []
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.parent == training / "cases":
            assert path in expected_paths, f"Unselected training case read: {path}"
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    manifest = derive_debug_cases(training, output, sessions_per_case=2, messages_per_session=2)

    assert reads == expected_paths
    assert manifest.case_ids == [f"debug-{index}" for index in expected_indices]
    assert manifest.query_count == 3
    derivation = json.loads((output / "derivation.json").read_text())
    assert [case["source_case_index"] for case in derivation["cases"]] == expected_indices
    assert all(case["session_indices"] == [0, 2] for case in derivation["cases"])
    for path in sorted((output / "cases").glob("*.json")):
        case = CaseSpec.model_validate_json(path.read_text())
        assert [session.session_id for session in case.sessions] == ["s0", "s2"]
        assert all([message["content"] for message in session.messages] == ["0", "1"] for session in case.sessions)
