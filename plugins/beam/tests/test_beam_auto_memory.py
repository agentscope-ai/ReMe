"""Unit tests for BEAM auto-memory segmented ingestion and line numbering."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from agentscope.message import Msg

from reme_beam.steps.auto_memory import (
    BeamAutoMemoryStep,
    split_turn_segments,
)

from reme.components.runtime_context import RuntimeContext


def _msg(role: str, words: int, created_at: str = "2024-03-01T09:00:00") -> Msg:
    return Msg(
        name=role,
        role=role,
        content=[{"type": "text", "text": " ".join(["w"] * words)}],
        created_at=created_at,
    )


def _dialog(n_turns: int, words_per_msg: int) -> list[Msg]:
    messages: list[Msg] = []
    for _ in range(n_turns):
        messages.append(_msg("user", words_per_msg))
        messages.append(_msg("assistant", words_per_msg))
    return messages


class TestSplitTurnSegments:
    def test_empty(self):
        assert not split_turn_segments([], 100)

    def test_disabled_returns_single_segment(self):
        messages = _dialog(3, 10)
        segments = split_turn_segments(messages, 0)
        assert len(segments) == 1
        assert segments[0] == (0, messages)

    def test_under_limit_single_segment(self):
        messages = _dialog(3, 10)  # 60 words total
        segments = split_turn_segments(messages, 100)
        assert len(segments) == 1
        assert segments[0][0] == 0
        assert segments[0][1] == messages

    def test_splits_at_turn_boundaries(self):
        # 4 turns x 20 words each; limit 40 -> 2 turns per segment
        messages = _dialog(4, 10)
        segments = split_turn_segments(messages, 40)
        assert len(segments) == 2
        offsets = [offset for offset, _ in segments]
        assert offsets == [0, 4]
        # Every segment starts with a user message and ends with an assistant
        for _, segment in segments:
            assert segment[0].role == "user"
            assert segment[-1].role == "assistant"
        # No message lost or duplicated, order preserved
        flattened = [m for _, segment in segments for m in segment]
        assert flattened == messages

    def test_never_splits_inside_a_turn(self):
        # One turn alone exceeds the limit -> becomes its own oversized segment
        messages = [
            _msg("user", 5),
            _msg("assistant", 5),
            _msg("user", 50),
            _msg("assistant", 50),  # 100-word turn > limit 60
            _msg("user", 5),
            _msg("assistant", 5),
        ]
        segments = split_turn_segments(messages, 60)
        assert [offset for offset, _ in segments] == [0, 2, 4]
        assert [len(segment) for _, segment in segments] == [2, 2, 2]

    def test_offsets_are_original_indices(self):
        messages = _dialog(5, 30)  # 60 words per turn
        segments = split_turn_segments(messages, 120)
        # 2 turns per segment -> offsets 0, 4, 8
        assert [offset for offset, _ in segments] == [0, 4, 8]
        for offset, segment in segments:
            for i, msg in enumerate(segment):
                assert msg is messages[offset + i]

    def test_multi_assistant_turn_stays_together(self):
        messages = [
            _msg("user", 10),
            _msg("assistant", 10),
            _msg("assistant", 10),
            _msg("user", 10),
            _msg("assistant", 10),
        ]
        segments = split_turn_segments(messages, 30)
        assert [offset for offset, _ in segments] == [0, 3]
        assert len(segments[0][1]) == 3


class TestFormatHistoryLineNumbers:
    def _step(self, offset: int) -> BeamAutoMemoryStep:
        step = BeamAutoMemoryStep(name="beam_auto_memory_step", backend="beam_auto_memory_step")
        step.context = RuntimeContext(
            session_id="beam_1M_1_batch1",
            beam_line_offset=offset,
        )
        return step

    def test_numbers_start_at_one_without_offset(self):
        step = self._step(0)
        history = step._format_history(_dialog(2, 3))
        assert "[L1 | user @" in history
        assert "[L4 | assistant @" in history
        assert "lines 1-4" in history
        assert "session/dialog/beam_1M_1_batch1.jsonl" in history

    def test_numbers_use_original_file_offset(self):
        step = self._step(40)
        history = step._format_history(_dialog(2, 3))
        assert "[L41 | user @" in history
        assert "[L44 | assistant @" in history
        assert "lines 41-44" in history
        assert "[L1 " not in history

    def test_build_messages_passes_msg_objects_through(self):
        step = self._step(0)
        messages = _dialog(2, 3)
        rebuilt = step._build_messages(messages)
        assert [m.id for m in rebuilt] == [m.id for m in messages]
