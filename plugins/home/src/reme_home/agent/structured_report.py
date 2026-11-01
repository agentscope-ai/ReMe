"""Shared recovery for schema-constrained final reports from ReAct agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from agentscope.message import AssistantMsg, Msg, Usage, UserMsg
from pydantic import BaseModel

from .bounded import BoundedReActAgent

ReportT = TypeVar("ReportT", bound=BaseModel)

logger = logging.getLogger(__name__)


class StructuredReportAgent(BoundedReActAgent):
    """Validate review reports and make at most one context-preserving correction."""

    def __init__(self, *args, report_model: type[ReportT], **kwargs):
        super().__init__(*args, **kwargs)
        self.report_model = report_model

    async def reply(self, *args, **kwargs) -> Msg:
        """Keep normal wrapper persistence and accounting around report recovery."""
        reply = await super().reply(*args, **kwargs)
        try:
            report = self.report_model.model_validate(parse_json_object(reply, report_name=self.name))
        except ValueError as exc:
            logger.warning(
                "%s returned invalid structured output; requesting one format-only correction: %s",
                self.name,
                exc,
            )
            model_input = await self._prepare_model_input()
            correction = UserMsg(
                name="home_report_recovery",
                content=(
                    "Your previous report did not conform to the required JSON schema. Using only the "
                    "analysis and evidence already present in this conversation, "
                    "immediately return the final decision. "
                    "Do not inspect more files, call tools, add Markdown, or include any other commentary."
                ),
            )
            # Do not forward tools: this call only reformats the existing conclusion.
            corrected = await self.model.generate_structured_output(
                messages=[*model_input["messages"], correction],
                structured_model=self.report_model,
                **_thinking_disabled_kwargs(self.model),
            )
            report = self.report_model.model_validate(corrected.content)
            usage = None
            if reply.usage is not None and corrected.usage is not None:
                usage = Usage(
                    **{key: getattr(reply.usage, key) + getattr(corrected.usage, key, 0) for key in Usage.model_fields},
                )
            reply = AssistantMsg(name=self.name, content=report.model_dump_json(), usage=usage)
            self.state.context.extend([correction, reply])
            return reply
        return reply.model_copy(
            update={"content": AssistantMsg(name=self.name, content=report.model_dump_json()).content},
        )


def parse_json_object(message: Msg, *, report_name: str) -> dict[str, Any]:
    """Parse an object with conservative repairs for common model formatting errors."""

    text = message.get_text_content() or ""
    candidates = [text.strip()]
    without_thinking = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if without_thinking and without_thinking != candidates[0]:
        candidates.append(without_thinking)
    fenced_candidates = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", without_thinking, flags=re.DOTALL | re.IGNORECASE)
    ]
    if len(fenced_candidates) == 1:
        candidates.extend(fenced_candidates)
    object_candidates = _extract_json_objects(without_thinking)
    if len(object_candidates) == 1:
        candidates.extend(object_candidates)

    saw_non_object = False
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        repaired = _remove_trailing_commas(candidate)
        for attempt in (candidate,) if repaired == candidate else (candidate, repaired):
            try:
                payload = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
            saw_non_object = True
    if saw_non_object:
        raise ValueError(f"{report_name} report must be a JSON object")
    raise ValueError(f"{report_name} did not return a JSON object") from None


def _extract_json_objects(text: str) -> list[str]:
    """Extract balanced JSON-looking objects without interpreting braces inside strings."""

    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _remove_trailing_commas(text: str) -> str:
    """Remove commas immediately before closing JSON containers, outside strings."""

    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
        if character == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        output.append(character)
        index += 1
    return "".join(output)


def _thinking_disabled_kwargs(model: Any) -> dict[str, Any]:
    """Build per-request settings for a deterministic format-only correction."""

    extra_body = dict(getattr(model, "extra_body", None) or {})
    extra_body.pop("enable_thinking", None)
    extra_body.pop("thinking_budget", None)
    extra_body.update({"reasoning_effort": "none", "preserve_thinking": False})
    return {"extra_body": extra_body}
