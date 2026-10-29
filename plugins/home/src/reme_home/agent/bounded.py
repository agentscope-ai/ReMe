"""AgentScope agent that turns an exhausted ReAct budget into a final answer."""

from __future__ import annotations

import inspect
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable

from agentscope.agent import Agent
from agentscope.message import HintBlock, Msg, TextBlock, UserMsg
from agentscope.model import ChatResponse, FinishedReason
from agentscope.tool import ToolChoice

logger = logging.getLogger(__name__)

_REMINDER_RATIOS = (0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class TimeBudgetReminder:
    """One reminder plus whether it should survive the current reasoning call."""

    content: str
    persistent: bool


class RunTimeBudgetTimer:
    """Track one agent run and emit advisory reminders without stopping it."""

    def __init__(
        self,
        time_budget_minutes: float,
        *,
        agent_label: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not isinstance(time_budget_minutes, (int, float))
            or isinstance(time_budget_minutes, bool)
            or not math.isfinite(time_budget_minutes)
            or time_budget_minutes <= 0
        ):
            raise ValueError("time_budget_minutes must be a positive number")
        if not isinstance(agent_label, str) or not agent_label.strip():
            raise ValueError("agent_label must be a non-empty string")
        self.time_budget_minutes = float(time_budget_minutes)
        self.agent_label = agent_label.strip()
        self._clock = clock
        self._started_at: float | None = None
        self._last_reminded_ratio = 0.0
        self._next_reminder_is_persistent = False
        self.context_compression_count = 0

    @property
    def active(self) -> bool:
        """Return whether a run is currently being timed."""

        return self._started_at is not None

    def start(self) -> None:
        """Start a fresh run timer."""

        self._started_at = self._clock()
        self._last_reminded_ratio = 0.0
        self._next_reminder_is_persistent = False
        self.context_compression_count = 0

    def stop(self) -> None:
        """Stop timing the current run."""

        self._started_at = None

    def record_context_compression(self) -> None:
        """Record one completed context compression in the current run."""

        self.context_compression_count += 1

    def next_reminder(self) -> TimeBudgetReminder | None:
        """Return the latest newly reached soft-budget reminder."""

        if self._started_at is None:
            return None
        elapsed_minutes = max(0.0, (self._clock() - self._started_at) / 60.0)
        elapsed_ratio = elapsed_minutes / self.time_budget_minutes
        reached_ratio: float | None = None
        if elapsed_ratio >= 1.0:
            reached_ratio = math.floor(elapsed_ratio * 10 + 1e-9) / 10
        else:
            for threshold in (0.90, 0.75, 0.60):
                if elapsed_ratio >= threshold:
                    reached_ratio = threshold
                    break
        if reached_ratio is None or reached_ratio <= self._last_reminded_ratio:
            return None
        self._last_reminded_ratio = reached_ratio
        persistent = self._next_reminder_is_persistent
        self._next_reminder_is_persistent = not persistent
        elapsed = _format_minutes(elapsed_minutes)
        percentage = round(reached_ratio * 100)
        return TimeBudgetReminder(
            content=(
                f"<system-reminder>The {self.agent_label} run has used {elapsed} ({percentage}% of its "
                f"{_format_minutes(self.time_budget_minutes)} soft time budget). "
                f"The run has completed {self.context_compression_count} context compressions. "
                "This reminder does not terminate the run. Prioritize the highest-value remaining investigation and "
                "produce the required final report as soon as the evidence is sufficient.</system-reminder>"
            ),
            persistent=persistent,
        )


def _format_minutes(minutes: float) -> str:
    """Format a timer duration compactly."""

    if minutes < 1:
        return f"{max(0, round(minutes * 60))} seconds"
    rounded = round(minutes, 1)
    value = int(rounded) if rounded.is_integer() else rounded
    unit = "minute" if value == 1 else "minutes"
    return f"{value} {unit}"


def _remove_hint_by_identity(messages: list[Msg], target: HintBlock) -> None:
    """Remove only the temporary hint, retaining the reply message and its usage."""
    for message in messages:
        for index, block in enumerate(message.content):
            if block is target:
                del message.content[index]
                return


class BoundedReActAgent(Agent):
    """Add advisory budgets and safe finalization to AgentScope's bounded loop.

    AgentScope owns iteration accounting, tool execution, events and cancellation.
    Its one final reasoning call after ``max_iters`` receives no tools. Native
    structured-output requests retain AgentScope's schema and grace-iteration rules.
    Instances, like AgentScope agents, must not serve concurrent replies.
    """

    def __init__(self, *args: Any, run_time_budget_timer: RunTimeBudgetTimer | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_time_budget_timer = run_time_budget_timer
        self._sent_iteration_reminders: set[int] = set()
        self._original_task_prompt = ""

    async def _reply_impl(self, inputs: Any = None, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Own the advisory timer for this invocation, including error cleanup."""
        self._sent_iteration_reminders.clear()
        if isinstance(inputs, Msg):
            self._original_task_prompt = inputs.get_text_content() or ""
        elif isinstance(inputs, list):
            self._original_task_prompt = "\n\n".join(message.get_text_content() or "" for message in inputs)
        elif inputs is None:
            # ReMe observes inputs before reply(); retain that task for finalization.
            self._original_task_prompt = next(
                (
                    message.get_text_content() or ""
                    for message in reversed(self.state.context)
                    if message.role == "user"
                ),
                "",
            )
        timer = self.run_time_budget_timer
        if timer is not None:
            timer.start()
        try:
            async for item in super()._reply_impl(inputs=inputs, **kwargs):
                yield item
        finally:
            if timer is not None:
                timer.stop()

    async def compress_context(self, *args: Any, **kwargs: Any) -> None:
        """Count successful compressions; allow final reports after compression failure."""
        summary_before, context_before = self.state.summary, self.state.context
        try:
            await super().compress_context(*args, **kwargs)
        except Exception as exc:
            if not self._is_finalizing():
                raise
            logger.warning(
                "Compression before the final report failed for Home agent %s; attempting finalization: %s",
                self.name,
                type(exc).__name__,
            )
            return
        if self.run_time_budget_timer is not None and (
            self.state.summary is not summary_before or self.state.context is not context_before
        ):
            self.run_time_budget_timer.record_context_compression()

    async def _reasoning(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Deliver each iteration threshold once and alternate time reminders."""
        timer = self.run_time_budget_timer
        reminder = timer.next_reminder() if timer is not None else None
        reminder_hint = None
        if reminder is not None:
            reminder_hint = HintBlock(source="home_run_time_guard", hint=reminder.content)
            self.state.append_context(self.name, [reminder_hint])
        completed = self.state.cur_iter
        thresholds = {math.ceil(self.react_config.max_iters * ratio) for ratio in _REMINDER_RATIOS}
        reached = thresholds - self._sent_iteration_reminders
        reached = {threshold for threshold in reached if threshold <= completed}
        if reached:
            remaining = max(0, self.react_config.max_iters - completed)
            self.state.append_context(
                self.name,
                [
                    HintBlock(
                        source="home_iteration_guard",
                        hint=(
                            f"<system-reminder>You have completed {completed} reasoning-acting iterations; "
                            f"{remaining} iteration(s) remain.</system-reminder>"
                        ),
                    ),
                ],
            )
            self._sent_iteration_reminders.update(reached)
        try:
            async for item in super()._reasoning(*args, **kwargs):
                yield item
        finally:
            if reminder is not None and not reminder.persistent and reminder_hint is not None:
                _remove_hint_by_identity(self.state.context, reminder_hint)

    def _is_finalizing(self) -> bool:
        """Only the iteration cap triggers finalization; elapsed time is advisory."""
        return self.state.cur_iter >= self.react_config.max_iters and self.state.reply_context.structured_schema is None

    async def _call_model(self, messages: list[Msg], tools: list[dict], tool_choice: ToolChoice | None = None):
        """Use the native finalization slot; never start a second synthesis call."""
        if not self._is_finalizing():
            return await super()._call_model(messages=messages, tools=tools, tool_choice=tool_choice)

        last_text = next(
            (
                message.get_text_content()
                for message in reversed(self.state.context)
                if message.role == "assistant" and message.name == self.name and message.get_text_content()
            ),
            "(no previous displayable model response)",
        )
        messages = list(messages)
        messages.append(
            UserMsg(
                name="home_iteration_guard",
                content=(
                    "The iteration budget is exhausted. Return the best final report from collected evidence. "
                    "Do not call tools or propose more investigation. Distinguish findings from uncertainty.\n"
                    f"Original task: {self._original_task_prompt or '(see the conversation above)'}"
                ),
            ),
        )
        response_usage = None
        try:
            limit = int(self.model.context_size * self.context_config.trigger_ratio)
            while await self.model.count_tokens(messages=messages, tools=[]) >= limit:
                if len(messages) <= 2:
                    raise ValueError("Final instructions exceed the model context budget")
                del messages[1]
            response = await super()._call_model(messages=messages, tools=[], tool_choice=ToolChoice(mode="none"))
            response = await self._consume_final_response(response)
            response_usage = response.usage
            if response.finished_reason == FinishedReason.INTERRUPTED:
                return response
            content = [block for block in response.content if isinstance(block, TextBlock) and block.text.strip()]
            if not content:
                raise ValueError("Final model response contains no displayable answer")
            # Ignore unsolicited tool calls, even when accompanied by valid text.
            response.content = content
            return response
        except Exception as exc:
            logger.exception("Final report failed for Home agent %s", self.name)
            return ChatResponse(
                content=[
                    TextBlock(
                        text=(
                            f"Reached the maximum of {self.react_config.max_iters} reasoning-acting iterations. "
                            f"Final report generation failed ({type(exc).__name__}); "
                            f"the following is incomplete prior model output:\n\n{last_text}"
                        ),
                    ),
                ],
                is_last=True,
                usage=response_usage,
            )

    @staticmethod
    async def _consume_final_response(response: ChatResponse | AsyncGenerator[ChatResponse, None]) -> ChatResponse:
        """Require a completed response for streaming and non-streaming models."""
        if isinstance(response, ChatResponse):
            return response
        if not inspect.isasyncgen(response):
            raise TypeError(f"Unexpected final response: {type(response)}")
        completed = None
        try:
            async for chunk in response:
                if chunk.is_last:
                    completed = chunk
        finally:
            await response.aclose()
        if completed is None:
            raise ValueError("Final response stream ended without a completed response")
        return completed
