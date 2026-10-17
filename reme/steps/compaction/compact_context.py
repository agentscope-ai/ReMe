"""Propose recoverable tool-result previews without changing source history."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math
import time

from ..base_step import BaseStep, Ref
from ...components import R
from ...components.as_decision_model import BaseAsDecisionModel
from ...enumeration import ComponentEnum
from ...schema.compaction import CallDecision, CompactionRequest, CompactionResult
from .messages import collect_pairs, preview_result
from .state import build_batches, estimate_decision_tokens, serialize


def probability(answer) -> float | None:
    """Malformed or missing answers express no usable judgment."""
    value = getattr(answer, "probability", None)
    if (
        getattr(answer, "type", None) == "binary_answer"
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    ):
        return float(value)
    return None


@R.register("compact_context_step")
class CompactContextStep(BaseStep):
    """The host may inject ``token_counter(messages)`` to enforce its full input budget."""

    as_decision_model = Ref(BaseAsDecisionModel, ComponentEnum.AS_DECISION_MODEL)

    async def execute(self):
        started = time.monotonic()
        request = CompactionRequest.model_validate(
            {key: self.context[key] for key in CompactionRequest.model_fields if key in self.context},
        )
        options = request.options
        counter = self.kwargs.get("token_counter")
        if options.max_output_tokens is not None and not callable(counter):
            raise ValueError("max_output_tokens requires a host token_counter(messages).")

        def count(messages):
            if options.max_output_tokens is None:
                return None
            value = counter(messages)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("token_counter must return a non-negative integer.")
            return value

        before = count(request.messages)
        within_budget = before is not None and before <= options.max_output_tokens
        pairs = collect_pairs(request)
        batches = [] if within_budget else build_batches(request, pairs)
        answers, failures, usage = {}, {}, []
        semaphore = asyncio.Semaphore(options.max_concurrency)

        async def ask(state, questions):
            try:
                async with semaphore:
                    response = await self.as_decision_model(state=state, questions=questions)
                batch_answers = {name: probability(response.content.get(name)) for name in questions}
                if response.usage is not None:
                    usage.append(asdict(response.usage) if is_dataclass(response.usage) else response.usage)
                answers.update(batch_answers)
            except Exception:  # A provider failure must never become evidence for removal.
                failures.update(dict.fromkeys(questions, "model_error"))

        try:
            async with asyncio.timeout(options.total_timeout_seconds):
                async with asyncio.TaskGroup() as tasks:
                    for state, questions in batches:
                        tasks.create_task(ask(state, questions))
        except TimeoutError:
            for _, questions in batches:
                failures.update({name: "timeout" for name in questions if name not in answers})
        decisions = [
            CallDecision(
                id=pair.id,
                call_id=pair.call_id,
                action="keep",
                reason=(
                    "protected"
                    if pair.protected
                    else (
                        "within_budget"
                        if within_budget
                        else failures.get(pair.id, "invalid_answer" if pair.id in answers else "unobserved")
                    )
                ),
                probability=answers.get(pair.id),
                source_ref=pair.source_ref,
            )
            for pair in pairs
        ]
        messages, after = request.messages, before
        ranked = sorted(
            (d for d in decisions if d.probability is not None and d.probability <= options.drop_threshold),
            key=lambda d: d.probability,
        )
        for decision in decisions:
            if decision.probability is not None:
                decision.reason = "needed_or_uncertain"
        proposed = deepcopy(request.messages) if ranked else request.messages
        by_id = {pair.id: pair for pair in pairs}
        for decision in ranked:
            if after is not None and after <= options.max_output_tokens:
                decision.reason = "within_budget"
                continue
            pair = by_id[decision.id]
            mi, bi = pair.result_location
            block = proposed[mi]["content"][bi]
            replacement = preview_result(block, pair.source_ref, options.preview_chars)
            if replacement is None:
                decision.reason = "no_savings"
                continue
            original_output, block["output"] = block["output"], replacement
            proposed_tokens = count(proposed)
            if after is not None and proposed_tokens >= after:
                block["output"] = original_output
                decision.reason = "no_savings"
                continue
            decision.action, decision.reason = "preview", "low_relevance"
            messages, after = proposed, proposed_tokens
        changed = messages != request.messages
        budget_met = None if after is None else after <= options.max_output_tokens
        stats = {
            "messages_before": len(request.messages),
            "messages_after": len(messages),
            "chars_before": len(serialize(request.messages)),
            "chars_after": len(serialize(messages)),
            "tokens_before": before,
            "tokens_after": after,
            "candidates": sum(not pair.protected for pair in pairs),
            "protected": sum(pair.protected for pair in pairs),
            "state_tokens": max((estimate_decision_tokens(state) for state, _ in batches), default=0),
            "requests": len(batches),
            "usage": usage,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        result = CompactionResult(
            messages=messages,
            decisions=decisions,
            changed=changed,
            reason=(
                "budget_exceeded"
                if budget_met is False
                else "compacted" if changed else "no_candidates" if not stats["candidates"] else "no_change"
            ),
            budget_met=budget_met,
            stats=stats,
        )
        self.context.response.answer = result.model_dump(mode="json")
        self.context.response.metadata["compaction"] = stats
        return self.context.response
