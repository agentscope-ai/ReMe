"""One bounded, read-only Context Agent evaluation shared with host coordination."""

import asyncio
from pathlib import Path
import time

from ..base_step import BaseStep, Ref
from ...components.as_classifier import BaseAsClassifier
from ...components.component_registry import R
from ...enumeration import ComponentEnum
from ...schema.context_agent import ContextAgentOptions, ContextEvaluationResult, ContextEvent
from .candidates import discover_candidates, evidence_is_current
from .policy import answer_probabilities, build_hint, fit_request, select_evidence, usage_dict


async def evaluate_context(event, options, *, store, classifier, workspace: Path, allowed_paths=None):
    """Evaluate a snapshot without changing messages, memory, or search-seen state."""
    started = time.monotonic()
    result = ContextEvaluationResult(
        event_id=event.event_id,
        task_id=event.task_id,
        based_on_revision=event.revision,
        reason_code="no_candidates",
    )
    async with asyncio.timeout(options.total_timeout_ms / 1000):
        candidates = await discover_candidates(store, workspace, event, options, allowed_paths)
        result.stats["retrieved_candidates"] = len(candidates)
        result.stats["retrieval_ms"] = round((time.monotonic() - started) * 1000, 2)
        result.stats["classifier_calls"] = 0
        if candidates:
            state, questions, fitted, estimate = fit_request(event, candidates, options)
            before_classifier = time.monotonic()
            response = await classifier(state=state, questions=questions)
            probabilities = answer_probabilities(response, questions)
            evidence = select_evidence(fitted, probabilities, options)
            result.stats.update(
                evaluated_candidates=len(fitted),
                estimated_request_tokens=estimate,
                estimate_method="utf8_bytes_upper_estimate",
                classifier_calls=1,
                classifier_ms=round((time.monotonic() - before_classifier) * 1000, 2),
                usage=usage_dict(response),
                candidate_scores={
                    c.path: {
                        "relevant": probabilities[f"relevant_c{i}"],
                        "useful": probabilities[f"useful_c{i}"],
                    }
                    for i, c in enumerate(fitted)
                },
                intervene_probability=probabilities["intervene_now"],
            )
            result.reason_code = "no_incremental_value"
            if evidence:
                if await evidence_is_current(workspace, evidence, allowed_paths, options.max_file_bytes):
                    result = result.model_copy(
                        update={
                            "action": "hint",
                            "hint": build_hint(event, evidence, time.time() + options.hint_ttl_seconds),
                        },
                    )
                    result.reason_code = "selected"
                else:
                    result.reason_code = "source_changed"
        result.stats["total_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result


@R.register("evaluate_context_step")
class EvaluateContextStep(BaseStep):
    """Return the full typed evaluation in Response.answer; no persistent Step state."""

    as_classifier = Ref(BaseAsClassifier, ComponentEnum.AS_CLASSIFIER)

    async def execute(self):
        assert self.context is not None
        event = ContextEvent.model_validate(self.context.get("event"))
        options = ContextAgentOptions.model_validate(self.context.get("options", self.kwargs.get("options", {})))
        result = await evaluate_context(
            event,
            options,
            store=self.file_store,
            classifier=self.as_classifier,
            workspace=self.workspace_path,
            allowed_paths=self.context.get("_allowed_paths"),
        )
        self.context.response.answer = result.model_dump(mode="json")
        self.context.response.metadata["context_agent"] = {
            "action": result.action,
            "reason_code": result.reason_code,
            "total_ms": result.stats["total_ms"],
        }
        return self.context.response
