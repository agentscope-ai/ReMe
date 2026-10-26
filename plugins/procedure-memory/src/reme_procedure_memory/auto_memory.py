"""Progressive trajectory extraction using controlled ReMe Job tools."""

import json
from uuid import uuid4

from reme.schema import Response
from reme.steps.evolve._evolve import agent_reply_result_text

from .checkpoint import atomic_json, digest, json_text, load_json
from .schema import ProcessingCheckpoint, SegmentResult
from .search import ProcedureStep
from .store import ProcedureIndexError
from .trajectory import SEGMENTATION_VERSION, prepare_text, segments

TOOLS = ["procedure_search", "procedure_read", "procedure_create", "procedure_update"]


class ProcedureAutoMemoryStep(ProcedureStep):
    """One file per invocation, serialized by an application-owned processing lock."""

    async def execute(self) -> Response:
        """Save one input file and run or resume its progressive extraction."""
        store = self.procedure_memory
        request_id = self.context.get("request_id") or uuid4().hex
        if not isinstance(request_id, str) or len(request_id) > 1024:
            raise ValueError("request_id must be a string of at most 1024 characters")
        async with store.processing_lock:
            record = store.import_trajectory(self.context["file"], self.context["filename"], request_id)
            source = record["source"]
            try:
                checkpoint = await self._process(record)
                return self.respond(
                    {
                        "request_id": request_id,
                        **checkpoint.model_dump(),
                        "status": "complete" if checkpoint.complete else "incomplete",
                    },
                    success=checkpoint.complete,
                )
            except Exception as exc:
                # The source copy and journals survive validation, model and index failures.
                return self.respond(
                    {"request_id": request_id, "source": source, "status": "incomplete", "error": str(exc)},
                    success=False,
                )

    def _fingerprint(self) -> str:
        store = self.procedure_memory
        return digest(
            json_text(
                {
                    "version": 1,
                    "segmentation": SEGMENTATION_VERSION,
                    "tokenizer": store.budget.source,
                    # Raising an execution context budget must remain a valid
                    # way to resume; it does not alter the saved segmentation.
                    "limits": {
                        key: value for key, value in store.budget.limits.items() if key not in {"context", "reserve"}
                    },
                    "memory_hint": self.context.get("memory_hint", ""),
                    "source_id": self.context.get("source_id", ""),
                    "system_prompt": self.get_prompt("system_prompt"),
                },
            ),
        )

    async def _process(self, record: dict) -> ProcessingCheckpoint:
        store = self.procedure_memory
        source = record["source"]
        path = store.source_path(source)
        checksum = record["input"]["sha256"]
        store.verify_source(source, checksum)
        await store.reconcile()
        fingerprint = self._fingerprint()
        checkpoint_path = store.checkpoint_path(source)
        if checkpoint_path.exists():
            checkpoint = ProcessingCheckpoint.model_validate(load_json(checkpoint_path))
            if checkpoint.sha256 != checksum or checkpoint.fingerprint != fingerprint:
                raise ValueError("Checkpoint input or processing configuration changed; cannot resume")
        else:
            checkpoint = ProcessingCheckpoint(source=source, sha256=checksum, fingerprint=fingerprint)
            atomic_json(checkpoint_path, checkpoint.model_dump())
        if checkpoint.complete:
            return checkpoint
        encoding = await prepare_text(path)
        for number, segment in enumerate(segments(path, encoding, store.budget)):
            if number < checkpoint.next_segment:
                continue
            scope = store.open_scope(source, number)
            try:
                result = await self._extract(checkpoint, segment, scope)
                store.budget.check("summary", result.summary)
                try:
                    await store.reconcile(source=source, segment=number)
                except ProcedureIndexError:
                    # A cached "complete" reply cannot repair a failed payload.
                    # Preserve its summary, but let the agent run again on resume.
                    checkpoint.active["last_summary"] = result.summary
                    checkpoint.active.pop("reply", None)
                    atomic_json(checkpoint_path, checkpoint.model_dump())
                    raise
                if not result.complete:
                    checkpoint.active["last_summary"] = result.summary
                    checkpoint.active.pop("reply", None)
                    atomic_json(checkpoint_path, checkpoint.model_dump())
                    return checkpoint
                checkpoint.summary = result.summary
                checkpoint.next_segment = number + 1
                checkpoint.active = {}
                atomic_json(checkpoint_path, checkpoint.model_dump())
            finally:
                store.scopes.pop(scope, None)
        checkpoint.complete = True
        atomic_json(checkpoint_path, checkpoint.model_dump())
        return checkpoint

    def _check_context(self, message: str, tools: list[str]) -> None:
        budget = self.procedure_memory.budget
        schemas = []
        for name in tools:
            job = self.get_job(name)
            schemas.append({"name": name, "description": job.description, "parameters": job.parameters})
        size = budget.count(self.get_prompt("system_prompt") + message + json_text(schemas))
        if size + budget.limits["reserve"] > budget.limits["context"]:
            raise ValueError(
                "Segment, summary and tool context exceed the context budget; processing remains resumable",
            )

    async def _extract(self, checkpoint: ProcessingCheckpoint, segment: dict, scope: str) -> SegmentResult:
        store = self.procedure_memory
        checkpoint_path = store.checkpoint_path(checkpoint.source)
        if not checkpoint.active:
            checkpoint.active = {
                "segment": checkpoint.next_segment,
                "previous_summary": checkpoint.summary,
                "offset": segment["offset"],
                "segment_hash": digest(segment["text"]),
            }
            atomic_json(checkpoint_path, checkpoint.model_dump())
        if checkpoint.active["segment_hash"] != digest(segment["text"]):
            raise ValueError("Segment differs from saved tool context")
        recovery = store.operation_records(checkpoint.source, checkpoint.next_segment)
        message = json_text(
            {
                "source": checkpoint.source,
                "segment": checkpoint.next_segment,
                "offset": segment["offset"],
                "trajectory_text": segment["text"],
                "previous_summary": checkpoint.active["previous_summary"],
                "unfinished_summary": checkpoint.active.get("last_summary", ""),
                "completed_operations": recovery,
                "index_recovery": (
                    "Search results with index_complete=false exclude pending_indexes paths. "
                    "Read a pending path and correct it with procedure_update, its current expected_version "
                    "and a new operation_key. This supersedes the failed operation; keep the filename. "
                    "Unrelated pending paths do not block this trajectory."
                ),
                "memory_hint": self.context.get("memory_hint", ""),
                "token_limits": store.budget.limits,
            },
        )
        self._check_context(message, TOOLS)
        # Persisted input + tool journal form the recovery context even if the SDK
        # crashes before saving its own agent session. Do not assume identical output.
        checkpoint.active["input"] = message
        atomic_json(checkpoint_path, checkpoint.model_dump())
        raw = checkpoint.active.get("reply")
        attempts = int(self.kwargs.get("repair_attempts", 2))
        if attempts < 0:
            raise ValueError("repair_attempts must be nonnegative")
        for attempt in range(attempts + 1):
            if raw is None:
                reply = await self.agent_wrapper.reply(
                    message,
                    system_prompt=self.get_prompt("system_prompt"),
                    job_tools=TOOLS if attempt == 0 else [],
                    builtin_tools=[],
                    skills=[],
                    injected_job_kwargs={"_procedure_scope": scope},
                    sequential_tool_calls=True,
                )
                reason = (reply.get("last_message") or {}).get("finished_reason")
                if attempt == 0:
                    checkpoint.active["execution_incomplete"] = reason in {"interrupted", "exceed_max_iters", "error"}
                elif reason in {"interrupted", "exceed_max_iters", "error"}:
                    checkpoint.active["execution_incomplete"] = True
                raw = agent_reply_result_text(reply)
                checkpoint.active["reply"] = raw
                atomic_json(checkpoint_path, checkpoint.model_dump())
            try:
                result = SegmentResult.model_validate(json.loads(raw))
                store.budget.check("summary", result.summary)
                if checkpoint.active.get("execution_incomplete"):
                    result.complete = False
                return result
            except ValueError as exc:
                if attempt == attempts:
                    # Keep all committed operations, but permit a fresh continuation.
                    checkpoint.active.pop("reply", None)
                    atomic_json(checkpoint_path, checkpoint.model_dump())
                    raise ValueError(f"Invalid segment summary after bounded repairs: {exc}") from exc
                message = json_text(
                    {
                        "instruction": "Repair only the final JSON and bounded summary. Do not perform any writes.",
                        "error": str(exc),
                        "previous_response": raw,
                        "summary_token_limit": store.budget.limits["summary"],
                    },
                )
                self._check_context(message, [])
                raw = None
        raise RuntimeError("Unreachable summary repair state")
