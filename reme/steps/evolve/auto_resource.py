"""Unified processor router for automatic resource interpretation."""

import copy
import inspect

from ...components import R
from ...enumeration import ComponentEnum
from ..base_step import BaseStep
from ._auto_resource import BaseAutoResourceStep, _results_answer

_ProcessorSpec = str | dict
_IndexedChange = tuple[int, dict]
_ProcessorRoute = tuple[dict, type[BaseAutoResourceStep], list[_IndexedChange]]


@R.register("auto_resource_step")
class AutoResourceStep(BaseStep):
    """Route each resource change to one configured processor and aggregate once."""

    def __init__(self, **kwargs):
        router_options = dict(kwargs)
        super().__init__(**kwargs)
        self._router_options = router_options

    def _processor_spec(self, spec: _ProcessorSpec, step_cls: type[BaseAutoResourceStep]) -> dict:
        """Merge router options declared by the processor into its explicit Step spec."""
        params = {"backend": spec} if isinstance(spec, str) else dict(spec)
        inherited = {
            key: self._router_options[key]
            for key in step_cls.router_inherit_keys
            if key in self._router_options and self._router_options[key] is not None
        }
        return {**inherited, **params}

    def _processor_routes(self) -> list[_ProcessorRoute]:
        """Resolve configured processors and allocate their per-invocation batches."""
        raw_specs = self.dispatch_step_specs
        if not raw_specs:
            registry = self.app_context.registry if self.app_context is not None else R
            raw_specs = [
                backend
                for backend, step_cls in registry.get_all(ComponentEnum.STEP).items()
                if isinstance(step_cls, type)
                and issubclass(step_cls, BaseAutoResourceStep)
                and step_cls.resource_fallback
            ]
            if len(raw_specs) != 1:
                candidates = ", ".join(sorted(raw_specs)) or "none"
                raise RuntimeError(
                    "AutoResourceStep without dispatch_steps requires exactly one registered "
                    f"fallback resource processor; found: {candidates}",
                )
            self.logger.warning(
                f"[{self.name}] dispatch_steps omitted; using registered fallback processor={raw_specs[0]}",
            )

        routes: list[_ProcessorRoute] = []
        fallback_indexes: list[int] = []
        for index, raw_spec in enumerate(raw_specs):
            step_cls, _ = self._resolve_dispatch_step(raw_spec)
            if not isinstance(step_cls, type) or not issubclass(step_cls, BaseAutoResourceStep):
                backend = raw_spec if isinstance(raw_spec, str) else raw_spec.get("backend", "")
                raise TypeError(f"Resource processor '{backend}' must inherit BaseAutoResourceStep")
            spec = self._processor_spec(raw_spec, step_cls)
            if step_cls.resource_fallback:
                fallback_indexes.append(index)
            routes.append((spec, step_cls, []))
        if len(fallback_indexes) > 1:
            raise ValueError("AutoResourceStep accepts at most one fallback processor")
        if fallback_indexes and fallback_indexes[0] != len(routes) - 1:
            raise ValueError("AutoResourceStep fallback processor must be last in dispatch_steps")
        return routes

    async def _dispatch_processor(
        self,
        spec: dict,
        indexed_changes: list[_IndexedChange],
        result_slots: list[dict | None],
    ) -> None:
        """Dispatch one routed sub-batch and snapshot its shared Response immediately."""
        if not indexed_changes:
            return
        changes = [item for _, item in indexed_changes]
        responses = await self.dispatch_steps([spec], changes=changes)
        processor_response = responses[-1]
        processor_results = copy.deepcopy(processor_response.metadata.get("results") or [])
        if len(processor_results) != len(indexed_changes):
            raise RuntimeError(
                f"Resource processor returned {len(processor_results)} result(s) "
                f"for {len(indexed_changes)} change(s)",
            )
        for (index, _), result in zip(indexed_changes, processor_results):
            result_slots[index] = result

    @staticmethod
    def _unsupported_result(item: dict, file_path: str) -> dict:
        """Return a stable failure result when no configured processor accepts a resource."""
        answer = f"No configured resource processor accepts: {file_path}"
        return {
            "success": False,
            "path": file_path,
            "change": str(item.get("change", "")),
            "answer": answer,
            "metadata": {
                "path": file_path,
                "action": "failed",
                "reason": "unsupported_resource",
                "modified": False,
            },
        }

    async def _emit_result_hook(self, *, changes: list[dict], results: list[dict]) -> None:
        """Notify embedding hosts once with the final aggregate response."""
        if self.app_context is None or self.context is None:
            return
        metadata = getattr(self.app_context, "metadata", None)
        if not isinstance(metadata, dict):
            return
        response_metadata = getattr(self.context.response, "metadata", None)
        if isinstance(response_metadata, dict) and response_metadata.get("modified") is False:
            self.logger.info(f"[{self.name}] result hook skipped; no resource note change modified=False")
            return
        hook = metadata.get("qwenpaw_memory_result_hook")
        if hook is None:
            return
        try:
            modified = response_metadata.get("modified") if isinstance(response_metadata, dict) else None
            self.logger.info(f"[{self.name}] result hook emit modified={modified}")
            value = hook(
                job_name="auto_resource",
                response=self.context.response,
                kwargs={"changes": changes},
                metadata={"results": results},
            )
            if inspect.isawaitable(value):
                await value
        except Exception:
            self.logger.exception(f"[{self.name}] result hook failed")

    async def execute(self):
        assert self.context is not None
        changes = self.context.get("changes")
        if not isinstance(changes, list):
            self.context.response.success = False
            self.context.response.answer = "AutoResourceStep requires changes: list[dict]"
            self.logger.warning(f"[{self.name}] invalid changes payload type={type(changes).__name__}")
            return self.context.response

        routes = self._processor_routes()
        fallback = next((route for route in routes if route[1].resource_fallback), None)
        specific_routes = [route for route in routes if not route[1].resource_fallback]
        result_slots: list[dict | None] = [None] * len(changes)
        for index, item in enumerate(changes):
            if not isinstance(item, dict):
                self.logger.warning(
                    f"[{self.name}] skip invalid change item index={index + 1} type={type(item).__name__}",
                )
                continue
            file_path = item.get("path") or item.get("file_path", "")
            target = next(
                (route for route in specific_routes if route[1].matches_change(item)),
                fallback,
            )
            if target is None:
                result_slots[index] = self._unsupported_result(item, file_path)
                continue
            target[2].append((index, item))

        route_counts = ", ".join(f"{spec['backend']}={len(batch)}" for spec, _, batch in routes)
        self.logger.info(f"[{self.name}] route changes={len(changes)} processors=({route_counts})")
        try:
            for spec, _, indexed_changes in routes:
                await self._dispatch_processor(spec, indexed_changes, result_slots)
        finally:
            # dispatch_steps merges the sub-batch into the shared context.
            # Downstream steps and the result hook must see the original batch.
            self.context["changes"] = changes

        results = [item for item in result_slots if item is not None]
        success_count = sum(1 for item in results if item.get("success"))
        self.context.response.success = len(results) == len(changes) and success_count == len(changes)
        processed_answer = f"Processed {success_count}/{len(changes)} resource change(s)"
        self.context.response.answer = _results_answer(results, processed_answer)
        self.context.response.metadata = {
            "processed": len(results),
            "results": results,
            "modified": any(bool((item.get("metadata") or {}).get("modified")) for item in results),
        }
        await self._emit_result_hook(changes=changes, results=results)
        self.logger.info(
            f"[{self.name}] done success={success_count}/{len(changes)} "
            f"processed={len(results)} modified={self.context.response.metadata['modified']}",
        )
        return self.context.response
