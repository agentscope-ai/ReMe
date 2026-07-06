"""Tests for the _local_instantiation_ mechanism in BaseJob."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
from unittest.mock import MagicMock

import pytest

from reme.components.component_registry import ComponentRegistry
from reme.components.job.base_job import BaseJob, _LOCAL_INSTANTIATION_KEY
from reme.enumeration import ComponentEnum
from reme.steps.base_step import BaseStep


# -- helpers ------------------------------------------------------------------

_CALL_LOG: list[tuple[str, int]] = []
_INSTANCE_COUNTER: int = 0


class _TrackingStep(BaseStep):
    """Step that logs a unique instance counter on every execute to detect reuse vs. re-creation."""

    component_type = ComponentEnum.STEP

    def __init__(self, **kwargs):
        global _INSTANCE_COUNTER
        super().__init__(**kwargs)
        _INSTANCE_COUNTER += 1
        self._instance_uid = _INSTANCE_COUNTER

    async def execute(self):
        _CALL_LOG.append((self.name, self._instance_uid))


def _make_job(step_configs: list[dict], registry: ComponentRegistry) -> BaseJob:
    """Create a BaseJob with a mocked app_context and a controlled registry."""
    job = BaseJob(name="test_job", steps=step_configs)
    job.app_context = MagicMock()
    job.app_context.components = {}
    # Patch the registry lookup to use our local registry.

    def patched_resolve(raw):
        from reme.schema import ComponentConfig

        config = raw if isinstance(raw, ComponentConfig) else ComponentConfig(**raw)
        step_cls = registry.get(ComponentEnum.STEP, config.backend)
        if not step_cls:
            raise ValueError(f"Unregistered backend '{config.backend}'")
        params = config.model_dump()
        params["app_context"] = job.app_context
        local_id = int(params.pop(_LOCAL_INSTANTIATION_KEY, 0))
        return step_cls, params, local_id

    job._resolve_step = patched_resolve
    return job


@pytest.fixture(autouse=True)
def reset_call_log():
    global _INSTANCE_COUNTER
    _CALL_LOG.clear()
    _INSTANCE_COUNTER = 0
    yield
    _CALL_LOG.clear()
    _INSTANCE_COUNTER = 0


@pytest.fixture
def registry():
    reg = ComponentRegistry()
    reg.register(_TrackingStep, "tracking")
    return reg


# -- Tests: default behavior (local_instantiation=0) --------------------------


def test_default_creates_new_instance_each_call(registry):
    """With _local_instantiation_=0 (default), each __call__ creates fresh step instances."""

    async def run():
        job = _make_job([{"backend": "tracking"}], registry)
        await job._start()

        await job()
        await job()

        # Two calls, each should have a different step instance id.
        assert len(_CALL_LOG) == 2
        assert _CALL_LOG[0][1] != _CALL_LOG[1][1]

    asyncio.run(run())


# -- Tests: persistent step (_local_instantiation_ > 0) -----------------------


def test_persistent_step_reuses_instance_across_calls(registry):
    """With _local_instantiation_=1, the same step instance is reused across calls."""

    async def run():
        job = _make_job([{"backend": "tracking", "_local_instantiation_": 1}], registry)
        await job._start()

        await job()
        await job()
        await job()

        # All three calls should use the same step instance.
        assert len(_CALL_LOG) == 3
        ids = {entry[1] for entry in _CALL_LOG}
        assert len(ids) == 1, "Expected all calls to reuse the same instance"

    asyncio.run(run())


def test_same_nonzero_value_shares_instance(registry):
    """Two steps with the same _local_instantiation_ value share one instance."""

    async def run():
        job = _make_job(
            [
                {"backend": "tracking", "_local_instantiation_": 2},
                {"backend": "tracking", "_local_instantiation_": 2},
            ],
            registry,
        )
        await job._start()
        await job()

        # Both slots ran but should be the same instance.
        assert len(_CALL_LOG) == 2
        assert _CALL_LOG[0][1] == _CALL_LOG[1][1]

    asyncio.run(run())


def test_different_nonzero_values_separate_instances(registry):
    """Different _local_instantiation_ values create independent persistent instances."""

    async def run():
        job = _make_job(
            [
                {"backend": "tracking", "_local_instantiation_": 1},
                {"backend": "tracking", "_local_instantiation_": 2},
            ],
            registry,
        )
        await job._start()
        await job()

        # Two different instances.
        assert len(_CALL_LOG) == 2
        assert _CALL_LOG[0][1] != _CALL_LOG[1][1]

    asyncio.run(run())


def test_mixed_persistent_and_ephemeral(registry):
    """Mix of persistent (non-zero) and ephemeral (zero) steps."""

    async def run():
        job = _make_job(
            [
                {"backend": "tracking", "_local_instantiation_": 1},
                {"backend": "tracking"},  # default=0, ephemeral
            ],
            registry,
        )
        await job._start()

        await job()
        await job()

        # 4 total calls: 2 calls * 2 steps each.
        assert len(_CALL_LOG) == 4
        # Persistent step (index 0 and 2) should share instance.
        persistent_ids = {_CALL_LOG[0][1], _CALL_LOG[2][1]}
        assert len(persistent_ids) == 1
        # Ephemeral step (index 1 and 3) should differ.
        ephemeral_ids = {_CALL_LOG[1][1], _CALL_LOG[3][1]}
        assert len(ephemeral_ids) == 2

    asyncio.run(run())


def test_local_instantiation_key_stripped_from_params(registry):
    """The _local_instantiation_ key must not appear in the step constructor params."""

    async def run():
        job = _make_job([{"backend": "tracking", "_local_instantiation_": 1}], registry)
        await job._start()

        # Check that the persistent step's kwargs don't contain the marker.
        persistent_step = job._persistent_steps[1]
        assert _LOCAL_INSTANTIATION_KEY not in persistent_step.kwargs

    asyncio.run(run())


def test_close_clears_persistent_steps(registry):
    """_close() should clear _persistent_steps dict."""

    async def run():
        job = _make_job([{"backend": "tracking", "_local_instantiation_": 1}], registry)
        await job._start()
        assert len(job._persistent_steps) == 1

        await job._close()
        assert len(job._persistent_steps) == 0
        assert len(job.step_specs) == 0

    asyncio.run(run())
