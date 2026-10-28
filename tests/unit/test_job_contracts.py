"""Job invocation isolation and streaming contracts for framework refactoring."""

import asyncio
from types import SimpleNamespace

from reme.components import ComponentRegistry
from reme.components.job import BaseJob, StreamJob
from reme.enumeration import ChunkEnum
from reme.steps import BaseStep
from reme.utils import global_counter_get


def test_concurrent_job_calls_use_fresh_steps_and_share_only_their_own_context():
    """Real Step chains share one request context while concurrent calls remain independent."""

    async def run():
        entered = asyncio.Event()
        observed = []
        registry = ComponentRegistry()

        class RecordingStep(BaseStep):
            """Pause both producers before consuming their request-local state."""

            async def execute(self):
                """Produce or consume state shared only by this invocation's Steps."""
                observed.append((self, self.context, self.context.response))
                if self.kwargs["role"] == "producer":
                    self.context["result"] = [self.context["value"]]
                    if len(observed) == 2:
                        entered.set()
                    await entered.wait()
                else:
                    self.context["result"].append("consumed")
                    self.context.response.answer = self.context["result"]

        registry.register(RecordingStep, "recording")
        app_context = SimpleNamespace(
            registry=registry,
            metadata={},
            app_config=SimpleNamespace(language=""),
        )
        job = BaseJob(
            name="recording",
            app_context=app_context,
            steps=[
                {"backend": "recording", "role": "producer"},
                {"backend": "recording", "role": "consumer"},
            ],
        )
        async with job:
            first, second = await asyncio.wait_for(
                asyncio.gather(job(value="first"), job(value="second")),
                timeout=5,
            )

        assert first.success and second.success
        assert first.answer == ["first", "consumed"]
        assert second.answer == ["second", "consumed"]
        assert len({id(step) for step, _, _ in observed}) == 4
        for value in ("first", "second"):
            invocation = [(ctx, response) for _, ctx, response in observed if ctx["value"] == value]
            assert len(invocation) == 2
            assert invocation[0][0] is invocation[1][0]
            assert invocation[0][1] is invocation[1][1]
        assert len({id(ctx) for _, ctx, _ in observed}) == 2
        assert first is not second
        first.metadata["caller"] = "first"
        assert second.metadata == {}
        first.answer.append("changed")
        assert second.answer == ["second", "consumed"]
        assert global_counter_get(app_context.metadata, ["__job_counter", "recording"]) == 2

    asyncio.run(run())


def test_stream_job_emits_error_then_one_done_when_a_real_step_raises():
    """A failing Step stops its chain and leaves one terminal marker for the stream consumer."""

    async def run():
        registry = ComponentRegistry()
        calls = []

        class FailingStep(BaseStep):
            """Emit progress before failing, or record an unexpected later execution."""

            async def execute(self):
                """Fail the producer so the following Step must remain uncalled."""
                calls.append(self.kwargs["role"])
                if self.kwargs["role"] == "fail":
                    await self.context.add_stream_string("before failure", ChunkEnum.CONTENT)
                    raise ValueError("rejected")

        registry.register(FailingStep, "failing")
        app_context = SimpleNamespace(
            registry=registry,
            metadata={},
            app_config=SimpleNamespace(language=""),
        )
        job = StreamJob(
            name="failing",
            app_context=app_context,
            steps=[
                {"backend": "failing", "role": "fail"},
                {"backend": "failing", "role": "after"},
            ],
        )
        queue = asyncio.Queue()
        async with job:
            await job(stream_queue=queue)
        chunks = [queue.get_nowait() for _ in range(queue.qsize())]
        assert calls == ["fail"]
        assert [chunk.chunk_type for chunk in chunks] == [ChunkEnum.CONTENT, ChunkEnum.ERROR, ChunkEnum.DONE]
        assert [chunk.chunk for chunk in chunks] == ["before failure", "rejected", ""]
        assert [chunk.done for chunk in chunks] == [False, False, True]

    asyncio.run(run())
