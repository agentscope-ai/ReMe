"""Demo steps for integration-testing the application stack."""

from ..base_step import BaseStep
from ...components import R


@R.register("demo_echo_step1")
class DemoEchoStep1(BaseStep):
    """Read query/min_score from context, normalize, and write back for Step2."""

    async def execute(self):
        assert self.context is not None
        query = self.context.get("query", "")
        min_score = self.context.get("min_score", 0.5)

        self.logger.info(f"[{self.name}] query={query!r}, min_score={min_score}")

        processed_query = query.strip().lower()
        adjusted_min_score = float(min_score) * 0.9

        self.context["processed_query"] = processed_query
        self.context["adjusted_min_score"] = adjusted_min_score

        return self.context.response


@R.register("demo_echo_step2")
class DemoEchoStep2(BaseStep):
    """Consume Step1's outputs from context and emit the final response."""

    async def execute(self):
        assert self.context is not None
        query = self.context.get("query", "")
        min_score = self.context.get("min_score", 0.5)
        processed_query = self.context.get("processed_query", "")
        adjusted_min_score = self.context.get("adjusted_min_score", min_score)

        self.logger.info(
            f"[{self.name}] query={query!r}, min_score={min_score}, "
            f"processed_query={processed_query!r}, adjusted_min_score={adjusted_min_score}",
        )

        self.context.response.answer = f"echo: {processed_query} (min_score={adjusted_min_score})"
        self.context.response.metadata.update(
            {
                "step": self.name,
                "query": query,
                "min_score": min_score,
                "processed_query": processed_query,
                "adjusted_min_score": adjusted_min_score,
            },
        )
        return self.context.response
