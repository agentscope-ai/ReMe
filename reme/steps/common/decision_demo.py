"""Demo for typed probabilistic decisions over text or JSON state."""

import json
from dataclasses import asdict

from ..base_step import BaseStep, Ref
from ...components import R
from ...components.as_decision_model import BaseAsDecisionModel
from ...enumeration import ComponentEnum


@R.register("decision_demo_step")
class DecisionDemoStep(BaseStep):
    """Return complete decision model output and share it with subsequent Steps."""

    as_decision_model = Ref(BaseAsDecisionModel, ComponentEnum.AS_DECISION_MODEL)

    @staticmethod
    def default_questions() -> dict:
        """Create independent demo questions for each invocation."""
        return {
            "urgent": {"type": "binary", "instructions": "Does this customer request need urgent attention?"},
            "route": {
                "type": "choice",
                "instructions": "Select the team that should handle this customer request.",
                "criteria": {"billing": "Payments and charges", "support": "Technical help", "other": "Other requests"},
            },
            "priority": {
                "type": "score",
                "instructions": "Rate the priority of this customer request.",
                "criteria": ["Low", "Medium", "High"],
            },
        }

    async def execute(self):
        assert self.context is not None
        # Explicit null/empty questions are invalid; only omission enables the demo defaults.
        questions = self.context.get("questions", self.default_questions())
        result = await self.as_decision_model(state=self.context.get("state"), questions=questions)
        answer = json.loads(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False))
        self.context.data["decision_result"] = answer
        self.context.response.answer = answer
        return self.context.response
