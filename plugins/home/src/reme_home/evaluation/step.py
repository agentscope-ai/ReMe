"""Expose immutable batch evaluation through ReMe's normal Job lifecycle."""

from reme.steps.base_step import BaseStep, Ref

from .component import HomeEvaluation
from .inputs import EvaluationSpec
from .repetition import UnchangedCandidateError


class HomeEvaluateStep(BaseStep):
    """One evaluation Job with optional host-configured adaptive sampling."""

    evaluation: HomeEvaluation = Ref(HomeEvaluation, "home.evaluation")

    async def execute(self):
        """Evaluate via the Docker pool and persist host-side scores and evidence."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("Home evaluation Jobs require unrestricted access to their workspace output directories")
        arguments = {name: self.context.get(name) for name in EvaluationSpec.model_fields if name in self.context}
        # Do not silently accept a removed toggle that conflicts with reuse paths.
        if "eval_only" in self.context:
            raise ValueError("Use reuse_construction_from instead of eval_only")
        if "code_ref" in self.context:
            raise ValueError("code_ref is unsupported; commit candidate changes and evaluate the current HEAD")
        if "judge_concurrency" in self.context:
            raise ValueError("Use concurrency to control both case and judge concurrency")
        if "dataset_kind" in self.context:
            raise ValueError("Use debug=true for debug cases; omit debug or use false for training cases")
        try:
            result = await self.evaluation.evaluate(**arguments)
        except UnchangedCandidateError as error:
            self.context.response.answer = {
                "status": "rejected",
                "error_code": "UNCHANGED_CANDIDATE",
                "error": str(error),
                "result_dir": str(error.path),
            }
            self.context.response.success = False
            self.context.response.metadata["result_dir"] = str(error.path)
            return self.context.response
        self.context.response.answer = result
        if result.get("status") == "rejected":
            self.context.response.success = False
            return self.context.response
        # Candidate failures are evaluation observations; infrastructure failures
        # must remain visible to ordinary Job callers checking Response.success.
        self.context.response.success = result["case_status_counts"]["infra_error"] == 0
        if arguments.get("debug"):
            self.context.response.success &= result["case_status_counts"]["candidate_failure"] == 0
        self.context.response.metadata["result_dir"] = result["result_dir"]
        return self.context.response
