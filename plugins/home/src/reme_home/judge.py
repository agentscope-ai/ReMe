"""Adapt host-installed benchmark judge Jobs without loading candidate code."""

import hashlib
import json
from importlib import import_module
import math
from pathlib import Path

from reme.components.base_component import BaseComponent
from reme.schema import ComponentConfig

from .preparation.data.models import fingerprint
from .evaluation.artifacts import contained, read_json
from .workspace import HomeWorkspace

JUDGE_PLUGINS = {"beam": "beam-judge", "longmemeval": "lme-judge"}
_BACKENDS = {"beam-judge": "beam_rubric_judge_step", "lme-judge": "lme_answer_judge_step"}
_MODULES = {"beam-judge": "judge_beam", "lme-judge": "judge_lme"}


class JudgeResultError(ValueError):
    """Preserve the host Job's response when scoring or result validation fails."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class HomeJudge(BaseComponent):
    """Validate the host judge selection and adapt its standard answer_judge Job."""

    component_type = "home.judge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspace = self.bind("default", HomeWorkspace, optional=False)

    async def _start(self):
        self.require_benchmark()
        marker = contained(self.workspace.home_path, "metadata/preparation.json")
        if marker.is_file():
            self.require_benchmark(read_json(marker)["dataset"]["dataset"])

    def require_benchmark(self, benchmark=None):
        """Reject missing, extra, mismatched or overridden benchmark judge Jobs."""
        plugins = self.app_context.app_config.plugins
        judges = [name for name in plugins if name.endswith("-judge")]
        if len(judges) != 1 or judges[0] not in _BACKENDS:
            raise ValueError("Home requires exactly one supported judge plugin: beam-judge or lme-judge")
        plugin = judges[0]
        if benchmark is not None and JUDGE_PLUGINS.get(benchmark) != plugin:
            raise ValueError(f"Benchmark {benchmark!r} requires {JUDGE_PLUGINS.get(benchmark)!r}, enabled {plugin!r}")
        job = self.app_context.jobs.get("answer_judge")
        if job is None or not job.enabled:
            raise ValueError("Home requires an enabled answer_judge Job")
        steps = job.step_configs
        if len(steps) != 1 or ComponentConfig.model_validate(steps[0]).backend != _BACKENDS[plugin]:
            raise ValueError(f"Home requires answer_judge from {plugin}")
        return plugin

    def identity(self, benchmark):
        """Record host plugin code and effective non-secret scoring configuration."""
        plugin = self.require_benchmark(benchmark)
        root = Path(import_module(_MODULES[plugin]).__file__).parent
        files = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}
        }
        config = self.app_context.app_config
        step = ComponentConfig.model_validate(self.app_context.jobs["answer_judge"].step_configs[0])
        wrapper_name = step.model_dump().get("agent_wrapper", "judge")
        wrapper = config.components.get("agent_wrapper", {}).get(wrapper_name)
        wrapper_config = wrapper.model_dump(mode="json") if wrapper else {}
        model_name = wrapper_config.get("as_llm", "judge")
        model = config.components.get("as_llm", {}).get(model_name)
        model_config = model.model_dump(mode="json", exclude={"credential"}) if model else {}
        return {
            "plugin": plugin,
            "model": model_config.get("model"),
            "fingerprint": fingerprint(
                {
                    "files": files,
                    "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "step": step.model_dump(mode="json"),
                    "wrapper": wrapper_config,
                    "model": model_config,
                },
            ),
        }

    async def evaluate(self, dataset, query, answer):
        """Call the host Job and reject invalid scores as infrastructure failures."""
        self.require_benchmark(dataset)
        job = self.app_context.jobs["answer_judge"]
        if not job.is_started:
            raise RuntimeError("Start answer_judge before evaluating")
        if dataset == "beam":
            arguments = {
                "llm_response": answer,
                "rubric": query.metadata.get("rubric", query.golden_answer),
                "probing_question": query.question,
                "question_type": query.metadata.get("question_type", ""),
            }
        else:
            arguments = {
                "query": query.question,
                "agent_answer": answer,
                "golden_answer": query.golden_answer,
                "question_type": query.metadata.get("question_type", ""),
            }
        response = await job(**arguments)
        result = json.loads(response.model_dump_json())
        try:
            score = self._score_response(dataset, response, arguments.get("rubric", []))
        except (ValueError, TypeError) as error:
            raise JudgeResultError(str(error), result) from error
        return result, score

    @staticmethod
    def _score_response(dataset, response, rubric):
        """Keep benchmark-specific scores and reject failed/invalid Job output."""
        if not response.success:
            raise ValueError(f"answer_judge failed: {response.answer}")
        if dataset == "beam":
            scores = response.metadata.get("llm_judge_responses")
            if not isinstance(scores, list) or len(scores) != len(rubric):
                raise ValueError("BEAM judge returned incomplete rubric results")
            for item in scores:
                if not isinstance(item, dict) or str(item.get("reason", "")).startswith("Failed to parse:"):
                    raise ValueError("BEAM judge failed to parse rubric output")
                if item.get("score") not in {0, 0.5, 1}:
                    raise ValueError("BEAM judge returned an invalid rubric score")
            score = float(response.metadata.get("llm_judge_score", float("nan")))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("BEAM judge returned an invalid aggregate score")
        else:
            if response.answer not in {"yes", "no"}:
                raise ValueError("LME judge returned an invalid yes/no judgement")
            score = 1.0 if response.answer == "yes" else 0.0
        return score
