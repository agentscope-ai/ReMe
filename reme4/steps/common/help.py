"""Return a one-line summary of every registered job for LLM consumption."""

from ..base_step import BaseStep
from ...components import R


def _format_params(parameters: dict) -> str:
    props = (parameters or {}).get("properties") or {}
    if not props:
        return "no args"
    required = set((parameters or {}).get("required") or [])
    parts = []
    for pname, pschema in props.items():
        ptype = pschema.get("type", "any")
        if pname in required:
            parts.append(f"{pname}:{ptype}*")
        elif "default" in pschema:
            parts.append(f"{pname}:{ptype}={pschema['default']}")
        else:
            parts.append(f"{pname}:{ptype}")
    return ", ".join(parts)


@R.register("help_step")
class HelpStep(BaseStep):
    """List all registered jobs (excluding self) as compact one-liners for an LLM."""

    async def execute(self):
        assert self.context is not None

        lines = []
        if self.app_context is not None:
            for name, job in self.app_context.jobs.items():
                if name == "help":
                    continue
                lines.append(f"🛠️ `{name}` — {job.description} 📥 {_format_params(job.parameters)}")

        self.logger.info(f"[{self.name}] returning {len(lines)} jobs")

        self.context.response.answer = "\n".join(lines)
        self.context.response.metadata["job_count"] = len(lines)
        return self.context.response
