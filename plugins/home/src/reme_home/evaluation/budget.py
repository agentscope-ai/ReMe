"""Formal optimizer budgets derived from Home's durable sandbox-start evidence."""

from .history import read_state


def validate_max_validations(value):
    """Accept only a positive host-supplied integer, matching dev_meta."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_validations must be a positive integer")
    return value


def validation_budget_status(home, max_validations):
    """Report started formal runs, independently of completion, score or sampler success."""
    validate_max_validations(max_validations)
    used = sum(not key.startswith("init/") for key in read_state(home)["started_validations"])
    remaining = max(0, max_validations - used)
    gated = max_validations // 2
    if not remaining:
        phase = "exhausted"
        guidance = (
            "The formal validation budget is exhausted; select among existing evidence without another validation."
        )
    elif used < gated:
        phase = "structural_exploration"
        guidance = (
            "Prefer a substantial code-level mechanism or coherent plugin architecture change. "
            "Do not spend an early validation on prompt-only or single-parameter tuning."
        )
    else:
        phase = "evidence_driven_refinement"
        guidance = (
            "Prefer evidence-driven parameter tuning, fine-grained prompt refinement, or a small scoped code change. "
            "Avoid a new broad architecture unless existing validation evidence clearly requires it."
        )
    return {
        "used_validations": used,
        "max_validations": max_validations,
        "remaining_validations": remaining,
        "gated_validation_count": gated,
        "optimization_phase": phase,
        "next_validation_guidance": guidance,
    }
