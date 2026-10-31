"""Exact token budgets, internal field slack and lossless prefix boundaries."""

# pylint: disable=missing-function-docstring

import pytest

from reme_procedure_memory.token_budget import TokenBudget


def test_exact_counter_and_boundaries():
    budget = TokenBudget(counter=lambda text: len(text.encode()), content=6)
    budget.check("content", "你好")
    with pytest.raises(ValueError, match="maximum is 6"):
        budget.check("content", "你好!")
    assert budget.prefix("你好!", 6) == 2
    assert budget.prefix("skip你好!", 6, start=4) == 2
    assert budget.prefix("skip", 6, start=4) == 0
    with pytest.raises(ValueError, match="cannot hold one character"):
        budget.prefix("你", 1)


def test_no_estimate_fallback():
    budget = TokenBudget()
    budget.tools._tokenizer_unavailable = True  # pylint: disable=protected-access
    with pytest.raises(RuntimeError, match="Exact tokenizer unavailable"):
        budget.count("abc")


@pytest.mark.parametrize("target,accepted,rejected", [(8192, 9011, 9012), (10, 11, 12), (1, 1, 2)])
@pytest.mark.parametrize("field", ["summary", "description", "when-to-use", "content"])
def test_field_slack_preserves_advertised_target(field, target, accepted, rejected):
    budget = TokenBudget(counter=len, **{field: target})
    budget.check(field, "s" * accepted)
    assert budget.limits[field] == target
    with pytest.raises(ValueError, match=f"maximum is {target}\\."):
        budget.check(field, "s" * rejected)


@pytest.mark.parametrize("field", ["segment", "context", "reserve"])
def test_execution_budgets_remain_strict(field):
    budget = TokenBudget(counter=len, **{field: 10})
    with pytest.raises(ValueError, match="maximum is 10"):
        budget.check(field, "s" * 11)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_budget(limit):
    with pytest.raises(ValueError):
        TokenBudget(segment=limit)
