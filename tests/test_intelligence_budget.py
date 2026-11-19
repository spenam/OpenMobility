import pytest

from openmobility.intelligence.graph import (
    ToolBudget,
    ToolBudgetExceeded,
)


def test_tool_budget_records_successful_call():
    budget = ToolBudget(
        agent="traffic",
        used=0,
        maximum=2,
    )

    result = budget.call(
        "example",
        lambda value: value + 1,
        4,
        arguments={"value": 4},
    )

    assert result == 5
    assert budget.used == 1
    assert len(budget.history) == 1
    assert budget.history[0]["success"] is True


def test_tool_budget_records_failed_call():
    budget = ToolBudget(
        agent="traffic",
        used=0,
        maximum=2,
    )

    def fail():
        raise ValueError("test failure")

    with pytest.raises(ValueError):
        budget.call("failing_tool", fail)

    assert budget.used == 1
    assert budget.history[0]["success"] is False
    assert budget.history[0]["error"] == "test failure"


def test_tool_budget_rejects_calls_after_limit():
    budget = ToolBudget(
        agent="traffic",
        used=1,
        maximum=1,
    )

    with pytest.raises(ToolBudgetExceeded):
        budget.call("blocked", lambda: "should not run")

    assert budget.used == 1
    assert budget.history == []