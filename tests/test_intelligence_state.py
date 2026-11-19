import pytest

from openmobility.intelligence.state import (
    create_initial_state,
)


def test_initial_state_has_independent_finding_lists():
    first = create_initial_state("Investigate traffic")
    second = create_initial_state("Investigate road connectivity")

    first["traffic_findings"].append({"summary": "test"})

    assert second["traffic_findings"] == []
    assert first["network_findings"] == []


def test_initial_state_rejects_empty_question():
    with pytest.raises(ValueError):
        create_initial_state("   ")


def test_initial_state_rejects_invalid_limits():
    with pytest.raises(ValueError):
        create_initial_state("Investigate traffic", max_iterations=0)

    with pytest.raises(ValueError):
        create_initial_state("Investigate traffic", max_tool_calls=0)