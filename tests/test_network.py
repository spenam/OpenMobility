from openmobility.processing.network import (
    _union_find_components,
)


def test_connected_graph_has_one_component():
    nodes = {(0, 0), (1, 0), (2, 0), (1, 1)}
    edges = [
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((1, 0), (1, 1)),
    ]

    assert _union_find_components(nodes, edges) == 1


def test_disconnected_graph_counts_components():
    nodes = {
        (0, 0), (1, 0),
        (5, 0), (6, 0),
        (10, 0),
    }
    edges = [
        ((0, 0), (1, 0)),
        ((5, 0), (6, 0)),
    ]

    # Two connected pairs and one isolated node.
    assert _union_find_components(nodes, edges) == 3


def test_empty_graph_has_no_components():
    assert _union_find_components(set(), []) == 0