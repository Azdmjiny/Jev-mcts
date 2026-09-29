import random

from jev_mcts.belief import BeliefState


def test_hidden_true_location_is_not_leaked():
    graph = {
        "nodes": [
            {"id": 1, "class_name": "apple", "properties": ["GRABBABLE"]},
            {"id": 2, "class_name": "fridge", "states": ["CLOSED"]},
            {"id": 3, "class_name": "table", "states": []},
        ],
        "edges": [{"from_id": 1, "to_id": 2, "relation_type": "INSIDE"}],
    }
    belief = BeliefState(graph, {"apple": [["ON", "table", 1.0]]})
    sample = belief.sample_graph(random.Random(1))
    assert {"from_id": 1, "to_id": 2, "relation_type": "INSIDE"} not in sample["edges"]
    assert {"from_id": 1, "to_id": 3, "relation_type": "ON"} in sample["edges"]


def test_observed_position_overrides_prior():
    graph = {
        "nodes": [
            {"id": 1, "class_name": "apple", "properties": ["GRABBABLE"]},
            {"id": 2, "class_name": "fridge", "states": []},
            {"id": 3, "class_name": "table", "states": []},
        ],
        "edges": [],
    }
    belief = BeliefState(graph, {"apple": [["ON", "table", 1.0]]})
    belief.observe({
        "nodes": graph["nodes"],
        "edges": [{"from_id": 1, "to_id": 2, "relation_type": "INSIDE"}],
    })
    sample = belief.sample_graph(random.Random(1))
    assert {"from_id": 1, "to_id": 2, "relation_type": "INSIDE"} in sample["edges"]


def test_open_visible_container_excludes_hidden_object():
    graph = {
        "nodes": [
            {"id": 1, "class_name": "apple", "properties": ["GRABBABLE"]},
            {"id": 2, "class_name": "fridge", "states": ["CLOSED"]},
            {"id": 3, "class_name": "table", "states": []},
        ],
        "edges": [{"from_id": 1, "to_id": 2, "relation_type": "INSIDE"}],
    }
    belief = BeliefState(graph, {"apple": [["INSIDE", "fridge", 0.9], ["ON", "table", 0.1]]})
    belief.observe({
        "nodes": [{"id": 2, "class_name": "fridge", "states": ["OPEN"]}],
        "edges": [],
    })
    sample = belief.sample_graph(random.Random(1))
    assert {"from_id": 1, "to_id": 3, "relation_type": "ON"} in sample["edges"]
    assert "OPEN" in next(node["states"] for node in sample["nodes"] if node["id"] == 2)
