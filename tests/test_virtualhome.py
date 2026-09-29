from jev_mcts.virtualhome import goal_text, valid_actions


def test_valid_actions_include_walk_open_and_grab():
    observation = {
        "nodes": [
            {"id": 1, "class_name": "character", "category": "Characters"},
            {"id": 2, "class_name": "fridge", "states": ["CLOSED"]},
            {"id": 3, "class_name": "apple", "states": []},
        ],
        "edges": [
            {"from_id": 1, "to_id": 2, "relation_type": "CLOSE"},
            {"from_id": 1, "to_id": 3, "relation_type": "CLOSE"},
        ],
    }
    vocabulary = {
        "objects_grab": ["apple"],
        "objects_inside": ["fridge"],
        "objects_surface": [],
        "objects_switchonoff": [],
    }
    actions = valid_actions(observation, vocabulary)
    assert "[walk] <fridge> (2)" in actions
    assert "[open] <fridge> (2)" in actions
    assert "[grab] <apple> (3)" in actions


def test_goal_text_uses_object_names():
    graph = {"nodes": [{"id": 42, "class_name": "fridge"}], "edges": []}
    assert goal_text({"inside_apple_42": 1}, graph) == "put 1 apple inside fridge"
