"""Adapters around the original VirtualHome Unity and evolving-graph runtimes."""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))


def graph_text(graph: dict[str, Any]) -> str:
    names = sorted({node["class_name"] for node in graph["nodes"]})
    return "Visible objects: " + ", ".join(names)


def prune_graph(
    graph: dict[str, Any],
    goal_spec: dict[str, list[Any]],
    vocabulary: dict[str, Any],
) -> dict[str, Any]:
    """Keep every executable action target in the planning graph.

    Unity can reveal an object after walking to a new room. Dropping it at reset
    leaves a valid Unity action with no corresponding Python-graph node.
    """
    return copy.deepcopy(graph)


def goal_text(goals: dict[str, int], graph: dict[str, Any]) -> str:
    id_to_name = {int(node["id"]): node["class_name"] for node in graph["nodes"]}
    descriptions = []
    for predicate, count in sorted(goals.items()):
        parts = predicate.split("_")
        relation = parts[0]
        if relation in {"on", "inside"} and len(parts) >= 3:
            target = id_to_name.get(int(parts[2]), parts[2])
            preposition = "on" if relation == "on" else "inside"
            descriptions.append(f"put {count} {parts[1]} {preposition} {target}")
        else:
            descriptions.append(f"satisfy {predicate} {count} time(s)")
    return "; ".join(descriptions)


def valid_actions(observation: dict[str, Any], vocabulary: dict[str, Any]) -> list[str]:
    """Reproduce the source project's seven action families without torch imports."""
    nodes = {int(node["id"]): node for node in observation["nodes"]}
    visible = list(nodes.values())
    char_id = next(
        (int(node["id"]) for node in visible if node.get("category") == "Characters"),
        1,
    )
    close = {
        int(edge["to_id"]) if int(edge["from_id"]) == char_id else int(edge["from_id"])
        for edge in observation["edges"]
        if edge["relation_type"] == "CLOSE"
        and char_id in (int(edge["from_id"]), int(edge["to_id"]))
    }
    holding = [
        int(edge["to_id"])
        for edge in observation["edges"]
        if int(edge["from_id"]) == char_id and "HOLD" in edge["relation_type"]
    ]
    ignored_walk = {
        "walllamp", "doorjamb", "ceilinglamp", "door", "curtains",
        "candle", "wallpictureframe", "powersocket", "wall", "floor",
        "ceiling", "curtain", "window",
    }
    actions = [
        f"[walk] <{node['class_name']}> ({node['id']})"
        for node in visible
        if int(node["id"]) != char_id and node["class_name"] not in ignored_walk
    ]
    if not holding:
        actions.extend(
            f"[grab] <{nodes[i]['class_name']}> ({i})"
            for i in sorted(close)
            if i in nodes and nodes[i]["class_name"] in vocabulary["objects_grab"]
        )
    for index in sorted(close):
        node = nodes.get(index)
        if node is None:
            continue
        name = node["class_name"]
        states = node.get("states", [])
        if name in vocabulary["objects_inside"]:
            if "CLOSED" in states:
                actions.append(f"[open] <{name}> ({index})")
            if "OPEN" in states:
                actions.append(f"[close] <{name}> ({index})")
        if name in vocabulary["objects_switchonoff"] and "OFF" in states:
            actions.append(f"[switchon] <{name}> ({index})")
        if len(holding) == 1 and holding[0] in nodes:
            object_id = holding[0]
            object_name = nodes[object_id]["class_name"]
            if name in vocabulary["objects_inside"] and "OPEN" in states:
                actions.append(
                    f"[putin] <{object_name}> ({object_id}) <{name}> ({index})"
                )
            if name in vocabulary["objects_surface"] and index != object_id:
                actions.append(
                    f"[putback] <{object_name}> ({object_id}) <{name}> ({index})"
                )
    return list(dict.fromkeys(actions))


def goal_complete(graph: dict[str, Any], goal_spec: dict[str, list[Any]]) -> bool:
    from vh.data_gene.utils.utils_environment import check_progress

    _, unsatisfied = check_progress(graph, goal_spec)
    return all(count <= 0 for count in unsatisfied.values())


class GraphSimulation:
    """A fresh evolving-graph simulation for each MCTS iteration."""

    def __init__(
        self,
        graph: dict[str, Any],
        goal_spec: dict[str, list[Any]],
        vocabulary: dict[str, Any],
    ) -> None:
        from vh.vh_mdp.vh_graph.envs.vh_env import VhGraphEnv

        self.environment = VhGraphEnv()
        self.environment.pomdp = True
        self.environment.reset(copy.deepcopy(graph))
        self.goal_spec = goal_spec
        self.vocabulary = vocabulary
        observation = self.environment.get_observations()
        self.valid_actions = valid_actions(observation, vocabulary)

    def step(self, action: str) -> tuple[str, float, bool, list[str]]:
        next_state, succeeded = self.environment.transition(
            self.environment.vh_state, {0: action}
        )
        self.environment.vh_state = next_state
        graph = next_state.to_dict()
        self.environment.state = graph
        observation = self.environment._mask_state(graph, 0)
        done = bool(succeeded and goal_complete(graph, self.goal_spec))
        reward = 10.0 if done else 0.0
        self.valid_actions = [] if done else valid_actions(observation, self.vocabulary)
        return graph_text(observation), reward, done, self.valid_actions


class UnityTask:
    """Real task execution with the same goal spec and action strings."""

    def __init__(
        self,
        dataset: Path,
        executable: Path,
        vocabulary: dict[str, Any],
        *,
        seed: int = 42,
        port: int = 8184,
    ) -> None:
        from vh.data_gene.envs.unity_environment import UnityEnvironment

        tasks = pickle.loads(dataset.read_bytes())
        self.environment = UnityEnvironment(
            num_agents=1,
            max_episode_length=30,
            env_task_set=tasks,
            observation_types=["partial"],
            use_editor=False,
            executable_args={"file_name": str(executable), "no_graphics": True},
            base_port=port,
            port_id=0,
            seed=seed,
        )
        self.vocabulary = vocabulary
        self.tasks = tasks

    def reset(self, task_index: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        observations = self.environment.reset(task_id=task_index)
        return observations[0], self.environment.get_graph()

    @property
    def task_goal(self) -> dict[str, int]:
        return self.environment.task_goal[0]

    @property
    def goal_spec(self) -> dict[str, list[Any]]:
        return self.environment.goal_spec[0]

    def step(self, action: str) -> tuple[dict[str, Any], bool, bool, dict[str, Any]]:
        observations, _, done, info = self.environment.step({0: action})
        return observations[0], bool(info.get("finished")), bool(done), info

    def close(self) -> None:
        self.environment.comm.close()


def load_vocabulary() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "data/object_info.json").read_text())
