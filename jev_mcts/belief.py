"""Object-location belief for partially observed VirtualHome graphs."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any


POSITION_RELATIONS = {"INSIDE", "ON", "HOLDS_LH", "HOLDS_RH"}


@dataclass(frozen=True)
class Position:
    relation: str
    target_id: int


class BeliefState:
    """Maintain location probabilities without reading hidden location edges.

    The full graph supplies object identities and legal candidate surfaces. Hidden
    relations are discarded before planning. Seen relations are authoritative.
    """

    def __init__(
        self,
        initial_graph: dict[str, Any],
        object_priors: dict[str, list[list[Any]]],
        furniture_priors: dict[str, list[list[Any]]] | None = None,
    ) -> None:
        self.base_graph = copy.deepcopy(initial_graph)
        self.nodes = {int(node["id"]): node for node in self.base_graph["nodes"]}
        self.visible_edges: list[dict[str, Any]] = []
        self.excluded: dict[int, set[Position]] = {}
        self.fixed: dict[int, Position] = {}
        self.distributions: dict[int, dict[Position, float]] = {}
        self.tracked_relations: dict[int, set[str]] = {}
        nodes_by_class: dict[str, list[int]] = {}
        for node in self.nodes.values():
            nodes_by_class.setdefault(node["class_name"], []).append(int(node["id"]))
        for node in self.nodes.values():
            grabbable = "GRABBABLE" in node.get("properties", [])
            if not grabbable and node["class_name"] not in (furniture_priors or {}):
                continue
            object_id = int(node["id"])
            if grabbable:
                class_prior = object_priors.get(node["class_name"], [])
                self.tracked_relations[object_id] = POSITION_RELATIONS
            else:
                class_prior = (furniture_priors or {}).get(node["class_name"], [])
                self.tracked_relations[object_id] = {"INSIDE"}
            candidates: dict[Position, float] = {}
            for relation, target_class, weight in class_prior:
                targets = [i for i in nodes_by_class.get(target_class, []) if i != object_id]
                for target_id in targets:
                    position = Position(relation, target_id)
                    candidates[position] = candidates.get(position, 0.0) + float(weight) / len(targets)
            if not candidates:
                # The object class may be absent from the controlled vocabulary.
                # Fail explicitly instead of retaining its hidden true location.
                raise ValueError(f"no Jev location candidates for {node['class_name']}")
            self.distributions[object_id] = self._normalize(candidates)
            self.excluded[object_id] = set()

    @staticmethod
    def _normalize(weights: dict[Position, float]) -> dict[Position, float]:
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0:
            raise ValueError("location candidates have zero probability")
        return {key: max(0.0, value) / total for key, value in weights.items()}

    def observe(self, observation: dict[str, Any]) -> None:
        visible_ids = {int(node["id"]) for node in observation["nodes"]}
        observed_nodes = {int(node["id"]): node for node in observation["nodes"]}
        for index, node in enumerate(self.base_graph["nodes"]):
            if int(node["id"]) in observed_nodes:
                updated = copy.deepcopy(observed_nodes[int(node["id"])])
                self.base_graph["nodes"][index] = updated
                self.nodes[int(node["id"])] = updated
        self.visible_edges = [
            copy.deepcopy(edge)
            for edge in observation["edges"]
            if int(edge["from_id"]) in self.nodes and int(edge["to_id"]) in self.nodes
        ]
        observed_positions = {
            int(edge["from_id"]): Position(edge["relation_type"], int(edge["to_id"]))
            for edge in self.visible_edges
            if edge["relation_type"] in POSITION_RELATIONS
            and int(edge["from_id"]) in self.distributions
            and edge["relation_type"] in self.tracked_relations[int(edge["from_id"])]
        }
        for object_id in self.distributions:
            if object_id in observed_positions:
                self.fixed[object_id] = observed_positions[object_id]
                continue
            if object_id in visible_ids:
                self.fixed.pop(object_id, None)
            # An opened, visible container is negative evidence for a hidden item.
            for target_id in visible_ids:
                target = self.nodes.get(target_id)
                if target and "OPEN" in target.get("states", []):
                    self.excluded[object_id].add(Position("INSIDE", target_id))

    def sample_graph(self, rng: random.Random) -> dict[str, Any]:
        graph = copy.deepcopy(self.base_graph)
        tracked = set(self.distributions)
        graph["edges"] = [
            edge for edge in graph["edges"]
            if not (
                int(edge["from_id"]) in tracked
                and edge["relation_type"] in self.tracked_relations[int(edge["from_id"])]
            )
        ]
        for object_id, weights in self.distributions.items():
            if object_id in self.fixed:
                position = self.fixed[object_id]
            else:
                available = {
                    position: weight
                    for position, weight in weights.items()
                    if position not in self.excluded[object_id]
                }
                if not available:
                    raise ValueError(f"all candidate positions excluded for object {object_id}")
                available = self._normalize(available)
                position = rng.choices(
                    list(available), weights=list(available.values()), k=1
                )[0]
            graph["edges"].append(
                {
                    "from_id": object_id,
                    "to_id": position.target_id,
                    "relation_type": position.relation,
                }
            )
        # Preserve all observed facts; they supersede hypotheses above.
        visible_keys = {
            (int(edge["from_id"]), edge["relation_type"])
            for edge in self.visible_edges
        }
        graph["edges"] = [
            edge for edge in graph["edges"]
            if (int(edge["from_id"]), edge["relation_type"]) not in visible_keys
        ]
        graph["edges"].extend(copy.deepcopy(self.visible_edges))
        return graph
