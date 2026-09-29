"""MCTS using the public source variant's search semantics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol


class SimEnvironment(Protocol):
    def step(self, action: str) -> tuple[str, float, bool, list[str]]: ...


class Policy(Protocol):
    def prior(
        self, goal: str, observation: str, history: list[str], valid_actions: list[str]
    ) -> list[float]: ...


@dataclass
class ActionNode:
    action: str
    visits: int = 0
    q: float = 0.0
    returns: list[float] = field(default_factory=list)
    child: StateNode | None = None


@dataclass
class StateNode:
    history: tuple[str, ...]
    observation: str
    actions: list[ActionNode]
    priors: list[float]
    done: bool = False
    visits: int = 0


def softmax_weighted_return(returns: list[float], temperature: float = 10.0) -> float:
    """Match the original source's optimistic softmax-weighted Q update."""
    if not returns:
        return 0.0
    maximum = max(value / temperature for value in returns)
    weights = [math.exp(value / temperature - maximum) for value in returns]
    return sum(value * weight for value, weight in zip(returns, weights)) / sum(weights)


class MCTS:
    def __init__(
        self,
        policy: Policy,
        *,
        simulations: int = 20,
        max_depth: int = 20,
        exploration: float = 24.0,
        discount: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.policy = policy
        self.simulations = simulations
        self.max_depth = max_depth
        self.exploration = exploration
        self.discount = discount
        self.rng = random.Random(seed)

    def search(
        self,
        goal: str,
        observation: str,
        history: list[str],
        valid_actions: list[str],
        sample_graph: Callable[[random.Random], dict],
        env_factory: Callable[[dict], SimEnvironment],
    ) -> tuple[str, StateNode]:
        if not valid_actions:
            raise ValueError("cannot search without valid actions")
        root = self._make_state(goal, observation, history, valid_actions, False)
        # One belief sample is shared by all simulations for this real action,
        # following the released source rather than the paper pseudocode.
        graph = sample_graph(self.rng)
        for _ in range(self.simulations):
            environment = env_factory(graph)
            self._simulate(goal, root, environment, 0)
        chosen = max(
            range(len(root.actions)),
            key=lambda index: (root.actions[index].q, root.priors[index], -index),
        )
        return root.actions[chosen].action, root

    def _make_state(
        self,
        goal: str,
        observation: str,
        history: list[str],
        valid_actions: list[str],
        done: bool,
    ) -> StateNode:
        if not valid_actions:
            priors = []
        elif done:
            priors = [1 / len(valid_actions)] * len(valid_actions)
        else:
            priors = self.policy.prior(goal, observation, history, valid_actions)
        if len(priors) != len(valid_actions) or (
            valid_actions and abs(sum(priors) - 1) > 1e-6
        ):
            raise ValueError("policy probabilities do not match valid actions")
        return StateNode(
            history=tuple(history),
            observation=observation,
            actions=[ActionNode(action) for action in valid_actions],
            priors=priors,
            done=done,
        )

    def _select(self, node: StateNode) -> int:
        return max(
            range(len(node.actions)),
            key=lambda index: (
                node.actions[index].q
                + self.exploration
                * node.priors[index]
                * math.sqrt(node.visits)
                / (node.actions[index].visits + 1),
                node.priors[index],
                -index,
            ),
        )

    def _simulate(
        self, goal: str, node: StateNode, env: SimEnvironment, depth: int
    ) -> float:
        if node.done or depth >= self.max_depth or not node.actions:
            return 0.0
        index = self._select(node)
        edge = node.actions[index]
        observation, reward, done, actions = env.step(edge.action)
        next_history = [*node.history, edge.action]
        if edge.child is None or edge.child.history != tuple(next_history):
            edge.child = self._make_state(goal, observation, next_history, actions, done)
            future = 0.0 if done else self._rollout(env, depth + 1)
        else:
            future = self._simulate(goal, edge.child, env, depth + 1)
        total_return = reward + self.discount * future
        node.visits += 1
        edge.visits += 1
        edge.returns.append(total_return)
        edge.q = softmax_weighted_return(edge.returns)
        return total_return

    def _rollout(self, env: SimEnvironment, depth: int) -> float:
        total = 0.0
        weight = 1.0
        actions = getattr(env, "valid_actions", [])
        while actions and depth < self.max_depth:
            action = self.rng.choice(actions)
            _, reward, done, actions = env.step(action)
            total += weight * reward
            if done:
                break
            weight *= self.discount
            depth += 1
        return total
