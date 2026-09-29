"""Jev policy probabilities over executable VirtualHome actions."""

from __future__ import annotations

from typing import Protocol


class ChoiceClient(Protocol):
    def ask_choice(
        self, state: object, instructions: str, options: dict[str, str | None]
    ) -> dict[str, float]: ...


class JevPolicy:
    def __init__(self, client: ChoiceClient, uniform_fraction: float = 0.5) -> None:
        if not 0 <= uniform_fraction <= 1:
            raise ValueError("uniform_fraction must be between 0 and 1")
        self.client = client
        self.uniform_fraction = uniform_fraction

    def prior(
        self,
        goal: str,
        observation: str,
        history: list[str],
        valid_actions: list[str],
    ) -> list[float]:
        if not valid_actions:
            raise ValueError("no valid actions")
        if len(valid_actions) == 1:
            return [1.0]
        state = {
            "goal": goal,
            "observation": observation,
            "previous_actions": history[-12:],
        }
        instructions = (
            "Which executable action best advances the goal now? Consider "
            "the observed scene, prerequisites, and already completed actions."
        )
        # A Choice has a 255-option ceiling. Chunk only when necessary, and
        # combine P(chunk) * P(action | chunk) without discarding valid actions.
        chunks = [
            valid_actions[index:index + 255]
            for index in range(0, len(valid_actions), 255)
        ]
        if len(chunks) == 1:
            raw = self._choice(state, instructions, valid_actions)
        else:
            group_options = {
                f"group_{index}": ", ".join(chunk)
                for index, chunk in enumerate(chunks)
            }
            group_probs = self.client.ask_choice(
                state,
                "Which group contains the best action to advance the goal now?",
                group_options,
            )
            raw = []
            for index, chunk in enumerate(chunks):
                if len(chunk) == 1:
                    within = [1.0]
                else:
                    within = self._choice(state, instructions, chunk)
                raw.extend(group_probs[f"group_{index}"] * p for p in within)
        size = len(valid_actions)
        probabilities = [
            self.uniform_fraction / size + (1 - self.uniform_fraction) * p
            for p in raw
        ]
        total = sum(probabilities)
        return [p / total for p in probabilities]

    def _choice(
        self, state: object, instructions: str, actions: list[str]
    ) -> list[float]:
        options = {f"a{index}": action for index, action in enumerate(actions)}
        probabilities = self.client.ask_choice(state, instructions, options)
        return [probabilities[f"a{index}"] for index in range(len(actions))]
