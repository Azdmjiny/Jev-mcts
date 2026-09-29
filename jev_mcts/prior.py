"""Turn Jev location choices into VirtualHome commonsense prior files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class ChoiceClient(Protocol):
    def ask_choice(
        self, state: object, instructions: str, options: dict[str, str | None]
    ) -> dict[str, float]: ...


ALIASES = {
    "bathroom_cabinet": "bathroomcabinet",
    "bathroom_counter": "bathroomcounter",
}


def location_options(containers: list[str], surfaces: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for relation, names in (("INSIDE", containers), ("ON", surfaces)):
        for name in names:
            name = ALIASES.get(name, name)
            options[f"{relation}|{name}"] = f"{relation} {name}"
    return options


def object_prior(
    client: ChoiceClient,
    object_name: str,
    containers: list[str],
    surfaces: list[str],
) -> list[list[object]]:
    options = location_options(containers, surfaces)
    probabilities = client.ask_choice(
        {"object": object_name, "environment": "ordinary household"},
        "Where is this object most likely to be found before observing it? "
        "Choose among the given spatial relations and locations.",
        options,
    )
    return [
        [relation, location, probability]
        for key, probability in probabilities.items()
        for relation, location in [key.split("|", 1)]
        if probability > 0
    ]


def room_prior(client: ChoiceClient, furniture_name: str) -> list[list[object]]:
    options = {
        "livingroom": "living room",
        "kitchen": "kitchen",
        "bedroom": "bedroom",
        "bathroom": "bathroom",
    }
    probabilities = client.ask_choice(
        {"furniture": furniture_name, "environment": "ordinary household"},
        "Which room is this furniture most likely to be in?",
        options,
    )
    return [["INSIDE", room, p] for room, p in probabilities.items() if p > 0]


def generate_priors(
    client: ChoiceClient, object_info: Path, output_dir: Path
) -> tuple[Path, Path]:
    info = json.loads(object_info.read_text())
    object_priors = {
        name: object_prior(client, name, info["objects_inside"], info["objects_surface"])
        for name in sorted(set(info["objects_grab"]))
    }
    furniture_names = sorted(
        set(info["objects_inside"] + info["objects_surface"])
        - set(info["objects_grab"])
    )
    furniture_priors = {name: room_prior(client, name) for name in furniture_names}
    output_dir.mkdir(parents=True, exist_ok=True)
    object_path = output_dir / "obj_commonsense.json"
    furniture_path = output_dir / "fur_commonsense.json"
    object_path.write_text(json.dumps(object_priors, indent=2))
    furniture_path.write_text(json.dumps(furniture_priors, indent=2))
    return object_path, furniture_path
