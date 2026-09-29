"""Eight-case, bounded VirtualHome pilot with durable episode checkpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experiment import generate_one, run_one, save_json
from .jev import Usage


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    mode: str
    unseen_apartment: bool = False
    unseen_item: bool = False


CASES = tuple(
    Case(
        f"{apartment}_{category}",
        task,
        mode,
        unseen_apartment=(apartment == "unseen"),
        unseen_item=novel,
    )
    for apartment in ("seen", "unseen")
    for category, task, mode, novel in (
        ("simple", "put_fridge", "simple", False),
        ("comp", "setup_table_prepare_food", "full", False),
        ("novel_simple", "put_fridge", "simple", True),
        ("novel_comp2", "unseen_comp", "full", True),
    )
)


def summarize(directory: Path, started_at: float, budget_usd: float, max_seconds: float) -> dict[str, Any]:
    cases = []
    total = Usage()
    for case in CASES:
        path = directory / "episodes" / f"{case.name}.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        usage = Usage.from_dict(result.get("jev_usage", {}))
        total.add(usage)
        cases.append({
            "case": case.name,
            "task": case.task,
            "status": result.get("status", "pending"),
            "success": result.get("success", False),
            "actions": len(result.get("steps", [])),
            "goal": result.get("goal"),
            "jev_usage": usage.as_dict(),
            "episode": str(path),
        })
    return {
        "started_at": started_at,
        "updated_at": time.time(),
        "limit_usd": budget_usd,
        "limit_seconds": max_seconds,
        "elapsed_seconds": time.time() - started_at,
        "jev_usage": total.as_dict(),
        "completed": sum(item["status"] in {"success", "step_limit", "no_valid_actions", "unity_episode_done"} for item in cases),
        "successes": sum(item["success"] for item in cases),
        "cases": cases,
    }


def run_pilot(
    executable: Path,
    priors: Path,
    directory: Path,
    *,
    seed: int = 42,
    simulations: int = 20,
    max_usd: float = 5.0,
    max_seconds: float = 7200.0,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        started_at = json.loads(summary_path.read_text())["started_at"]
    else:
        started_at = time.time()
    terminal = {"success", "step_limit", "no_valid_actions", "unity_episode_done"}
    for case in CASES:
        summary = summarize(directory, started_at, max_usd, max_seconds)
        save_json(summary_path, summary)
        if summary["jev_usage"]["estimated_usd"] >= max_usd or summary["elapsed_seconds"] >= max_seconds:
            summary["status"] = "budget_exceeded"
            save_json(summary_path, summary)
            return summary
        output = directory / "episodes" / f"{case.name}.json"
        previous = json.loads(output.read_text()) if output.exists() else {}
        if previous.get("status") in terminal:
            continue
        dataset = directory / "datasets" / f"{case.name}.pik"
        if not dataset.exists():
            generate_one(
                executable, dataset, task=case.task, mode=case.mode,
                unseen_apartment=case.unseen_apartment, unseen_item=case.unseen_item,
                seed=seed,
            )
        summary = summarize(directory, started_at, max_usd, max_seconds)
        usage = Usage.from_dict(summary["jev_usage"])
        remaining_seconds = max_seconds - (time.time() - started_at)
        if remaining_seconds <= 0:
            break
        result = run_one(
            dataset, executable, priors, output, seed=seed,
            simulations=simulations, max_usd=max_usd,
            max_seconds=remaining_seconds, usage=usage,
        )
        summary = summarize(directory, started_at, max_usd, max_seconds)
        summary["last_case"] = case.name
        save_json(summary_path, summary)
        if result["status"] in {"budget_exceeded", "error"}:
            break
    summary = summarize(directory, started_at, max_usd, max_seconds)
    if summary["completed"] == len(CASES):
        summary["status"] = "complete"
    elif summary["jev_usage"]["estimated_usd"] >= max_usd or summary["elapsed_seconds"] >= max_seconds:
        summary["status"] = "budget_exceeded"
    else:
        summary["status"] = "interrupted"
    save_json(summary_path, summary)
    return summary
