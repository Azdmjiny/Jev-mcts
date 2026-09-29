"""Export a compact, commit-safe record from ignored full pilot traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/pilot"))
    parser.add_argument("--output", type=Path, default=Path("experiments/pilot-2026-09-29.json"))
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())
    if summary["status"] != "complete" or summary["completed"] != 8:
        raise RuntimeError("pilot is not complete")
    prior_usage_path = args.input.parent / "priors" / "usage.json"
    prior_usage = json.loads(prior_usage_path.read_text()) if prior_usage_path.exists() else None
    cases = []
    for item in summary["cases"]:
        episode = json.loads((args.input / "episodes" / f"{item['case']}.json").read_text())
        cases.append({
            "case": item["case"],
            "task": item["task"],
            "goal": episode["goal"],
            "status": episode["status"],
            "success": episode["success"],
            "actions": [step["action"] for step in episode["steps"]],
            "unity_failed_executions": sum(step["unity_failed_execution"] for step in episode["steps"]),
            "jev_usage": episode["jev_usage"],
        })
    exported = {
        "experiment": "Jev-mcts eight-case minimal pilot",
        "seed": 42,
        "simulations_per_real_action": 20,
        "max_real_actions": 30,
        "budget_usd": summary["limit_usd"],
        "time_limit_seconds": summary["limit_seconds"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "successes": summary["successes"],
        "cases_completed": summary["completed"],
        "jev_pilot_usage": summary["jev_usage"],
        "jev_prior_usage": prior_usage,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
