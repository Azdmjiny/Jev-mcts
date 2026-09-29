"""Generate and run reproducible local VirtualHome episodes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .belief import BeliefState
from .jev import BudgetExceeded, JevClient, Usage
from .policy import JevPolicy
from .search import MCTS
from .virtualhome import (
    PROJECT_ROOT,
    RUNTIME_ROOT,
    GraphSimulation,
    UnityTask,
    goal_text,
    graph_text,
    load_vocabulary,
    prune_graph,
    valid_actions,
)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def generate_one(
    executable: Path,
    output: Path,
    *,
    task: str = "put_fridge",
    mode: str = "simple",
    unseen_apartment: bool = False,
    unseen_item: bool = False,
    seed: int = 42,
    port: int = 8191,
) -> Path:
    """Invoke the original task generator, then keep its output by case name."""
    generator = RUNTIME_ROOT / "vh/data_gene/gen_data/vh_init.py"
    if not generator.is_file():
        raise FileNotFoundError(f"missing VirtualHome generator: {generator}")
    command = [
        sys.executable,
        str(generator),
        "--task", task,
        "--mode", mode,
        "--usage", "test",
        "--num-per-apartment", "1",
        "--seed", str(seed),
        "--port", str(port),
        "--exec_file", str(executable),
    ]
    if unseen_apartment:
        command.append("--unseen-apartment")
    if unseen_item:
        command.append("--unseen-item")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(RUNTIME_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    (RUNTIME_ROOT / "vh/dataset").mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=RUNTIME_ROOT, env=environment, check=True, timeout=900)
    if unseen_item:
        suffix = "unseen_composition_unseen_item" if task == "unseen_comp" else "unseen_item"
    elif unseen_apartment:
        suffix = "unseen_apartment"
    elif task == "unseen_comp":
        suffix = "unseen_composition"
    else:
        suffix = "seen"
    generated = RUNTIME_ROOT / f"vh/dataset/env_task_set_1_{mode}_{suffix}.pik"
    if not generated.is_file():
        raise RuntimeError(f"generator did not produce {generated}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(generated.read_bytes())
    return output


def run_one(
    dataset: Path,
    executable: Path,
    priors_directory: Path,
    output: Path,
    *,
    seed: int = 42,
    simulations: int = 20,
    max_usd: float = 5.0,
    max_seconds: float = 7200.0,
    port: int = 8184,
    usage: Usage | None = None,
) -> dict[str, Any]:
    """Run one actual Unity task with Jev priors and source-style MCTS."""
    object_priors = json.loads((priors_directory / "obj_commonsense.json").read_text())
    furniture_priors = json.loads((priors_directory / "fur_commonsense.json").read_text())
    vocabulary = load_vocabulary()
    deadline = time.monotonic() + max_seconds
    client = JevClient(max_usd=max_usd, deadline=deadline, usage=usage)
    prior_result = json.loads(output.read_text()) if output.exists() else None
    prior_episode_usage = Usage.from_dict(prior_result.get("jev_usage", {})) if prior_result else Usage()
    usage_before = Usage.from_dict(client.usage.as_dict())

    def episode_usage() -> dict[str, Any]:
        previous = Usage(
            calls=client.usage.calls - usage_before.calls,
            input_tokens=client.usage.input_tokens - usage_before.input_tokens,
            output_tokens=client.usage.output_tokens - usage_before.output_tokens,
            models=set(client.usage.models),
        )
        previous.add(prior_episode_usage)
        return previous.as_dict()

    policy = JevPolicy(client)
    search = MCTS(policy, simulations=simulations, seed=seed)
    task = None
    result: dict[str, Any] = prior_result or {
        "dataset": str(dataset),
        "seed": seed,
        "simulations_per_action": simulations,
        "max_real_actions": 30,
        "status": "running",
        "success": False,
        "steps": [],
        "started_at": time.time(),
    }
    result["status"] = "running"
    result.pop("error", None)
    try:
        task = UnityTask(dataset, executable, vocabulary, seed=seed, port=port)
        observation, full_graph = task.reset(0)
        goal_spec = task.goal_spec
        result["goal"] = goal_text(task.task_goal, full_graph)
        result["goal_spec"] = goal_spec
        planning_graph = prune_graph(full_graph, goal_spec, vocabulary)
        result["planning_graph_size"] = {
            "nodes": len(planning_graph["nodes"]),
            "edges": len(planning_graph["edges"]),
        }
        belief = BeliefState(planning_graph, object_priors, furniture_priors)
        belief.observe(observation)
        history: list[str] = []
        for old_step in result["steps"]:
            action = old_step["action"]
            observation, finished, _, info = task.step(action)
            if bool(info.get("failed_exec", False)) != old_step["unity_failed_execution"]:
                raise RuntimeError("Unity replay differs from saved action trajectory")
            history.append(action)
            belief.observe(observation)
            if finished and old_step is not result["steps"][-1]:
                raise RuntimeError("Unity replay finished before saved trajectory ended")
        for step_number in range(len(history), 30):
            client.check_budget()
            actions = valid_actions(observation, vocabulary)
            if not actions:
                result["status"] = "no_valid_actions"
                break
            action, root = search.search(
                result["goal"],
                graph_text(observation),
                history,
                actions,
                belief.sample_graph,
                lambda graph: GraphSimulation(graph, goal_spec, vocabulary),
            )
            observation, finished, done, info = task.step(action)
            history.append(action)
            belief.observe(observation)
            result["steps"].append({
                "number": step_number + 1,
                "action": action,
                "unity_failed_execution": bool(info.get("failed_exec", False)),
                "unity_finished": finished,
                "root_values": {
                    item.action: {"visits": item.visits, "q": item.q}
                    for item in root.actions
                },
            })
            result["jev_usage"] = episode_usage()
            save_json(output, result)
            if finished:
                result["success"] = True
                result["status"] = "success"
                break
            if done:
                result["status"] = "unity_episode_done"
                break
        else:
            result["status"] = "step_limit"
    except BudgetExceeded as exc:
        result["status"] = "budget_exceeded"
        result["error"] = str(exc)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = time.time()
        result["jev_usage"] = episode_usage()
        save_json(output, result)
        if task is not None:
            task.close()
    return result
