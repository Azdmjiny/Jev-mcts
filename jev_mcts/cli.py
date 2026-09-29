"""Project diagnostics and command entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


def load_local_env(project_root: Path) -> None:
    """Read a small, untracked .env file without overriding shell variables."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "TYPESAFE_API_KEY":
            os.environ.setdefault(key, value.strip().strip("'\""))


def doctor() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_local_env(project_root)
    executable = os.environ.get("VIRTUALHOME_EXECUTABLE", "")
    print(f"host: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Jev API key: {'configured' if os.getenv('TYPESAFE_API_KEY') else 'missing'}")
    if not executable:
        print("Unity executable: missing (set VIRTUALHOME_EXECUTABLE)")
        return 1
    path = Path(executable).expanduser()
    if not path.is_file():
        print(f"Unity executable: missing at {path}")
        return 1
    details = subprocess.run(["file", str(path)], capture_output=True, text=True, check=False)
    print(f"Unity executable: {details.stdout.strip() or path}")
    if platform.system() == "Darwin" and "ELF" in details.stdout:
        print("Unity executable is Linux-only; use a macOS build or Linux x86-64 runtime.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="jev-mcts")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check credentials and local Unity executable")
    priors_parser = sub.add_parser("prepare-priors", help="query Jev for location priors")
    priors_parser.add_argument("--output", type=Path, default=Path("results/priors"))
    priors_parser.add_argument("--max-usd", type=float, default=5.0)
    generate_parser = sub.add_parser("generate-one", help="generate one Unity task")
    generate_parser.add_argument("--executable", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, default=Path("results/datasets/seen_simple.pik"))
    generate_parser.add_argument("--task", default="put_fridge")
    generate_parser.add_argument("--mode", choices=["simple", "full"], default="simple")
    generate_parser.add_argument("--unseen-apartment", action="store_true")
    generate_parser.add_argument("--seed", type=int, default=42)
    run_parser = sub.add_parser("run-one", help="run one actual Unity episode")
    run_parser.add_argument("--executable", type=Path, required=True)
    run_parser.add_argument("--dataset", type=Path, default=Path("results/datasets/seen_simple.pik"))
    run_parser.add_argument("--priors", type=Path, default=Path("results/priors"))
    run_parser.add_argument("--output", type=Path, default=Path("results/episodes/seen_simple.json"))
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--simulations", type=int, default=20)
    run_parser.add_argument("--max-usd", type=float, default=5.0)
    run_parser.add_argument("--max-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "prepare-priors":
        from .jev import JevClient
        from .prior import generate_priors

        client = JevClient(max_usd=args.max_usd)
        object_path, furniture_path = generate_priors(
            client, Path(__file__).resolve().parent.parent / "data/object_info.json", args.output
        )
        (args.output / "usage.json").write_text(
            __import__("json").dumps(client.usage.as_dict(), indent=2)
        )
        print(f"object priors: {object_path}")
        print(f"furniture priors: {furniture_path}")
        print(f"Jev usage: {client.usage.as_dict()}")
        return 0
    if args.command == "generate-one":
        from .experiment import generate_one

        path = generate_one(
            args.executable.resolve(),
            args.output.resolve(),
            task=args.task,
            mode=args.mode,
            unseen_apartment=args.unseen_apartment,
            seed=args.seed,
        )
        print(f"generated task dataset: {path}")
        return 0
    if args.command == "run-one":
        from .experiment import run_one

        result = run_one(
            args.dataset.resolve(),
            args.executable.resolve(),
            args.priors.resolve(),
            args.output.resolve(),
            seed=args.seed,
            simulations=args.simulations,
            max_usd=args.max_usd,
            max_seconds=args.max_seconds,
        )
        print(f"status: {result['status']}")
        print(f"steps: {len(result['steps'])}")
        print(f"Jev usage: {result['jev_usage']}")
        return 0 if result["status"] == "success" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
