# Jev-mcts

Jev-guided Monte Carlo tree search for VirtualHome household tasks. This is an
independent, runnable adaptation of [LLM-MCTS](https://github.com/1989Ryan/llm-mcts)
(Apache-2.0). It uses the original task generator, Unity environment and Python
evolving-graph simulator as external runtime dependencies.

## What Jev does

1. **Location belief:** input is an object name and candidate `INSIDE`/`ON`
   household locations. Jev returns a probability for each candidate. The
   probabilities initialize a belief over hidden object locations; actual Unity
   observations replace visible relations and exclude inspected empty containers.
2. **Action prior:** input is the current goal, visible object classes, action
   history and the currently legal Unity actions. Jev returns one probability
   per action. MCTS uses these probabilities to prioritize search, then uses
   rewards from the Python graph simulator to choose one action for Unity.

The output of each real Unity action is a new partial observation and the Unity
success signal. The planner has at most 30 real actions per task. It does not
require expert trajectories or LLM-generated goal parsing.

## Local setup (Apple Silicon macOS)

```sh
git clone git@github.com:Azdmjiny/Jev-mcts.git
cd Jev-mcts
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,virtualhome]'
bash scripts/bootstrap_runtime.sh
.venv/bin/jev-mcts doctor --executable runtime/macos/macos_exec.v2.3.0.app
```

`bootstrap_runtime.sh` checks out the pinned original source, initializes its
VirtualHome submodule, and downloads the official macOS Unity v2.3.0 app. It
verifies the app archive's SHA-256 before extracting. These large files are
ignored by Git. If the source is already installed locally, pass its directory
to the script: `bash scripts/bootstrap_runtime.sh /path/to/llm-mcts`.

Put `TYPESAFE_API_KEY=...` in the ignored `.env` file, or export the variable
in the terminal that starts the run. Do not commit keys. The `doctor` command
reports only whether a key is present. On Linux, use a native x86-64 machine
and the Linux Unity executable from the original runtime; its binary fails
under the tested Apple Silicon Linux emulation.

## First task, then the eight-task pilot

```sh
.venv/bin/jev-mcts prepare-priors --output results/priors
.venv/bin/jev-mcts generate-one \
  --executable runtime/macos/macos_exec.v2.3.0.app \
  --output results/datasets/seen_simple.pik \
  --task put_fridge --mode simple --seed 42
.venv/bin/jev-mcts run-one \
  --executable runtime/macos/macos_exec.v2.3.0.app \
  --dataset results/datasets/seen_simple.pik \
  --priors results/priors --output results/episodes/seen_simple.json \
  --seed 42 --simulations 20
.venv/bin/jev-mcts pilot \
  --executable runtime/macos/macos_exec.v2.3.0.app \
  --priors results/priors --output results/pilot \
  --seed 42 --simulations 20 --max-usd 5 --max-seconds 7200
```

The pilot runs one task in each of `Simple`, `Comp`, `NovelSimple` and
`NovelComp.(2)` in both Seen and Unseen apartments. The source generator's
`unseen-item` name is misleading: its selected objects occur in ordinary
training tasks, but their object–destination pairs are absent there. It
therefore supplies the NovelSimple and NovelComp tasks. `unseen_comp` combines
two of those pairs. All eight cases use seed 42. The pilot writes generated
datasets, per-action episode JSON and `summary.json` under `results/pilot/`.
Running the same command again skips completed cases and replays a partial
Unity trajectory before continuing it. The $5 estimated Jev input-token
budget and two-hour wall-clock limit apply across cases and restarts.

The first task can be run alone before the pilot. If the pilot should reuse its
result, copy `seen_simple.pik` and `seen_simple.json` into `results/pilot/datasets/`
and `results/pilot/episodes/` respectively before starting the pilot.

The completed local pilot and its limitations are recorded in
[`docs/pilot-results.md`](docs/pilot-results.md), with compact trajectories in
[`experiments/pilot-2026-09-29.json`](experiments/pilot-2026-09-29.json).

## Search and provenance

Each real decision runs 20 Python-graph simulations by default. Like the
released LLM-MCTS source, this implementation samples one belief graph per
decision and reuses it for those simulations, identifies tree nodes by action
history, and updates action values with a softmax-weighted average of returns.
The paper pseudocode instead describes sampling a scene per simulation. This
pilot evaluates the released-source variant with Jev priors; it is not an exact
replication of the paper's full benchmark or few-shot GPT policy.

The source and license are retained in `LICENSE`. VirtualHome and its Unity
binary are separate downloads with their own licensing and provenance. The
Jev API endpoint is the official TypeSafe System One service; `jev-latest`
resolves to a version recorded in each episode's usage.
