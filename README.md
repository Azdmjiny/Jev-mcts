# Jev-mcts

An experimental Jev-guided MCTS planner for VirtualHome household tasks, derived
from [LLM-MCTS](https://github.com/1989Ryan/llm-mcts) (Apache-2.0). The first
milestone is a single end-to-end task, followed by a bounded eight-task pilot.

Jev receives a state and candidate locations or valid actions; it returns a
probability distribution used by the belief model or the MCTS policy prior.
VirtualHome executes actions and decides whether a task succeeded.

## Local setup

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,virtualhome]'
export VIRTUALHOME_EXECUTABLE=/absolute/path/to/VirtualHome/executable
.venv/bin/jev-mcts doctor
```

Put `TYPESAFE_API_KEY=...` in the ignored `.env` file, or export it in the
shell that starts the experiment. Do not commit keys or Unity binaries.

The existing local Unity binary from LLM-MCTS is Linux x86-64. On Apple Silicon
macOS, use the official macOS Unity build or run a Linux x86-64 environment.
The doctor command detects incompatible executables before an experiment.

## Provenance

The planning design follows Zhao et al., *Large language models as commonsense
knowledge for large-scale task planning* (NeurIPS 2023). VirtualHome is a
separate simulator dependency; its files and license are not redistributed here.
