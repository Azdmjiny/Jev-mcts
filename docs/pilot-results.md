# Eight-task pilot: 2026-09-29

The local Jev–MCTS and Unity loop completed all eight fixed-seed tasks. Six
succeeded within 30 real actions. The complete compact action trajectories,
goals, per-task usage and model version are in
[`experiments/pilot-2026-09-29.json`](../experiments/pilot-2026-09-29.json).
Full per-step search values remain in the ignored local `results/pilot/` output.

| Apartment | Task type | Unity result | Real actions | Jev calls |
| --- | --- | ---: | ---: | ---: |
| Seen | Simple | Success | 5 | 20 |
| Seen | Comp | Failed at step limit | 30 | 611 |
| Seen | NovelSimple | Success | 5 | 34 |
| Seen | NovelComp.(2) | Success | 10 | 90 |
| Unseen | Simple | Success | 5 | 31 |
| Unseen | Comp | Failed at step limit | 30 | 630 |
| Unseen | NovelSimple | Success | 4 | 11 |
| Unseen | NovelComp.(2) | Success | 10 | 107 |

Both Comp tasks required a wineglass and a coffeepot on the kitchen table.
The planner placed the wineglass, then repeatedly walked toward a coffeemaker
instead of finding the coffeepot. The two failures were planning failures:
Unity reported zero failed action executions across all eight tasks.

Configuration: seed 42, 20 MCTS simulations per real action, at most 30 real
actions per task. Jev resolved `jev-latest` to `jev-1.13.0`. Task execution
used 1,534 Jev calls, 4,524,098 input tokens and an estimated $0.1900 of
input-token charges. Preparing location priors separately used 104 calls and
an estimated $0.0030. The pilot took 1,224 seconds (20.4 minutes), below
the $5 and two-hour limits. The estimate is based on the documented input
token rate; it is not a billing statement.

A straight-line estimate for 800 similarly distributed tasks is about $19 in
Jev input charges and 34 hours of local runtime, plus one prior-preparation
run. This is an extrapolation from eight tasks; task mix, prompt size and
failure frequency can change it substantially. The full benchmark has not
been run. These eight observations do not establish a statistically reliable
success rate or a direct comparison with the paper's GPT results.
