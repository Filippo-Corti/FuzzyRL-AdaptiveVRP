## TODO:

[X] Literature review
[X] Graph + trucks + Pygame scaffold together
[X] Heuristics with visual before/after
[X] Disruption model with visual indicators
[X] Greedy baseline — watch it run visually, verify it looks sane
[] Q-learning agent without fuzzy — watch it learn visually, verify it looks sane
[] Fuzzy system — print state to screen
[] Q-learning + fuzzy — live reward plot
[] Eligibility traces + adaptive memberships
[] Evaluation and ablation study

Critical concerns:
> The state has no information about the truck's route quality. We should add a "route efficiency" that divides the
> current route length by a lower-bound. This will help the agent learn to prefer better routes, and also give it a more
> meaningful state representation.

> The state has no information about the truck's contribution when adding an orphan. We could express the relative
> proximity to the nearest orphan, w.r.t. the other trucks.

> We could add a new insertion heuristic that chooses the cheapest orphan.

> Introducing Fuzzy Logic to the state

> Introducing Eligibility Traces to learn n-step returns

## Repository Structure

vrp_fuzzy_rl/
│
├── README.md
├── requirements.txt
├── config.py # all hyperparameters and constants in one place
│
├── env/
│ ├── __init__.py
│ ├── graph.py # nodes, edges, distance matrix
│ ├── truck.py # truck state, capacity, route, active/broken
│ ├── fleet.py # manages all trucks, disruption model
│ └── episode.py # episode loop, reward computation, action masking
│
├── heuristics/
│ ├── __init__.py
│ ├── insertion.py # cheapest insertion, regret insertion
│ ├── improvement.py # 2-opt
│ └── swap.py # inter-route swap, rebuild
│
├── fuzzy/
│ ├── __init__.py
│ ├── membership.py # triangular/trapezoidal mf, adaptive breakpoints
│ └── fuzzifier.py # fuzzify_capacity, fuzzify_fleet, etc.
│
├── agent/
│ ├── __init__.py
│ ├── qtable.py # Q-table structure, weighted lookup
│ ├── traces.py # eligibility trace management
│ └── agent.py # action selection, Q update, ties together fuzzy+qtable+traces
│
├── baselines/
│ ├── __init__.py
│ └── greedy.py # greedy policy for comparison
│
├── viz/
│ ├── __init__.py
│ ├── renderer.py # pygame drawing: graph, trucks, routes, overlays
│ └── dashboard.py # live reward plot, fuzzy state display, action taken
│
├── evaluation/
│ ├── __init__.py
│ ├── metrics.py # recovery speed, recovery quality, recovery ratio
│ └── plots.py # learning curves, comparison plots (matplotlib)
│
├── train.py # training entry point
├── evaluate.py # evaluation entry point, runs trained agent vs greedy
└── run_viz.py # runs a trained agent with full pygame visualisation