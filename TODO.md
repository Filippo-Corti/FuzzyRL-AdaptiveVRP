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

## Revised Implementation Plan

### Phase 1 — Simulation mode structure

This is the foundation everything else builds on. Do this first before touching agents or observations.

Add SimulationMode enum with IDLE, BREAKDOWN_RECOVERY, FLEET_REBALANCING. Add mode transitions to VRPSimulation.step() —
breakdown fires → enter breakdown recovery, orphans reach zero and solution stable → enter idle, truck recovers → enter
fleet rebalancing, imbalance below threshold and solution stable → enter idle. Add stability counter. Add visual
indicator of current mode to the HUD. Verify visually that the three phases look distinct.

### Phase 2 — Dual agent scaffold (1-2 days)

Create BreakdownAgent and RebalancingAgent as separate classes both inheriting from VRPAgent. For now both can be copies
of the existing CrispQLambdaAgent with different observation subsets. Wire the simulation to call the correct agent
based on current mode. Verify that each agent only receives updates during its own mode. Log training events per agent
to confirm both are being trained.

### Phase 3 — Observation additions (1 day)

Add insertion_cost to the breakdown agent observation — the cheapest insertion cost for the best available orphan into
this truck's route, normalised by average edge distance. Add removal_gain to the rebalancing agent observation — the
distance saved by the best removal from this truck's route. Update bin definitions for each agent independently. These
two signals are the highest impact additions for eliminating the stupid decisions you are seeing.

### Phase 4 — Reward tuning (1 day)

Implement the two distinct reward functions as described above. Breakdown agent uses low crossing penalty. Rebalancing
agent uses high crossing penalty and imbalance term. Run 50k steps with each agent in isolation if possible — give
breakdown agent a scenario with frequent disruptions, give rebalancing agent a scenario with frequent recoveries. Verify
that learning curves trend in the right direction for each.

### Phase 5 — Fuzzy layer (2-3 days)

Replace crisp bins with triangular membership functions in both agents. Each observation value produces a dictionary of
label→membership pairs. Q-table is now indexed by fuzzy label combinations weighted by membership. Implement the fuzzy
Q-update and fuzzy trace accumulation. Start with fixed membership function breakpoints before adding adaptive ones.
Verify that the Q-table grows more smoothly than with crisp bins and that the HUD membership bars show partial
activation.

### Phase 6 — Adaptive membership functions (1-2 days)

Add learnable breakpoints to the triangular membership functions. After each Q-update nudge breakpoints in the direction
that would have increased the chosen action's Q-value. Keep nudge step size small — around 0.001. Monitor whether
breakpoints drift meaningfully or stay near their initialisation. This is the academically novel component so document
what happens carefully.

### Phase 7 — Evaluation and ablation (2 days)

Run the full system for 200k steps. Record recovery speed, recovery quality, and rebalancing quality metrics. Run the
same experiment with crisp Q(λ) as ablation. Run the greedy baseline. Produce the four evaluation plots. The story to
tell is: greedy baseline establishes a floor, crisp Q(λ) improves on it, fuzzy Q(λ) improves further, adaptive
memberships show the system learning its own perception of the state space.

### Phase 8 — Visualisation polish (1 day)

Add mode indicator to the Pygame window — colour coded, clearly visible. Show which agent is active. Show eligibility
trace strength in the HUD for the top few state-action pairs. Make sure the calm-disruption-recovery-calm cycle reads
clearly to a viewer who has never seen the project before. This is your demo and it should be self-explanatory.