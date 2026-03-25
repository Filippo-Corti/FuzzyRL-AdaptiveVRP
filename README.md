# Fuzzy Reinforcement Learning for the Dynamic Vehicle Routing Problem with Stochastic Disruptions

## Problem Setting

The project models a garbage collection scenario where a fleet of trucks must visit a set of nodes — representing
houses — distributed across a graph. Each node must be visited exactly once. Each truck has a fixed maximum load
capacity and
starts from a central depot. The primary objective is to minimise total route distance across all trucks while
guaranteeing full coverage of all nodes, with routes that are geometrically clean — minimal crossings between truck
paths.

The environment is made non-stationary by a stochastic disruption model: at every round, each active truck has a small
probability of breaking down. When a truck breaks down, it leaves the active fleet and all nodes assigned to its route
become orphaned — unassigned and unvisited. A broken truck recovers after a geometrically distributed number of rounds,
at which point it re-enters the fleet with full capacity and no assigned nodes. This dynamic creates two distinct
recovery challenges: absorbing orphaned nodes quickly after a breakdown, and rebalancing routes efficiently when a truck
returns to the fleet.

## Why This Is Interesting

A static VRP is a combinatorial optimisation problem best addressed by classical heuristics or exact solvers. The
stochastic disruption model transforms it into something fundamentally different: an online decision problem where the
agent must reason under uncertainty at runtime, not just plan offline. The interesting behaviour emerges not from
finding a good initial plan but from adapting to disruptions quickly and well — recovering coverage without
unnecessarily degrading route quality, and exploiting returned capacity without dismantling routes that are already
working.

The two disruption events — breakdown and recovery — require qualitatively different responses. A breakdown demands
urgent coverage: orphaned nodes must be absorbed as fast as possible, prioritising proximity and insertion cost. A
recovery demands geometric refinement: the returned truck represents unused capacity and an opportunity to rebalance
load and reduce route crossings. A single monolithic policy struggles to express both behaviours cleanly. This motivates
a dual-agent architecture where each agent specialises in one recovery mode.

## Why Fuzzy Reinforcement Learning

Two properties of this domain motivate Fuzzy RL specifically over plain RL or a fixed heuristic policy.

The most important signals — truck load, fleet availability, orphan pressure, proximity to orphans, route efficiency,
insertion cost, removal gain — are inherently gradual. A truck at 85% capacity is not full but should start behaving
differently than one at 40%. A fleet at 60% availability is not critically reduced but routes need some rebalancing.
Hard thresholds handle this poorly; fuzzy membership functions handle it gracefully by allowing a state to partially
activate multiple linguistic categories simultaneously.

The policy should also be interpretable. A fuzzy rule base produces readable rules like "IF orphan pressure is HIGH and
nearest orphan is NEAR and insertion cost is LOW → insert" or "IF route imbalance is HIGH and removal gain is HIGH →
remove and reinsert into recovered truck." This interpretability is valuable both for debugging during development and
for explaining the agent's behaviour in a presentation or report.

## Simulation Modes

The simulation operates in three distinct modes that gate which agent is active and when learning occurs:

- **Idle mode** — the solution is fully assigned, geometrically stable, and no disruption has recently fired. No agent
  acts. The simulation waits for the next disruption event. This produces the calm phases visible in the demo where
  routes sit clean and resolved.

- **Breakdown recovery mode** — triggered when a truck breaks down and its nodes become orphaned. The breakdown agent
  runs, absorbing orphans and restoring coverage as fast as possible while maintaining geometric quality. This mode ends
  when orphan pressure returns to zero and the solution stabilises.

- **Fleet rebalancing mode** — triggered when a broken truck recovers and re-enters the fleet with empty routes. The
  rebalancing agent runs, redistributing load toward the recovered truck and improving route geometry. This mode ends
  when load imbalance falls below a threshold and the solution stabilises.

In the ambiguous case where both orphans exist and a truck has just recovered, breakdown recovery takes priority since
coverage is the harder constraint.

## Dual-Agent Architecture

Two independent agents operate within the same hyper-heuristic framework, each with a focused objective and a tailored
state representation.

### Breakdown Recovery Agent

Responsible for fast, clean orphan absorption after a disruption. Operates only in breakdown recovery mode.

State signals:

- Orphan pressure — fraction of nodes currently unassigned
- Nearest orphan distance — normalised distance from this truck's route endpoint to the closest orphan
- Nearest orphan relative distance — this truck's proximity to the nearest orphan relative to the fleet average,
  capturing whether this truck is the natural absorber
- Truck load — fraction of capacity consumed by current planned route
- Fleet availability — fraction of trucks currently active
- Insertion cost — the actual cheapest insertion cost for the best available orphan into this truck's route, normalised
  by average edge distance

Actions: Insert, Remove, 2-opt, Do Nothing

Reward:

```
R = -delta_truck_distance - λ · orphans - γ_low · crossings
```

Where λ = 1.0 and γ_low = 0.1, prioritising coverage speed over geometric perfection.

### Fleet Rebalancing Agent

Responsible for load redistribution and route quality improvement after a truck returns. Operates only in fleet
rebalancing mode.

State signals:

- Route imbalance — standard deviation of load fractions across active trucks
- Truck load — fraction of capacity consumed by current planned route
- Route efficiency — actual route distance per node normalised by expected average
- Fleet availability — fraction of trucks currently active
- Removal gain — the actual distance saved by the best removal from this truck's route, normalised by average edge
  distance

Actions: Insert, Remove, 2-opt, Do Nothing

Reward:

```
R = -delta_truck_distance - μ · imbalance - γ_high · crossings
```

Where μ = 1.0 and γ_high = 0.3, prioritising geometric quality and load balance since coverage is already guaranteed.

## Action Space

All actions are defined at the level of a single truck. Both agents share the same action set:

- **Do nothing** — the truck skips its turn. Crucial for enabling anticipatory behaviour and avoiding unnecessary route
  disruption during stable phases.
- **Insert** — the heuristic finds the position in this truck's current planned route where inserting the nearest
  orphaned node adds the least additional distance.
- **Remove** — the heuristic identifies the node in this truck's current planned route whose removal yields the greatest
  reduction in route distance and drops it, making it an orphan.
- **2-opt** — the heuristic finds the single best improving 2-opt swap across all pairs of edges in this truck's planned
  route and applies it. If no improving swap exists the action has no effect.

The agent never chooses which specific node to insert or remove. The heuristic determines that internally. The agent
only decides which type of operation to invoke.

## Learning Algorithm: Fuzzy Q(λ)

Both agents use the same learning algorithm: Fuzzy Q-learning extended with eligibility traces. The Q-table is indexed
by fuzzy state label combinations rather than crisp state values. Each crisp observation value is passed through
triangular membership functions that produce a dictionary of label→membership pairs. A value like fleet availability of
0.65 might activate Reduced at 0.7 and Full at 0.3 simultaneously. Both activated labels contribute to the Q-value
lookup and update, weighted by their membership values.

Eligibility traces extend standard Q-learning by maintaining a trace value for every Q-table entry, decaying over time,
and updating all entries proportionally to their trace at every step. This is particularly important for the
remove-then-insert sequence: removing a node looks neutral or slightly negative immediately but pays off when the freed
node is reinserted more cheaply elsewhere. Traces propagate credit back to the removal decision.

Traces are reset at each mode transition — when a disruption fires or a truck recovers, past decisions become less
relevant to current credit assignment.

## Adaptive Membership Functions

The breakpoints of each triangular membership function are stored as learnable parameters rather than fixed values.
After each Q-update, the breakpoints are nudged slightly in the direction that would have increased the Q-value of the
chosen action. This means each agent learns not just what to do in each fuzzy state, but also what the right way to
perceive its specific state space is. The two agents may develop different membership function shapes, reflecting the
different value distributions relevant to each recovery mode.

## Shared Policy Across Trucks

Within each agent, a single Q-table and a single set of membership function parameters are shared across all trucks. The
agent is invoked once per truck per round, seeing the world through that truck's eyes. Every experience from every truck
updates the same Q-table, so the agent learns from the full fleet's behaviour simultaneously. Trucks are assumed
homogeneous — same capacity, same speed, same cost per unit distance.

## Greedy Baseline

A greedy policy serves as the comparison baseline. In breakdown recovery mode it always inserts the nearest orphan using
cheapest insertion when orphans exist and applies 2-opt otherwise. In fleet rebalancing mode it removes the node with
the highest removal gain from the most loaded truck and inserts it into the least loaded truck. It has no learning
component and no fuzzy state representation.

## Evaluation Metrics

- **Recovery speed** — after each breakdown, count the number of rounds until orphan pressure returns to zero. Average
  across many disruption events. Lower is better.
- **Recovery quality** — after stabilisation following a breakdown, compute the total route distance of the recovered
  solution. Compare against the greedy baseline's distance on the same post-disruption graph.
- **Rebalancing quality** — after a truck returns, measure the reduction in route imbalance and crossing count achieved
  before idle mode is re-entered. Compare against the greedy rebalancing baseline.
- **Learning curves** — total reward per mode episode over training for each agent independently. Should trend upward
  and stabilise.
- **Ablation** — the same experiment run with crisp Q(λ) instead of fuzzy membership, isolating the contribution of the
  fuzzy representation.

