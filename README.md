# Fuzzy Reinforcement Learning for the Dynamic Vehicle Routing Problem with Stochastic Disruptions

## Problem Setting

The project models a garbage collection scenario where a fleet of trucks must visit a set of nodes — representing
houses — distributed across a graph. Each node must be visited exactly once. Each truck has a fixed maximum load
capacity and starts from a central depot. The primary objective is to minimise total route distance across all trucks
while guaranteeing full coverage of all nodes.

The environment is made non-stationary by a stochastic disruption model: at every round, each active truck has a small
probability of breaking down. When a truck breaks down, it leaves the active fleet and all nodes assigned to its route
become orphaned — unassigned and unvisited. A broken truck recovers after a geometrically distributed number of rounds,
at which point it re-enters the fleet with full capacity and no assigned nodes. This dynamic creates a perpetual tension
between optimising existing routes and adapting to sudden changes in fleet composition.

## Why This Is Interesting

A static VRP is a combinatorial optimisation problem best addressed by classical heuristics or exact solvers. The
stochastic disruption model transforms it into something fundamentally different: an online decision problem where the
agent must reason under uncertainty at runtime, not just plan offline. The interesting behaviour emerges not from
finding a good initial plan but from adapting to disruptions quickly and well — recovering coverage without
unnecessarily degrading route quality.

The two dimensions of recovery quality matter independently. Recovery speed measures how quickly orphaned nodes get
reassigned after a disruption. Recovery quality measures how close the resulting solution is to what an optimal
re-planner would produce on the same post-disruption graph. A good agent should be fast and good, not just one or the
other.

## Why Fuzzy Reinforcement Learning

Two properties of this domain motivate Fuzzy RL specifically over plain RL or a fixed heuristic policy.

The most important signals — truck load, fleet availability, orphan pressure, proximity to orphans — are inherently
gradual. A truck at 85% capacity is not full but should start behaving differently than one at 40%. A fleet at 60%
availability is not critically reduced but routes need some rebalancing. Hard thresholds handle this poorly; fuzzy
membership functions handle it gracefully by allowing a state to partially activate multiple linguistic categories
simultaneously.

The policy should also be interpretable. A fuzzy rule base produces readable rules like "IF fleet is CRITICALLY REDUCED
and orphan pressure is HIGH and nearest orphan is NEAR → insert orphan via cheapest insertion." This interpretability is
valuable both for debugging during development and for explaining the agent's behaviour in a presentation or report.

## Architecture: Hyper-Heuristic Fuzzy Q-Learning

The agent operates as a hyper-heuristic: it never manipulates individual nodes or edges directly. Instead, at each
decision step it chooses which repair or improvement procedure to invoke, and the chosen heuristic handles all low-level
geometry internally. This keeps the action space small and tractable while still allowing the agent to express a rich
range of behaviours.

The learning algorithm is Fuzzy Q(λ) — Q-learning extended with eligibility traces and a fuzzy state representation. The
Q-table is indexed by fuzzy state label combinations rather than crisp state values. The update rule propagates credit
backwards through recent decisions weighted by their eligibility trace values, and both the Q-update and the trace
accumulation are weighted by fuzzy membership values.

## Episode Structure

All nodes begin as orphans and all trucks begin at the depot with empty routes. The agent iterates over active trucks
one by one each round; each truck observes its local fuzzy state and selects an action, the chosen heuristic executes,
the reward is computed, and Q and traces are updated. After all trucks have acted, the disruption model fires — each
active truck has probability p of breaking down, each broken truck has probability q of recovering.

This structure unifies the initial route construction phase and the disruption repair phase under the same mechanism.
The agent learns one policy that handles both building routes from scratch at the start of an episode and repairing them
after disruptions occur mid-episode.

The simulation runs continuously — there is no terminal state. The episode boundary used for resetting eligibility
traces is each disruption event, since the structural situation changes at that point and past decisions become less
relevant to current credit assignment.

## State Space

At each decision point the agent observes a state from the perspective of the truck currently acting. Four crisp values
are computed and then fuzzified:

- Truck remaining capacity is the fraction of this truck's load capacity not yet consumed by its currently planned
  route — (capacity - planned_load) / capacity. This is planned remaining capacity, not physical current load, since the
  truck is at the depot at decision time. Membership sets: Empty, Plenty, Tight, Almost Gone.
- Fleet availability is the fraction of trucks currently active — active_trucks / total_trucks. This is the primary
  disruption signal; when it drops the agent should shift toward absorbing orphans rather than improving existing
  routes. Membership sets: Full, Reduced, Critically Reduced.
- Orphan pressure is the fraction of total nodes currently unassigned — orphaned_nodes / total_nodes. This quantifies
  how much uncovered work exists globally. Membership sets: None, Low, Moderate, High.
- Nearest orphan distance is the normalised distance from this truck's current route endpoint — the last node in its
  planned sequence — to the closest orphaned node. This is truck-specific and captures how expensive it would be for
  this particular truck to absorb the nearest orphan. Membership sets: Near, Medium, Far.
- A fifth signal, route imbalance, captures the spread of load fractions across active trucks — computed as the standard
  deviation of planned_load / capacity across the fleet. This is fuzzified as Balanced, Uneven, Skewed and informs the
  reward function as well as the state.

Each crisp value is passed through triangular membership functions that produce a dictionary of label→membership pairs.
A value like fleet availability of 0.65 might activate Reduced at 0.7 and Full at 0.3 simultaneously. Both activated
labels contribute to the Q-value lookup and update, weighted by their membership values.

## Adaptive Membership Functions

The breakpoints of each triangular membership function are stored as learnable parameters rather than fixed values.
After each Q-update, the breakpoints are nudged slightly in the direction that would have increased the Q-value of the
chosen action. This means the agent learns not just what to do in each fuzzy state, but also what the right way to
perceive the state space is.

This is the key architectural feature that distinguishes the system from "tabular Q-learning with a preprocessing step."
The fuzzy layer is genuinely learned, not hand-designed.

## Action Space

All actions are defined at the level of a single truck. The agent chooses one per turn:

- Do nothing — the truck skips its turn. This action is always available and is crucial for enabling anticipatory
  behaviour: an agent that learns to wait for a recovering truck rather than expensively absorbing its orphans has
  discovered something non-trivial.
- Insert — the heuristic finds the position in this truck's current planned route where inserting the nearest orphaned
  node adds the least additional distance. Fast and greedy, appropriate under severe disruption when coverage matters
  more than optimality.
- Remove — the heuristic identifies the node in this truck's current planned route whose removal yields the greatest
  reduction in route distance and drops it, making it an orphan. Appropriate when the truck carries a geometrically
  poor node that would be better served by another truck.
- 2-opt — the heuristic finds the single best improving 2-opt swap across all pairs of edges in this truck's planned
  route and applies it. If no improving swap exists the action has no effect. Reduces route distance without changing
  which nodes belong to this truck. Appropriate when the state is stable and the agent is in improvement mode.

The agent never chooses which specific node to insert or remove. The heuristic determines that internally using
deterministic procedures. The agent only decides which type of operation to invoke.

## Action Masking

At each decision point, a masking function computes the set of legal actions for the current truck before the agent
selects. Insertion actions are masked if the truck's remaining planned capacity is zero — it cannot absorb more nodes.
Swap is masked if no other truck has more nodes than this one. 2-opt is always available if the route has at least three
nodes. Do nothing is always available.

Masking enforces the capacity constraint structurally. The agent physically cannot overfill a truck, so this never needs
to be learned. The agent only ever chooses among legal actions.

## Route Construction

Routes are never planned all at once. They emerge incrementally from the sequence of insertion actions the agent takes.
Each time an insertion heuristic is invoked, one orphaned node gets added to a truck's route in the best available
position according to that heuristic. By the time all nodes are assigned, each truck has a complete ordered sequence.
2-opt actions then improve the geometry of these incrementally-built routes. The agent never faces a TSP instance
directly.

## Reward Function

The reward is computed after every action — it is dense, not sparse:

```
R = -total_distance - λ · unvisited_nodes - μ · imbalance$
```

Where total_distance is the sum of route distances across all trucks given current planned routes, unvisited_nodes is
the count of orphaned nodes, and imbalance is the standard deviation of load fractions across active trucks.

The three terms have distinct and non-overlapping roles. Total distance is the primary objective — minimise driving.
Unvisited nodes enforces coverage — every node must be visited. Imbalance incentivises using available capacity
including recovered trucks — an idle returned truck while others are heavily loaded is penalised directly.

The weights are set by normalising to the same scale rather than tuned arbitrarily. λ is set to the average edge
distance in the graph so that one unvisited node costs approximately as much as an average route segment. μ is set to
total_distance / 4 so that a maximally imbalanced fleet costs roughly as much as a meaningfully worse route. This
principled scaling means the weights reflect deliberate operational priorities rather than arbitrary hyperparameter
choices.

## Q(λ): Eligibility Traces

Standard Q-learning updates only the Q-table entry for the state-action pair just taken. Eligibility traces extend this
by maintaining a trace value for every Q-table entry, decaying over time, and updating all entries proportionally to
their trace at every step.

The motivation in this problem is that the consequence of a decision often materialises several steps later. Choosing "
do nothing" because a truck is about to recover looks neutral or slightly bad immediately — orphans are still sitting
there — but pays off when the recovered truck absorbs them cheaply. Traces propagate credit back to that early patient
decision.

The trace accumulation and decay are weighted by fuzzy membership values, exactly like the Q-update. A fuzzy state that
was only partially active accumulates a weaker trace and receives weaker credit propagation. The fuzziness permeates the
entire learning mechanism.

The λ parameter controls temporal depth: λ=0 recovers standard Q-learning, λ=1 propagates credit all the way back. A
value around 0.8–0.9 is a reasonable starting point.

##Shared Policy Across Trucks

A single Q-table and a single set of membership function parameters are shared across all trucks. The agent is invoked
once per truck per round, seeing the world through that truck's eyes. It does not remember that it just acted as truck A
when it is now acting as truck B — each invocation is independent.

This works because trucks are assumed homogeneous: same capacity, same speed, same cost per unit distance. The shared
policy learns a generalised strategy that any truck can execute given its local situation. Every experience from every
truck — good and bad — updates the same Q-table, so the agent learns from the full fleet's behaviour simultaneously.

This is not a multi-agent system. It is one agent embodied sequentially in multiple trucks.

## Greedy Baseline

A greedy policy serves as the comparison baseline for evaluation. It always inserts the nearest orphan using cheapest
insertion when orphans exist, applies 2-opt whenever available, and never does nothing. It has no learning component and
no fuzzy state representation. It is a reasonable approximation of what a competent human dispatcher might do without
any learning.

The baseline is implemented as a drop-in replacement for the agent — it takes the same state input and returns an
action — so it can be evaluated in the same episode loop without any changes to the environment.

## Evaluation Metrics

Recovery speed: after each disruption event, count the number of rounds until orphan pressure returns to zero. Average
across many disruption events. Lower is better.

Recovery quality: after stabilisation following a disruption, compute the recovery ratio — total distance of the
recovered solution divided by the total distance the greedy baseline produces when re-planning from scratch on the same
post-disruption graph. A ratio close to 1 means the agent recovered as well as the greedy baseline. Below 1 means it did
better.

Learning curve: total reward per episode over training. Should trend upward and stabilise, demonstrating that learning
is occurring.
Ablation: the same experiment run with plain tabular Q-learning using crisp state bins instead of fuzzy membership. This
isolates the contribution of the fuzzy representation from the contribution of the learning algorithm itself.

## Implementation Stack

The project is implemented in Python. The environment, heuristics, fuzzy system, and agent are pure Python with numpy
for distance computations. Pygame handles real-time visualisation during training and demonstration. Matplotlib
generates evaluation plots. The codebase is structured so that the environment never imports from the agent, the agent
never imports from Pygame, and all hyperparameters live in a single configuration file.