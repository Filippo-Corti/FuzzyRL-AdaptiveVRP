# VRP Experimental Framework: Fuzzy RL vs Transformer vs Heuristic
## Project Report & Implementation Plan

---

## 1. Problem Statement

The objective is to compare three decision-making strategies for a single-vehicle routing problem under realistic,
structured conditions. The central question is whether learned policies — one interpretable (Fuzzy RL), one
black-box (Transformer) — can outperform a strong domain-informed heuristic, and what each gains and loses
relative to the other.

This is not a claim that one method is universally better. The Transformer is expected to achieve the best route
cost asymptotically. The Fuzzy RL agent is expected to close most of that gap while remaining fully interpretable.
The heuristic provides the honest baseline that neither learned method should be compared against a trivial policy.

---

## 2. Environment Design

### 2.1 Geometry

Customer locations are not uniformly distributed. Each episode generates:

- 4–5 spatial clusters of varying density and radius
- 1–2 isolated outlier customers placed outside cluster regions
- Total: approximately 50 customers per episode

This geometric structure is essential. It rewards policies that reason about global routing structure — visiting a
dense cluster in sequence rather than bouncing between distant nodes — and penalises purely myopic greedy strategies.
Instances are randomly generated each episode using fixed structural parameters (cluster count, spread, outlier
fraction), so no method can memorise a fixed instance.

### 2.2 Capacity and Demand

- The single truck has a fixed maximum load capacity
- Each customer has a variable demand drawn from a distribution (e.g. uniform over [1, 5])
- When the truck's remaining capacity cannot serve the next selected customer, the truck must return to the depot,
  reload to full capacity, and continue
- Depot returns are part of the decision sequence — the agent must decide when returning is necessary, not just
  which customer to serve next

### 2.3 Dynamic Arrivals

- Approximately 70% of customers are placed at episode start
- The remaining 30% arrive dynamically during the episode at random times
- Each newly arrived customer receives a time window that starts counting from the moment of their arrival
- Urgency for customer i is defined as: `urgency_i = time_elapsed_since_arrival / window_length`
- This is a soft constraint: serving a customer after their window expires incurs a lateness penalty rather than
  making the route infeasible. No route is ever infeasible.

### 2.4 Distance Model

The environment uses a fully connected cost matrix with pure Euclidean distances. All three agents observe and
reason over the same clean distance information. There is no travel time noise or congestion model — the
interesting challenge comes entirely from routing structure: clusters, variable demand, dynamic arrivals, and
soft time windows. The visualisation renders straight lines between nodes, with urgency colouring and route
traces providing the visual richness.

### 2.5 Objective Function

Total episode cost is:

```
C = total_travel_distance + α · Σ max(0, lateness_i)
```

Where `lateness_i` is the time by which customer i was served after their window expired, and α is the lateness
penalty weight. α should be calibrated so that there is genuine tension between serving a nearby cluster and
handling an urgent isolated customer. A practical calibration method: run TONN with pure distance-greedy vs
pure urgency-greedy on 20 instances, set α to the value where TONN's combined heuristic clearly beats both extremes.

---

## 3. Methods

### 3.1 Baseline: Time-Oriented Nearest Neighbour (TONN)

TONN is a strong handcrafted heuristic. At each step it selects the next customer by scoring all unvisited
reachable candidates:

```
score(c) = w_d · normalised_distance(c) + w_u · urgency(c) + w_f · feasibility_penalty(c)
```

Where feasibility_penalty is zero if the truck can serve c without violating capacity, and a large constant
otherwise. TONN returns to the depot when no feasible customer exists.

TONN is not a trivial baseline. It handles urgency, distance, and capacity simultaneously. It is the reference
point both learned methods must beat to be meaningful.

### 3.2 Fuzzy Reinforcement Learning Agent

#### Architecture

The Fuzzy RL agent scores each candidate customer independently using a fuzzy inference system, then selects the
highest-scoring candidate. The agent does not choose from a discrete action set — the selection is implicit in the
scoring. This is the correct formulation for routing: the "action" is which customer to serve next, and the policy
is expressed as a scoring function over candidates.

#### Per-candidate features (inputs to fuzzy inference)

| Feature | Description |
|---|---|
| Path cost | Shortest-path cost from current truck position to candidate under actual edge weights, normalised by mean edge weight |
| Urgency | `time_elapsed / window_length` for this candidate, in [0, 1] |
| Demand ratio | Candidate demand divided by truck remaining capacity |
| Cluster density | Number of unvisited customers within radius r of candidate, normalised |
| Detour cost | Shortest-path cost via candidate vs direct path to depot, normalised by mean edge weight |

Distance is replaced by shortest-path cost under actual weights, not Euclidean distance. This is the correct
signal given the sparse graph structure: a geometrically close node may be expensive to reach if the connecting
edges are congested. The cluster density signal is still critical — without it the fuzzy agent is blind to global
structure and will plateau near greedy performance on clustered instances.

#### Fuzzy membership functions

Each feature is mapped to linguistic labels via triangular (or gaussian?) membership functions:
- Distance: {Near, Medium, Far}
- Urgency: {Low, Moderate, High}
- Demand ratio: {Light, Moderate, Heavy}
- Cluster density: {Sparse, Moderate, Dense}
- Detour cost: {Cheap, Moderate, Expensive}

Breakpoints are stored as learnable parameters and updated after each episode.

#### Rule base

The rule base maps label combinations to a priority score. Example rules:

- IF Urgency=High AND Distance=Near → priority VERY_HIGH
- IF ClusterDensity=Dense AND DetourCost=Cheap → priority HIGH
- IF DemandRatio=Heavy AND Distance=Far → priority LOW
- IF Urgency=Low AND ClusterDensity=Sparse → priority VERY_LOW

Rule weights are learnable parameters updated by the REINFORCE signal.

#### Learning

REINFORCE with TONN advantage:

```
advantage = C_TONN(instance) - C_agent(instance)
```

Where both costs are computed on the same episode instance. The gradient updates both the rule weights and the
membership function breakpoints. This advantage formulation is correct: it measures improvement over the
heuristic on the same geometry, removing instance difficulty as a confound.

### 3.3 Transformer Agent

The Transformer follows the Kool et al. (2019) architecture with minor adaptations for this environment.

#### Node features (encoder input)

Each customer node is represented as a vector:
- x, y coordinates (normalised)
- Demand (normalised by truck capacity)
- Urgency (0 for static customers at episode start; updated as time passes for dynamic arrivals)
- Binary: already visited
- is_depot flag

Distances are Euclidean throughout. No edge features are needed — the attention mechanism learns pairwise
relationships from node coordinates directly, which is the standard Kool et al. formulation.

The depot is a special node with zero demand and zero urgency.

#### Dynamic arrivals

When a new customer arrives mid-episode, it is added to the node set with its current urgency value. The encoder
re-runs on the updated node set. This is the standard way to handle dynamic arrivals in attention-based VRP solvers
and requires no architectural change.

#### Training

REINFORCE with TONN advantage, identical signal to the Fuzzy agent. This ensures the comparison is fair — both
methods optimise against the same baseline on the same instances.

---

## 4. Evaluation Protocol

### 4.1 Metrics

Two primary metrics:

1. **Total route cost** — the objective function value C defined in Section 2.5. Lower is better. Report mean
   and standard deviation over a held-out test set of 200 instances.

2. **Interpretability demonstration** — for the Fuzzy agent, extract the top-10 most frequently fired rules
   during evaluation and display them with their membership activations. This is qualitative but essential for
   the thesis argument.

Secondary metrics (reported but not the focus):
- Recovery time under lateness pressure (how quickly urgency-aware routing prevents penalty accumulation)
- Performance gap vs TONN across instance difficulty buckets (easy = few outliers, hard = many outliers)

### 4.2 Test Set

200 instances generated with the same structural parameters as training but never seen during training. All three
methods evaluated on the same 200 instances. Report mean ± std for each method.

### 4.3 Ablation

Run Fuzzy agent with crisp features (hard thresholds replacing membership functions, no fuzzy aggregation).
This isolates the contribution of the fuzzy representation from the learning component.

