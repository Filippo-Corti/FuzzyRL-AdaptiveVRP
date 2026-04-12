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


---

Some other notes once again:
- I need to check the visualization package as I have not checked it yet.
- I need to find a way to pass some information about the training to the visualization, so that I can plot the baseline and the adv_ema on the HUD.
- I want a better way to uniformly configure the characteristics of the instances I am running on. These are the parameters of the BatchEnv but they should be shared across training and visualization.
- I need to check if the trained metrics actually improve over time.
- I want to add an exact solver for the VRP instances I am visualizing, so that I can show those stats as well on the HUD.



---


There are moments in which it changes instance...why?
I want same instance every time
Also check that the fuzzy RL is learning something

---

Basically I would expect:

- Animation of the simulation: nodes are houses, the truck moves step by step + a smoother transition from one node to
  another
- A training class, just like simulation, that has a run_one_step method of some sort
- A simulation class that runs the simulation and is linked to the animation in some way
- Importantly, I want the training and simulation to run at the same time: the training periodically writes the new
  model and the simulation loads it
  The saved model should be specific for the size we want.
- I want the HUD to show how the training is behaving and if the model is improving.

Then, my interface should allow to have a menu to choose:

- The size
- Pre-trained or live training
  And then we run the simulation, step by step or automatically at a chosen speed

The bonus part would be to use Fuzzy RL to do the same: it also trains and has a simulation running in parallel.

- In this case, we should allow visually a comparison of some sort
- Plus I would want to have an interpretation of what is going on

I ALSO WANT BETTER RESULTS CAUSE IT DOES NOT LOOK SO GOOD. I want to penalize crossings, if that makes any sense.

More precisely:

- BatchVRPEnv is my new environment. I do not care about anything in /env anymore. I should however have a way to return
  the observation of the trainer (only the 1st instance in the batch).
- I want to have a Trainer class that handles what is now done in train.py, in particular the usage of BatchVRPEnv and
  the agent, using REINFORCE on the agent.
    - A single step of the Trainer is a full step from one node to another for all instances.
- I want to have a Visualization class that handles the same thing (a BatchVRPEnv and an agent), but:
    - BatchVRPAgent has a fixed batch_size of 1. It only has one instance
    - The agent is not trained, but only used
    - I want a sort of microstep method that is used to do the visualization, only moving very slightly the truck.
    - A single step of the Visualization is therefore sometimes a full step and other times just a movement to the
      truck's position.
- I want the trainer and visualization class to be linked by the fact that trainer writes the agent on a file and
  visualization can load an agent from a file.

> How does the BatchVRPEnv class adapt to the Fuzzy RL agent?
> Well first I need to decide what the Fuzzy RL agent actually sees. To make it somewhat of a fair comparison I should
> give similar data to the Fuzzy Q lambda agent that I give to the REINFORCE Agent.
> Obviously, I cannot have such a large state.
> Either way, I can imagine using the same info from the environment (node features, truck state, mask) to make the
> decision.
> An intermediate step would have to transform that representation to the fuzzy representation, then the rest is fairly
> easy.
> Again, I would expect a Trainer and a Visualization class, separatedly. In this case, the visualization class simply
> goes greedy without updating the Q values.
> I should write the REINFORCE versions and then translate all.

How would the interface work if we have the two agents?
A split screen would be cool I guess, but maybe too much work?
Actually, I should have then run on the same instances probably?
This requires modifications to the BatchVRPEnv class I believe.
But again, how do they even compare if the Fuzzy Q lambda learns from 1 instance while the other in a single step?

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