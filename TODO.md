## 5. Implementation Plan

### Phase 0 — Environment (start here)
**Goal:** A working episode loop that all three methods can run against.

- [X] `VRPInstance` dataclass: cluster generator, outlier placement, demand sampling, Euclidean cost matrix
- [X] `VRPEnvironment`: step function takes target node, accumulates Euclidean travel cost, handles depot returns
- [X] Lateness tracking: per-customer arrival time, window, penalty accumulation
- [X] TONN implementation: scoring uses Euclidean distance and urgency
- [X] Penalty weight calibration: run TONN distance-only vs urgency-only on 20 instances, find interesting α
- [X] Episode visualisation: pygame renderer with straight-line routes, urgency-coloured nodes, cost counter


### Phase 1 — Transformer Agent
**Goal:** Port and adapt the existing course project Transformer to this environment.

- [X] Node feature construction: (x, y, demand, urgency, visited) per customer + depot
- [X] Dynamic arrival handling: re-encode on new customer arrival (or mask with urgency update)
- [X] Decoder: pointer network over unvisited feasible customers, depot as always-available option
- [X] REINFORCE loop: same advantage signal as Fuzzy agent
- [X] Entropy regularisation: confirm it does not collapse early (lesson from course project)
- [X] Frozen learned baseline as secondary option if TONN advantage proves unstable
- [X] Validation: confirm convergence on 10-customer instances before scaling to 30
- [ ] Scaling to 50 customers.

### Phase 2 — Fuzzy Agent
**Goal:** A trainable fuzzy scoring agent that improves over TONN.

- [ ] `FuzzyMembership`: triangular MF class with learnable breakpoints, forward pass returns label→weight dict
- [ ] `FuzzyScorer`: takes candidate feature vector, returns priority score via rule base
- [ ] `FuzzyAgent`: wraps scorer, implements candidate selection (argmax over scores)
- [ ] REINFORCE loop: episode rollout, cost computation, TONN advantage, gradient update
- [ ] Rule weight update: gradient of expected advantage w.r.t. rule weights
- [ ] Membership breakpoint update: gradient of expected advantage w.r.t. breakpoints
- [ ] Training diagnostics: reward curve, rule firing frequency, membership function shape evolution
- [ ] Validation: confirm agent improves over random and approaches TONN within ~500 episodes

### Phase 3 — Evaluation & Ablation
**Goal:** Clean comparative results ready for report and presentation.

- [ ] Generate held-out test set of 200 instances, fix random seed
- [ ] Evaluate all three methods on test set, record C for each instance
- [ ] Compute mean ± std per method, statistical significance test (paired t-test vs TONN)
- [ ] Ablation: crisp Fuzzy agent (replace MFs with hard thresholds), same training budget
- [ ] Rule extraction: log top-10 rules fired during Fuzzy agent evaluation, format for report
- [ ] Learning curves: plot reward vs episode for both learned methods

### Phase 4 — Visualisation & Demo
**Goal:** A pygame demo that makes the qualitative differences between methods visible.

- [ ] Side-by-side or sequential playback of the same instance solved by each method
- [ ] Urgency colouring on nodes (green → yellow → red as window expires)
- [ ] Cluster outlines visible as faint background regions
- [ ] Truck path trace with straight lines between selected nodes, cost counter
- [ ] Fuzzy rule display panel: show active rules and their weights during playback

---

## 6. Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Fuzzy agent plateaus near TONN, gap is too small to be interesting | Confirm cluster density signal is active; check if MF breakpoints are actually updating |
| Transformer fails to converge on 50-node instances | Start training on 10-node, curriculum to 50; check entropy collapse from course project lessons |
| Lateness penalty swamps distance signal | Tune α carefully in Phase 0; if necessary, normalise both components separately |
| Dynamic arrivals make training unstable | Fix arrival schedule per instance during early training, randomise later |
| TONN advantage has high variance across geometrically diverse instances | Use instance-normalised advantage; optionally bucket by instance difficulty |

---

## 7. Documented Limitations

The following are known approximations that should be stated explicitly in the report rather than hidden:

- The Fuzzy agent scores candidates independently — it has no global view of the unvisited node set beyond the
  cluster density signal. The Transformer does not have this limitation.
- Membership function updates use the REINFORCE gradient, which is high-variance. The adaptive MF component may
  learn slowly and its contribution may be hard to isolate without a longer ablation.
- Travel times are deterministic Euclidean distances. Real-world congestion and variability are not modelled;
  this is a deliberate simplification to keep the comparison clean and training stable.
- The single-truck formulation removes inter-truck coordination complexity but also removes a dimension where
  fuzzy interpretability would be especially valuable (explaining coordination decisions).

---

## 8. What Success Looks Like

A successful project demonstrates:

1. Transformer achieves significantly lower mean cost than TONN on the held-out test set
2. Fuzzy agent achieves meaningfully lower mean cost than TONN, with a smaller gap to the Transformer
3. The ablation shows crisp Fuzzy performs worse than full Fuzzy, confirming the membership function contribution
4. The extracted rule base is readable and sensible — rules like "IF urgency high AND distance near → high priority"
   should dominate

The interesting finding is not "Transformer wins." The interesting finding is how much interpretability costs in
terms of performance, and whether the Fuzzy agent's rule base actually reflects good routing intuitions rather than
arbitrary patterns.