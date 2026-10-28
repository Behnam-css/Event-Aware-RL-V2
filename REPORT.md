# Event-aware reinforcement learning for long-run patch-foraging returns

**Status:** reproducible simulation pilot · **Action set:** `exploit` / `leave` · **Objective:** average reward per unit time

> **Interpretation guardrail.** This is a synthetic proof-of-concept study. The event-aware agent is given the true persistent ecological phase; it does not learn an event detector. The event-blind and event-free baselines use approximate latent-state aggregation. The “Zeraati-style” model adapts a state/action logic and average-reward planning approach; it is not an exact replication of Zeraati et al. (2025).

## Abstract

Patch foraging requires an agent to decide whether to continue exploiting a depleting resource or leave for another patch. Average-reward model-based reinforcement learning can exploit information about resource dynamics and travel time that is not captured by a simple fixed leaving threshold. We extend a three-patch, two-action setting with a stochastic threat-and-recovery process whose transitions are independent of immediate food outcome. We compare five policies: event-aware average-reward policy iteration, event-blind policy iteration with transient boundary cues, a Zeraati-style event-free policy-iteration baseline adapted to this task, an MVT-inspired threshold rule, and a fixed threshold tuned on separate training seeds. Across four environmental scenarios and 20 evaluation seeds per policy/scenario, event-aware PI has the highest mean reward rate in all scenarios. The advantage over event-blind PI ranges from 0.042 to 0.058 reward units per time unit; the advantage over the adapted event-free PI baseline ranges from 0.066 to 0.083. These results support the value of explicit ecological-phase information within this simulator. They do not establish that event perception is universally necessary, because the event-aware model receives an oracle phase label and the event-free models aggregate hidden phase with a fixed prior. A fairer next comparison should allow an event-free recurrent or belief-state model to infer the ongoing situation from the same observation history.

## 1. Motivation and theoretical background

### 1.1 Patch foraging and MVT

Classical optimal-foraging theory frames patch-leaving as a rate-maximization problem. In the Marginal Value Theorem, a forager leaves a patch when the marginal rate of return from staying falls below the average rate available in the environment, under the theorem's assumptions (Stephens & Krebs, 1986). This is a valuable benchmark, but its applicability depends on the environment's temporal and spatial structure. In environments with gradual replenishment and revisitable patches, an agent's decision can depend on the current state of unattended resources and their future evolution. Kendall and Wikenheiser (2022) review temporal-cognition issues related to patch residence and leaving.

### 1.2 Average-reward world-model planning

Zeraati et al. (2025) formulate sequential patch foraging as an average-reward MDP. Their actions are exploit and leave, and the state tracks the current patch and reward probabilities of three patches. They solve the model-based optimization problem with policy iteration and report regimes in which the optimal policy outperforms MVT and fixed-threshold policies when replenishment is slow. In the present project, we retain the same two-action framing and the average-reward policy-iteration style, while changing the state representation and adding a stochastic threat/recovery mechanism.

### 1.3 Event perception as a signal for situation-model change

Event Segmentation Theory proposes that event boundaries can emerge when predictions about near-future states fail and the current event model needs updating (Zacks et al., 2007; Zacks & Swallow, 2007). This suggests a computational hypothesis: event-related cues may carry information about a change in ecological situation that is not reducible to an immediate reward change. An explicit phase representation may let the policy map the same location/resource configuration to different actions in different situations.

This is a motivation for the model—not evidence that the current simulation implements or validates biological event perception. The event-aware agent is supplied with the true phase. The code does not infer event boundaries from raw observations.

## 2. Environment

### 2.1 Two actions

Every policy has exactly two actions:

- `exploit`: attempt to obtain food from the current patch; duration 1.00.
- `leave`: move to the next patch in the deterministic cycle A → B → C → A; duration 1.25.

Action duration enters the reward-rate objective. The fixed cycle means the agent cannot choose which neighboring patch to visit.

### 2.2 Resource levels and food reward

Three patches have maximum reward probabilities `pmax = [0.7, 1.0, 0.3]`. Each patch is represented by a discrete resource level: depleted (0), intermediate (1), or rich (2), with multipliers `[0.18, 0.55, 1.0]`. For current patch `i`,

`p(food | state) = clip(pmax[i] * resource_factor[resource_i], 0.01, 0.98)`.

During exploitation, a successful food outcome causes a one-level resource drop with probability 0.72; a failed attempt causes a drop with probability 0.12, when the patch is not depleted. Other patches can replenish during exploitation. When leaving, all patches below their maximum level may replenish. The action-duration-adjusted regeneration probability is `1 - (1 - replenish_rate) ** duration`.

### 2.3 Ecological phase

The latent ecological phase has five values:

1. `FORAGE`: ordinary foraging; a threat may arise with duration-adjusted hazard probability.
2. `THREAT_ONSET`: the threat boundary; transition to recovery onset is more likely after leaving (0.48) than exploiting (0.08).
3. `THREAT_ONGOING`: continuing threat; transition to recovery onset is more likely after leaving (0.52) than exploiting (0.05).
4. `RECOVERY_ONSET`: one-step marker that recovery has begun, then transitions to `RECOVERY`.
5. `RECOVERY`: returns toward foraging (0.76 per transition), can persist during recovery, and can transition to a new threat according to a small hazard-dependent probability.

Threat onset is sampled independently of immediate food outcome. During threat onset and ongoing threat, exploitation receives a hand-set cost of 0.85. During recovery onset and recovery, exploitation receives a hand-set cost of 0.06. Leaving has zero immediate reward. These costs create a decision-relevant difference between phases; they are illustrative simulator parameters, not empirical estimates.

### 2.4 Scenarios

| Scenario | Threat hazard | Replenishment rate |
|---|---:|---:|
| Reference | 0.08 | 0.25 |
| High threat | 0.18 | 0.25 |
| Slow replenishment | 0.08 | 0.10 |
| High threat + slow replenishment | 0.18 | 0.10 |

## 3. MDP and observation maps

Let the latent simulator state be `x = (location, resource_A, resource_B, resource_C, phase)`. All models act on the same simulator and have the same action set `{exploit, leave}`.

### 3.1 Event-aware model

`S_aware = (location, r_A, r_B, r_C, phase)` with `3 × 3³ × 5 = 405` states. The phase is explicit and persistent, so identical location/resource values in `FORAGE` and `THREAT_ONGOING` remain different states.

### 3.2 Event-blind model

`O_blind = (location, r_A, r_B, r_C, cue)` with `3 × 3³ × 3 = 243` observations. `cue = +1` at threat onset, `cue = -1` at recovery onset, and `cue = 0` otherwise. The cue is transient; the model does not explicitly retain `THREAT_ONGOING` after onset cue disappears.

### 3.3 Zeraati-style event-free adaptation

`O_free = (location, r_A, r_B, r_C)` with `3 × 3³ = 81` observations. Neither persistent phase nor transient boundary cue is included. This baseline shares the broad event-free state logic and two-action average-reward planning style of the source paper, but it is not the source paper's original model. It uses this experiment's resource discretization and latent threat/recovery dynamics.

### 3.4 Aggregation caveat

The event-blind and event-free transition models are built by aggregating latent states that map to the same observation. Compatible phases are mixed with a scenario-specific stationary phase prior computed under a neutral 50:50 exploit/leave action mixture. The model then averages latent rewards, transitions and durations. Because latent phase affects future outcomes, the aggregate observation process need not be Markov exactly. The phase prior is not learned online or conditioned on the policy being evaluated. Thus these baselines approximate an observation-level MDP rather than solving the exact POMDP/belief-state problem.

## 4. Average-reward policy iteration

For policy `π`, let `Rπ(s)` be expected immediate reward, `Tπ(s)` be expected action duration, `Pπ(s,s′)` be the transition matrix under the policy, `ρπ` be average reward per unit time, and `hπ(s)` be relative state value (bias). Policy evaluation solves

`hπ(s) = Rπ(s) - ρπ Tπ(s) + Σ_s′ Pπ(s,s′) hπ(s′)`.

Policy improvement computes

`Qπ(s,a) = R(s,a) - ρπ T(s,a) + Σ_s′ P(s′|s,a) hπ(s′)`

and selects `π_new(s) = argmax_a Qπ(s,a)` over the two actions. A reference state's relative value is anchored to zero, yielding a linear system for policy evaluation. The code starts from exploit-first greedy actions, applies an epsilon-soft action distribution with ε = 0.01, and stops when a policy-improvement pass changes zero greedy actions (maximum 100 passes). The transition, reward and duration models are supplied; the solver is model-based planning rather than model-free RL or online world-model learning.

## 5. Baselines

1. **Event-aware PI:** full 405-state MDP; true persistent phase label is available.
2. **Event-blind PI:** 243-observation aggregated model; transient boundary cue is available, ongoing phase is not retained.
3. **Zeraati-style event-free PI:** 81-observation aggregated model; no phase/cue in the state.
4. **MVT-inspired threshold:** rule-based leave threshold based on the event-free PI's estimated average-reward rate clipped to [0.05, 0.80]. This is an adaptation, not the theorem evaluated under its canonical assumptions.
5. **Best fixed threshold:** threshold selected from 0.05–0.80 in increments of 0.05 using separate training seeds, then held fixed for evaluation within a scenario.

## 6. Experimental protocol and metrics

- 20 evaluation seeds per policy and scenario.
- 3,000 decision steps per seed; first 200 warm-up steps excluded.
- Fixed-threshold tuning: 5 training seeds, 900 steps, 100 warm-up steps per threshold.
- Four scenarios in a 2 × 2 design crossing hazard rate and replenishment rate.
- Primary measure: total reward divided by total elapsed action time after warm-up.
- Secondary measures: food per 100 actions, leave/exploit fractions, action fractions by latent phase, policy-iteration passes, and paired reward-rate differences.
- Approximate 95% confidence intervals use mean ± 1.96 × standard error across seeds. These are descriptive pilot intervals. No multiple-comparison correction is applied; the report does not claim a confirmatory statistical test.

## 7. Results

Event-aware PI has the highest mean reward rate in all four scenarios.

| Scenario | Event-aware PI | Event-blind PI | Event-free PI | MVT-inspired | Fixed threshold |
|---|---:|---:|---:|---:|---:|
| Reference | 0.3250 | 0.2741 | 0.2548 | 0.1802 | 0.2600 |
| High threat | 0.2929 | 0.2350 | 0.2150 | 0.0845 | 0.2156 |
| Slow replenishment | 0.2348 | 0.1926 | 0.1689 | −0.3062 | 0.1620 |
| High threat + slow replenishment | 0.2177 | 0.1655 | 0.1349 | −0.4959 | 0.1334 |

### Paired advantages

| Scenario | Event-aware − event-blind | Event-aware − event-free PI |
|---|---:|---:|
| Reference | +0.0509 | +0.0702 |
| High threat | +0.0579 | +0.0779 |
| Slow replenishment | +0.0422 | +0.0658 |
| High threat + slow replenishment | +0.0522 | +0.0828 |

The approximate paired 95% intervals are above zero for these contrasts in these runs. This is consistent with the designed benefit of knowing whether a threat is ongoing. However, the event-aware model's oracle label and the aggregate baselines' approximation mean that the comparisons do not isolate a learned event-perception mechanism.

The MVT-inspired baseline's negative reward rates under slow replenishment arise from the risk costs and the specific threshold rule in this simulation. This result should not be treated as an indictment of the classical MVT, because the implemented rule is not a faithful implementation under canonical assumptions.

### Policy-iteration convergence

| Scenario | Event-aware passes | Event-blind passes | Event-free passes |
|---|---:|---:|---:|
| Reference | 7 | 8 | 7 |
| High threat | 8 | 7 | 6 |
| Slow replenishment | 8 | 7 | 7 |
| High threat + slow replenishment | 8 | 7 | 6 |

Convergence means the greedy action map did not change during policy improvement. The supplied Zeraati abstract does not report its iteration count, so we cannot compare the number of passes to the original implementation.

## 8. Limitations and next steps

1. **Oracle phase label:** event-aware PI is not an event detector or learned event model.
2. **Approximate aggregation:** event-blind and event-free baselines use fixed phase priors, not a learned recurrent state or exact Bayesian filter.
3. **Not a Zeraati replication:** environment, transition dynamics, reward costs, discretization and state aggregation differ.
4. **Hand-set threat costs:** parameter values should be subjected to sensitivity analysis.
5. **Fixed cyclic travel:** the agent cannot choose any patch; leaving always follows A→B→C→A.
6. **Known dynamics:** all three PI solvers use a specified model; the code does not learn a world model from experience.
7. **Pilot size:** 20 seeds and approximate intervals are not a substitute for confirmatory statistics or independent validation.
8. **MVT approximation:** the implemented MVT-inspired threshold is a heuristic baseline, not the original theorem under its assumptions.

### Priority follow-ups

- Add a history-sensitive event-free world model with the same raw observations and interaction budget.
- Learn rather than supply the phase labels; introduce missed detections and false alarms.
- Sweep threat hazard, replenishment rate, risk penalty and event-cue reliability; report interaction effects.
- Evaluate on held-out environment parameters and phase-transition rules.
- Compare policies on patch-residence distributions and event-boundary-aligned behavior.
- Use paired bootstrap/permutation tests and larger seed counts for confirmatory claims.

## 9. References

- Zeraati, R., Oña Jodar, T., Safavi, S., Cruz, B., Poo, C., & Dayan, P. (2025). *Optimal Foraging by Learning the World Model*. CCN 2025 conference abstract. [PDF](https://2025.ccneuro.org/abstract_pdf/Zeraati_2025_Optimal_Foraging_Learning_World_Model.pdf)
- Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press. [DOI](https://doi.org/10.2307/j.ctvs32s6b)
- Mahadevan, S. (1996). Average reward reinforcement learning: Foundations, algorithms, and empirical results. *Machine Learning, 22*, 159–195. [DOI](https://doi.org/10.1007/BF00114727)
- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., & Reynolds, J. R. (2007). Event perception: A mind–brain perspective. *Psychological Bulletin, 133*(2), 273–293. [DOI](https://doi.org/10.1037/0033-2909.133.2.273)
- Zacks, J. M., & Swallow, K. M. (2007). Event segmentation. *Current Directions in Psychological Science, 16*(2), 80–84. [DOI](https://doi.org/10.1111/j.1467-8721.2007.00480.x)
- Kendall, R. K., & Wikenheiser, A. M. (2022). Quitting while you're ahead: Patch foraging and temporal cognition. *Behavioral Neuroscience, 136*(5), 467–478. [DOI](https://doi.org/10.1037/bne0000526)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Book page](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)

## Project files

- `index.html`: full GitHub Pages report website.
- `code/event_aware_two_action_foraging.py`: runnable simulation.
- `notebooks/Event_Aware_Two_Action_Foraging_RL.ipynb`: Colab notebook.
- `data/*.csv`: all analysis tables.
- `assets/figures/`: PNG and PDF figures.
- `downloads/event_aware_two_action_results.zip`: standalone results bundle.
- Source conference abstract: Zeraati et al. (2025), linked in the References section. The PDF is linked externally rather than redistributed in this package.
