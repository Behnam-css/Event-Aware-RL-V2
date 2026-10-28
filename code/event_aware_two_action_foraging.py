"""Two-action event-aware foraging experiment with average-reward policy iteration.

Purpose
-------
Compare event-aware and event-blind average-reward MDP policies while holding
the action set fixed to the two actions used in Zeraati et al. (2025): exploit
and leave. Include a Zeraati-style event-free policy-iteration baseline adapted
to this task, plus two threshold baselines. This is a new, simplified extension
inspired by the source paper, not a replication of its exact task or results.
"""
from __future__ import annotations
import argparse, itertools, json, math, os, zipfile, textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Configuration ---------------------------------
PHASES = ["FORAGE", "THREAT_ONSET", "THREAT_ONGOING", "RECOVERY_ONSET", "RECOVERY"]
FORAGE, THREAT_ONSET, THREAT_ONGOING, RECOVERY_ONSET, RECOVERY = range(5)
ACTIONS = ["exploit", "leave"]
EXPLOIT, LEAVE = range(2)
RESOURCE_LEVELS = (0, 1, 2)  # depleted, intermediate, rich
RESOURCE_FACTOR = np.array([0.18, 0.55, 1.0], dtype=float)
PATCH_PMAX = np.array([0.7, 1.0, 0.3], dtype=float)
PATCH_NAMES = ["A", "B", "C"]
ACTION_DURATION = np.array([1.0, 1.25], dtype=float)  # leave includes travel time

SCENARIOS = [
    {"name": "Reference", "hazard_rate": 0.08, "replenish_rate": 0.25},
    {"name": "High threat", "hazard_rate": 0.18, "replenish_rate": 0.25},
    {"name": "Slow replenishment", "hazard_rate": 0.08, "replenish_rate": 0.10},
    {"name": "High threat + slow replenishment", "hazard_rate": 0.18, "replenish_rate": 0.10},
]

@dataclass
class Config:
    output_dir: str = "event_aware_two_action_results"
    n_seeds: int = 20
    horizon: int = 3000
    warmup: int = 200
    threshold_train_seeds: int = 5
    threshold_train_horizon: int = 900
    threshold_train_warmup: int = 100
    epsilon_soft: float = 0.01
    max_policy_iterations: int = 100
    random_seed: int = 20261028
    verbose: bool = True

@dataclass
class Model:
    states: list
    state_idx: dict
    observations: list
    obs_idx: dict
    P: np.ndarray
    R: np.ndarray
    T: np.ndarray
    outcomes: list
    P_blind: np.ndarray
    R_blind: np.ndarray
    T_blind: np.ndarray
    base_observations: list
    base_obs_idx: dict
    P_zeraati: np.ndarray
    R_zeraati: np.ndarray
    T_zeraati: np.ndarray
    config: dict


def phase_cue(phase: int) -> int:
    if phase == THREAT_ONSET: return 1
    if phase == RECOVERY_ONSET: return -1
    return 0


def food_prob(loc: int, resources: Tuple[int, int, int]) -> float:
    return float(np.clip(PATCH_PMAX[loc] * RESOURCE_FACTOR[resources[loc]], 0.01, 0.98))


def duration_adjusted_probability(rate: float, duration: float) -> float:
    return float(1.0 - (1.0 - rate) ** duration)


def resource_update_outcomes(resources, loc, action, replenish_rate, duration, success=None):
    """Enumerate outcome branches (probability, new resources, food_success).

    Exploiting depletes the current patch with outcome-dependent probability;
    other patches can replenish during exploitation. Leaving permits all patches
    to replenish. Replenishment probability scales with action duration.
    """
    branches = []
    if action == EXPLOIT:
        pfood = food_prob(loc, resources)
        success_branches = [(1, pfood), (0, 1.0-pfood)]
        for food, pfood_branch in success_branches:
            pdrop = 0.72 if food else 0.12
            if resources[loc] == 0:
                drop_branches = [(0, 1.0)]
            else:
                drop_branches = [(1, pdrop), (0, 1.0-pdrop)]
            for drop, pdrop_branch in drop_branches:
                after = list(resources)
                after[loc] = max(0, after[loc] - drop)
                eligible = [i for i in range(3) if i != loc]
                regen_rate = duration_adjusted_probability(replenish_rate, duration)
                variable = [i for i in eligible if after[i] < 2]
                regen_combos = itertools.product((0,1), repeat=len(variable))
                for bits in regen_combos:
                    next_res = after.copy(); preg = 1.0
                    for i, bit in zip(variable, bits):
                        next_res[i] += bit
                        preg *= regen_rate if bit else (1.0-regen_rate)
                    prob = pfood_branch * pdrop_branch * preg
                    if prob > 0:
                        branches.append((prob, tuple(next_res), int(food)))
    else:
        after = list(resources)
        eligible = [i for i in range(3) if after[i] < 2]
        regen_rate = duration_adjusted_probability(replenish_rate, duration)
        for bits in itertools.product((0,1), repeat=len(eligible)):
            next_res = after.copy(); preg = 1.0
            for i, bit in zip(eligible, bits):
                next_res[i] += bit
                preg *= regen_rate if bit else (1.0-regen_rate)
            branches.append((preg, tuple(next_res), 0))
    merged = defaultdict(float)
    for p, r, f in branches: merged[(r,f)] += p
    return [(p,r,f) for (r,f),p in merged.items() if p > 0]


def phase_transition_outcomes(phase: int, action: int, hazard_rate: float, duration: float):
    """Ecological transition; threat onset is generated independently of food."""
    if phase == FORAGE:
        ph = duration_adjusted_probability(hazard_rate, duration)
        return [(1-ph, FORAGE), (ph, THREAT_ONSET)]
    if phase == THREAT_ONSET:
        # Leaving the threatened patch is more likely than exploiting to end the event.
        release = 0.08 if action == EXPLOIT else 0.48
        return [(release, RECOVERY_ONSET), (1-release, THREAT_ONGOING)]
    if phase == THREAT_ONGOING:
        release = 0.05 if action == EXPLOIT else 0.52
        return [(release, RECOVERY_ONSET), (1-release, THREAT_ONGOING)]
    if phase == RECOVERY_ONSET:
        return [(1.0, RECOVERY)]
    # Recovery eventually returns to ordinary foraging; new events can arise.
    p_new = min(0.25 * hazard_rate, 0.04)
    return [(0.76, FORAGE), (0.24-p_new, RECOVERY), (p_new, THREAT_ONSET)]


def reward_for_outcome(phase: int, action: int, food_success: int) -> float:
    """Unit food reward minus an explicit threat-related cost.

    Threat costs are illustrative task parameters. They make event recognition
    behaviorally relevant without altering the action set.
    """
    reward = float(food_success) if action == EXPLOIT else 0.0
    if action == EXPLOIT:
        if phase in (THREAT_ONSET, THREAT_ONGOING): reward -= 0.85
        elif phase in (RECOVERY_ONSET, RECOVERY): reward -= 0.06
    return reward


def build_model(hazard_rate: float, replenish_rate: float, verbose=False) -> Model:
    res_tuples = list(itertools.product(RESOURCE_LEVELS, repeat=3))
    states = [(loc, res, phase) for loc in range(3) for res in res_tuples for phase in range(5)]
    state_idx = {s:i for i,s in enumerate(states)}
    observations = [(loc, res, cue) for loc in range(3) for res in res_tuples for cue in (-1,0,1)]
    obs_idx = {o:i for i,o in enumerate(observations)}
    # Zeraati-style event-free state: current patch + resource states only.
    # This is an adaptation of the source paper's state representation to the
    # present discretized foraging/threat environment, not an exact replication.
    base_observations = [(loc, res) for loc in range(3) for res in res_tuples]
    base_obs_idx = {o:i for i,o in enumerate(base_observations)}
    ns, no, nz = len(states), len(observations), len(base_observations)
    P = np.zeros((2,ns,ns), dtype=float)
    R = np.zeros((2,ns), dtype=float)
    T = np.repeat(ACTION_DURATION[:,None], ns, axis=1)
    outcomes = [[None for _ in range(ns)] for _ in range(2)]

    for si,(loc,res,phase) in enumerate(states):
        for a in range(2):
            next_loc = loc if a == EXPLOIT else (loc+1)%3  # sequential patch order
            all_outcomes=[]
            res_out = resource_update_outcomes(res,loc,a,replenish_rate,ACTION_DURATION[a])
            phase_out = phase_transition_outcomes(phase,a,hazard_rate,ACTION_DURATION[a])
            for pr,nres,food in res_out:
                immediate = reward_for_outcome(phase,a,food)
                for pe,nphase in phase_out:
                    p = pr*pe
                    if p <= 0: continue
                    ni = state_idx[(next_loc,nres,nphase)]
                    P[a,si,ni] += p
                    R[a,si] += p*immediate
                    all_outcomes.append((p,ni,immediate,float(food)))
            total=sum(x[0] for x in all_outcomes)
            if not np.isclose(total,1.0,atol=1e-8):
                raise RuntimeError(f"Outcome sum {total} != 1, state={states[si]}, action={ACTIONS[a]}")
            outcomes[a][si]=[(p/total,ni,r,f) for p,ni,r,f in all_outcomes]

    # Compute a stationary prior over ecological phases under a neutral 50:50
    # exploit/leave action mixture. Baselines know scenario-level event statistics
    # but do not directly observe the current phase. This prior is an explicit
    # approximation, not a learned or policy-conditioned belief state.
    Qphase=np.zeros((5,5),dtype=float)
    for phase in range(5):
        for a in range(2):
            for pp,nphase in phase_transition_outcomes(phase,a,hazard_rate,ACTION_DURATION[a]):
                Qphase[phase,nphase] += 0.5*pp
    phase_prior=np.full(5,1.0/5.0)
    for _ in range(10000):
        updated=phase_prior@Qphase
        if np.max(np.abs(updated-phase_prior)) < 1e-14:
            phase_prior=updated
            break
        phase_prior=updated
    phase_prior=phase_prior/phase_prior.sum()

    # Event-blind model: retain transient boundary cues but aggregate persistent
    # phases that yield the same current observation. Within each cue class, use
    # the stationary prior conditional on that cue.
    B = np.zeros((ns,no)); W=np.zeros((no,ns)); compatible=defaultdict(list)
    for si,(loc,res,phase) in enumerate(states):
        oi=obs_idx[(loc,res,phase_cue(phase))]
        B[si,oi]=1.0; compatible[oi].append(si)
    for oi,latent in compatible.items():
        denom=sum(phase_prior[states[si][2]] for si in latent)
        for si in latent: W[oi,si]=phase_prior[states[si][2]]/denom
    Pblind=np.zeros((2,no,no)); Rblind=np.zeros((2,no)); Tblind=np.zeros((2,no))
    for a in range(2):
        Pblind[a]=W@P[a]@B
        Rblind[a]=W@R[a]
        Tblind[a]=W@T[a]

    # Zeraati-style event-free aggregation: only (location, resources) is observed.
    # Mix over phase using the same scenario-specific stationary prior, without
    # exposing event phase or boundary cues to this agent.
    Bz=np.zeros((ns,nz)); Wz=np.zeros((nz,ns)); compatible_z=defaultdict(list)
    for si,(loc,res,phase) in enumerate(states):
        oi=base_obs_idx[(loc,res)]
        Bz[si,oi]=1.0; compatible_z[oi].append(si)
    for oi,latent in compatible_z.items():
        denom=sum(phase_prior[states[si][2]] for si in latent)
        for si in latent: Wz[oi,si]=phase_prior[states[si][2]]/denom
    Pz=np.zeros((2,nz,nz)); Rz=np.zeros((2,nz)); Tz=np.zeros((2,nz))
    for a in range(2):
        Pz[a]=Wz@P[a]@Bz
        Rz[a]=Wz@R[a]
        Tz[a]=Wz@T[a]

    if not np.allclose(P.sum(axis=2),1,atol=1e-8): raise RuntimeError("Bad full transition matrix")
    if not np.allclose(Pblind.sum(axis=2),1,atol=1e-8): raise RuntimeError("Bad blind transition matrix")
    if not np.allclose(Pz.sum(axis=2),1,atol=1e-8): raise RuntimeError("Bad event-free transition matrix")
    if verbose: print(f"Built MDPs: event-aware={ns} states, event-blind={no} observations, Zeraati-style event-free={nz} observations, actions={ACTIONS}")
    return Model(states,state_idx,observations,obs_idx,P,R,T,outcomes,Pblind,Rblind,Tblind,
                 base_observations,base_obs_idx,Pz,Rz,Tz,
                 {"hazard_rate":hazard_rate,"replenish_rate":replenish_rate,
                  "stationary_phase_prior":phase_prior.tolist(),"phase_prior_action_mixture":"50% exploit / 50% leave"})

# ----------------------- Average-reward policy iteration ------------------
def evaluate_policy(P,R,T,policy_probs):
    ns=P.shape[1]
    Ppi=np.einsum('sa,asj->sj',policy_probs,P,optimize=True)
    Rpi=np.einsum('sa,as->s',policy_probs,R,optimize=True)
    Tpi=np.einsum('sa,as->s',policy_probs,T,optimize=True)
    # x=[rho,h(1)...h(ns-1)] with h(0)=0; rho*T - P*h + h = R.
    A=np.zeros((ns,ns)); A[:,0]=Tpi; A[:,1:]=-Ppi[:,1:]
    A[1:,1:]+=np.eye(ns-1)
    try: x=np.linalg.solve(A,Rpi)
    except np.linalg.LinAlgError: x=np.linalg.lstsq(A,Rpi,rcond=1e-10)[0]
    rho=float(x[0]); h=np.zeros(ns); h[1:]=x[1:]
    return rho,h


def soft_policy(greedy, epsilon):
    n=len(greedy); probs=np.full((n,2),epsilon/2.0)
    probs[np.arange(n),greedy]+=1.0-epsilon
    return probs


def policy_iteration(P,R,T,epsilon=0.01,max_iter=100,name="model"):
    ns=P.shape[1]
    # Initialize to exploit, which provides a reproducible neutral starting point.
    greedy=np.zeros(ns,dtype=int)
    history=[]
    converged=False
    for it in range(1,max_iter+1):
        policy=soft_policy(greedy,epsilon)
        rho,h=evaluate_policy(P,R,T,policy)
        Q=R-rho*T+np.einsum('asj,j->as',P,h,optimize=True)
        new_greedy=np.argmax(Q,axis=0)
        changed=int(np.sum(new_greedy!=greedy))
        history.append({"model":name,"iteration":it,"average_reward_rate_estimate":rho,"changed_states":changed})
        if changed==0:
            converged=True; greedy=new_greedy; break
        greedy=new_greedy
    final_policy=soft_policy(greedy,epsilon)
    rho,h=evaluate_policy(P,R,T,final_policy)
    return {"greedy_actions":greedy,"policy_probs":final_policy,"average_reward_rate_model":rho,
            "relative_values":h,"iterations":it,"converged":converged,"history":history}

# ------------------------------ Rollouts -----------------------------------
def choose_action(policy_probs, state_index, rng):
    return int(rng.choice(2,p=policy_probs[state_index]))


def rollout(model, policy, policy_kind, seed, horizon, warmup, threshold=None, collect_trace=False):
    rng=np.random.default_rng(seed)
    # Start patch A, all patches rich, ordinary forage.
    state=model.state_idx[(0,(2,2,2),FORAGE)]
    total_reward=0.0; total_time=0.0; n_food=0; n_leave=0; n_exploit=0
    phase_action=defaultdict(lambda: defaultdict(int))
    trace=[]; cumulative=[]; reward_acc=0.0; time_acc=0.0
    for t in range(horizon):
        loc,res,phase=model.states[state]
        cue=phase_cue(phase)
        if policy_kind=="event_aware_pi":
            a=choose_action(policy["policy_probs"],state,rng)
        elif policy_kind=="event_blind_pi":
            oi=model.obs_idx[(loc,res,cue)]
            a=choose_action(policy["policy_probs"],oi,rng)
        elif policy_kind=="zeraati_style_pi":
            oi=model.base_obs_idx[(loc,res)]
            a=choose_action(policy["policy_probs"],oi,rng)
        elif policy_kind in ("mvt_threshold","fixed_threshold"):
            p=food_prob(loc,res)
            a=EXPLOIT if p>=float(threshold) else LEAVE
        else: raise ValueError(policy_kind)
        outcomes=model.outcomes[a][state]
        probs=np.array([x[0] for x in outcomes],dtype=float); probs/=probs.sum()
        oi=int(rng.choice(len(outcomes),p=probs))
        _,next_state,reward,food=outcomes[oi]
        dur=float(ACTION_DURATION[a])
        if t>=warmup:
            total_reward += reward; total_time += dur
            n_food += int(food); n_leave += int(a==LEAVE); n_exploit += int(a==EXPLOIT)
            phase_action[PHASES[phase]][ACTIONS[a]] += 1
            reward_acc += reward; time_acc += dur
            if collect_trace and ((t-warmup+1)%50==0):
                cumulative.append({"step":t-warmup+1,"cumulative_reward_rate":reward_acc/max(time_acc,1e-12)})
        if collect_trace and t>=warmup:
            trace.append({"step":t-warmup,"location":PATCH_NAMES[loc],"phase":PHASES[phase],"cue":cue,
                          "resource_A":res[0],"resource_B":res[1],"resource_C":res[2],
                          "action":ACTIONS[a],"reward":reward,"food_success":food,"duration":dur})
        state=int(next_state)
    return {"reward_rate":total_reward/max(total_time,1e-12),"total_reward":total_reward,"total_time":total_time,
            "food_per_100_actions":100*n_food/max(horizon-warmup,1),"leave_fraction":n_leave/max(horizon-warmup,1),
            "exploit_fraction":n_exploit/max(horizon-warmup,1),"phase_action_counts":phase_action,
            "trace":trace,"cumulative":cumulative}


def tune_fixed_threshold(model, cfg, scenario_index):
    thresholds=np.round(np.arange(0.05,0.801,0.05),3)
    rows=[]
    for th_i,th in enumerate(thresholds):
        vals=[]
        for seed_i in range(cfg.threshold_train_seeds):
            seed=cfg.random_seed+100000+scenario_index*10000+th_i*100+seed_i
            out=rollout(model,None,"fixed_threshold",seed,cfg.threshold_train_horizon,cfg.threshold_train_warmup,threshold=th)
            vals.append(out["reward_rate"])
        rows.append({"threshold":th,"mean_training_reward_rate":float(np.mean(vals)),"sd_training_reward_rate":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0})
    df=pd.DataFrame(rows)
    best=df.loc[df.mean_training_reward_rate.idxmax()]
    return float(best.threshold),df

# ------------------------------- Analysis ----------------------------------
def mean_ci(values):
    x=np.asarray(values,dtype=float); n=len(x); m=float(np.mean(x)); sd=float(np.std(x,ddof=1)) if n>1 else 0.0
    # t critical approximation 1.96; clearly label as approximate.
    half=1.96*sd/math.sqrt(n) if n>1 else 0.0
    return m,sd,m-half,m+half


def run_experiment(cfg: Config):
    outdir=Path(cfg.output_dir); figdir=outdir/"figures"; outdir.mkdir(parents=True,exist_ok=True); figdir.mkdir(exist_ok=True)
    all_metrics=[]; histories=[]; thresholds_all=[]; action_rows=[]; traces_saved={}; phase_prior_rows=[]
    for sidx,sc in enumerate(SCENARIOS):
        if cfg.verbose: print(f"\nScenario: {sc['name']}")
        model=build_model(sc["hazard_rate"],sc["replenish_rate"],verbose=cfg.verbose)
        phase_prior_rows.extend([{"scenario":sc["name"],"phase":phase,"stationary_phase_prior":weight,
                                  "prior_action_mixture":"50% exploit / 50% leave"}
                                 for phase,weight in zip(PHASES,model.config["stationary_phase_prior"])])
        aware=policy_iteration(model.P,model.R,model.T,cfg.epsilon_soft,cfg.max_policy_iterations,"event_aware_pi")
        blind=policy_iteration(model.P_blind,model.R_blind,model.T_blind,cfg.epsilon_soft,cfg.max_policy_iterations,"event_blind_pi")
        zeraati=policy_iteration(model.P_zeraati,model.R_zeraati,model.T_zeraati,cfg.epsilon_soft,cfg.max_policy_iterations,"zeraati_style_pi")
        histories.extend([{**r,"scenario":sc["name"]} for r in aware["history"]])
        histories.extend([{**r,"scenario":sc["name"]} for r in blind["history"]])
        histories.extend([{**r,"scenario":sc["name"]} for r in zeraati["history"]])
        # MVT-style threshold uses the event-free model's estimated average reward rate.
        mvt_th=float(np.clip(zeraati["average_reward_rate_model"],0.05,0.80))
        fixed_th, tune_df=tune_fixed_threshold(model,cfg,sidx)
        tune_df["scenario"]=sc["name"]; thresholds_all.append(tune_df)
        if cfg.verbose:
            print(f"  PI aware rho={aware['average_reward_rate_model']:.4f}; iterations={aware['iterations']}; converged={aware['converged']}")
            print(f"  PI blind rho={blind['average_reward_rate_model']:.4f}; iterations={blind['iterations']}; converged={blind['converged']}")
            print(f"  Zeraati-style event-free PI rho={zeraati['average_reward_rate_model']:.4f}; iterations={zeraati['iterations']}; converged={zeraati['converged']}")
            print(f"  MVT-inspired threshold={mvt_th:.3f}; tuned fixed threshold={fixed_th:.2f}")
        models=[("Event-aware PI","event_aware_pi",aware,None),
                ("Event-blind PI","event_blind_pi",blind,None),
                ("Zeraati-style event-free PI","zeraati_style_pi",zeraati,None),
                ("MVT-inspired threshold","mvt_threshold",None,mvt_th),
                ("Best fixed threshold","fixed_threshold",None,fixed_th)]
        scenario_metrics={}
        for model_name,kind,pi,th in models:
            for seed_i in range(cfg.n_seeds):
                seed=cfg.random_seed+sidx*100000+seed_i*1000+17
                res=rollout(model,pi,kind,seed,cfg.horizon,cfg.warmup,threshold=th,collect_trace=(seed_i==0 and sidx==0))
                row={"scenario":sc["name"],"model":model_name,"seed":seed_i,"seed_value":seed,
                     "reward_rate":res["reward_rate"],"total_reward":res["total_reward"],"total_time":res["total_time"],
                     "food_per_100_actions":res["food_per_100_actions"],"leave_fraction":res["leave_fraction"],
                     "exploit_fraction":res["exploit_fraction"],"threshold":th,
                     "iterations":pi["iterations"] if pi else np.nan,"converged":pi["converged"] if pi else np.nan,
                     "model_reward_rate_estimate":pi["average_reward_rate_model"] if pi else np.nan}
                all_metrics.append(row); scenario_metrics.setdefault(model_name,[]).append(res["reward_rate"])
                for phase,counts in res["phase_action_counts"].items():
                    total=sum(counts.values())
                    for act in ACTIONS:
                        action_rows.append({"scenario":sc["name"],"model":model_name,"seed":seed_i,"phase":phase,
                                            "action":act,"count":counts.get(act,0),"fraction":counts.get(act,0)/max(total,1)})
                if seed_i==0 and sidx==0 and res["trace"]:
                    traces_saved[model_name]=res["trace"]
        # Save paired event-aware vs event-blind seed differences using matched seed values.
        if cfg.verbose:
            for m,vals in scenario_metrics.items(): print(f"    {m}: {np.mean(vals):.4f} ± {np.std(vals,ddof=1):.4f}")

    metrics=pd.DataFrame(all_metrics)
    metrics.to_csv(outdir/"metrics_by_seed.csv",index=False)
    pd.DataFrame(histories).to_csv(outdir/"policy_iteration_history.csv",index=False)
    pd.DataFrame(phase_prior_rows).to_csv(outdir/"scenario_phase_priors.csv",index=False)
    pd.concat(thresholds_all,ignore_index=True).to_csv(outdir/"threshold_tuning.csv",index=False)
    actions_df=pd.DataFrame(action_rows); actions_df.to_csv(outdir/"action_fractions_by_phase.csv",index=False)
    # Summary and paired differences
    summary_rows=[]
    for (scenario,model_name),g in metrics.groupby(["scenario","model"],sort=False):
        m,sd,lo,hi=mean_ci(g.reward_rate)
        summary_rows.append({"scenario":scenario,"model":model_name,"n_seeds":len(g),"mean_reward_rate":m,
                             "sd_reward_rate":sd,"ci95_approx_low":lo,"ci95_approx_high":hi,
                             "mean_food_per_100_actions":g.food_per_100_actions.mean(),"mean_leave_fraction":g.leave_fraction.mean(),
                             "mean_exploit_fraction":g.exploit_fraction.mean(),"policy_iterations":g.iterations.dropna().iloc[0] if g.iterations.notna().any() else np.nan,
                             "model_reward_rate_estimate":g.model_reward_rate_estimate.dropna().iloc[0] if g.model_reward_rate_estimate.notna().any() else np.nan})
    summary=pd.DataFrame(summary_rows); summary.to_csv(outdir/"summary_metrics.csv",index=False)
    diff_rows=[]
    for sc in [x["name"] for x in SCENARIOS]:
        sub=metrics[metrics.scenario==sc]
        for baseline in ["Event-blind PI", "Zeraati-style event-free PI", "MVT-inspired threshold", "Best fixed threshold"]:
            a=sub[sub.model=="Event-aware PI"].sort_values("seed").reward_rate.to_numpy()
            b=sub[sub.model==baseline].sort_values("seed").reward_rate.to_numpy()
            dif=a-b; m,sd,lo,hi=mean_ci(dif)
            diff_rows.append({"scenario":sc,"comparison":f"Event-aware PI minus {baseline}","baseline":baseline,
                              "mean_difference":m,"sd_paired_difference":sd,
                              "ci95_approx_low":lo,"ci95_approx_high":hi,"n_paired_seeds":len(dif)})
    diffs=pd.DataFrame(diff_rows); diffs.to_csv(outdir/"paired_model_comparisons.csv",index=False)
    # Keep a focused historical file with the two central event-awareness contrasts.
    diffs[diffs.baseline.isin(["Event-blind PI", "Zeraati-style event-free PI"])].to_csv(outdir/"event_awareness_advantage.csv",index=False)

    # Figure 1: mean reward-rate comparison with 95% approximate CIs.
    plt.figure(figsize=(11,6))
    scen_order=[x["name"] for x in SCENARIOS]; model_order=["Event-aware PI","Event-blind PI","Zeraati-style event-free PI","MVT-inspired threshold","Best fixed threshold"]
    x=np.arange(len(scen_order)); width=0.15
    for j,mname in enumerate(model_order):
        g=summary[summary.model==mname].set_index("scenario").reindex(scen_order)
        pos=x+(j-2)*width
        y=g.mean_reward_rate.to_numpy(); err=np.vstack([y-g.ci95_approx_low.to_numpy(),g.ci95_approx_high.to_numpy()-y])
        plt.bar(pos,y,width,label=mname,yerr=err,capsize=3)
    plt.axhline(0,color="black",linewidth=0.8); plt.xticks(x,scen_order,rotation=15,ha="right")
    plt.ylabel("Average reward per unit time"); plt.title("Two-action foraging: event-aware and baseline policies")
    plt.legend(frameon=False,ncol=2); plt.tight_layout(); plt.savefig(figdir/"01_reward_rate_comparison.png",dpi=200); plt.savefig(figdir/"01_reward_rate_comparison.pdf"); plt.close()

    # Figure 2: event-aware minus blind reward-rate advantage.
    central=diffs[diffs.baseline.isin(["Event-blind PI", "Zeraati-style event-free PI"])].copy()
    scen_order=[x["name"] for x in SCENARIOS]
    plt.figure(figsize=(9.5,5.8))
    markers={"Event-blind PI":"o","Zeraati-style event-free PI":"s"}
    offsets={"Event-blind PI":-0.12,"Zeraati-style event-free PI":0.12}
    for baseline in ["Event-blind PI", "Zeraati-style event-free PI"]:
        q=central[central.baseline==baseline].set_index("scenario").reindex(scen_order)
        yy=np.arange(len(scen_order))+offsets[baseline]; means=q.mean_difference.to_numpy()
        low=means-q.ci95_approx_low.to_numpy(); high=q.ci95_approx_high.to_numpy()-means
        plt.errorbar(means,yy,xerr=np.vstack([low,high]),fmt=markers[baseline],capsize=4,label=f"Event-aware minus {baseline}")
    plt.axvline(0,color='black',linewidth=1); plt.yticks(np.arange(len(scen_order)),scen_order); plt.xlabel("Reward-rate difference (event-aware − baseline)")
    plt.title("Event-aware reward-rate advantage over event-free baselines"); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(figdir/"02_event_awareness_advantage.png",dpi=200); plt.savefig(figdir/"02_event_awareness_advantage.pdf"); plt.close()

    # Figure 3: event-aware vs blind actions conditional on true event phase.
    action_summary=actions_df.groupby(["model","phase","action"],as_index=False).fraction.mean()
    chosen=action_summary[action_summary.model.isin(["Event-aware PI","Event-blind PI","Zeraati-style event-free PI"])]
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for ax,mname in zip(axs,["Event-aware PI","Event-blind PI","Zeraati-style event-free PI"]):
        piv=chosen[chosen.model==mname].pivot(index="phase",columns="action",values="fraction").reindex(PHASES)
        piv.plot(kind="bar",ax=ax); ax.set_title(mname); ax.set_xlabel("True ecological phase"); ax.set_ylabel("Action fraction")
        ax.tick_params(axis='x',rotation=35); ax.legend(frameon=False)
    fig.suptitle("Exploit/leave choices by true ecological phase"); fig.tight_layout(); fig.savefig(figdir/"03_action_fractions_by_phase.png",dpi=200); fig.savefig(figdir/"03_action_fractions_by_phase.pdf"); plt.close(fig)

    # Figure 4: policy-iteration convergence history.
    hist=pd.DataFrame(histories)
    fig,axs=plt.subplots(1,3,figsize=(15,4.3))
    for ax,model_name in zip(axs,["event_aware_pi","event_blind_pi","zeraati_style_pi"]):
        z=hist[hist.model==model_name]
        for scenario in scen_order:
            q=z[z.scenario==scenario]
            ax.plot(q.iteration,q.changed_states,marker='o',label=scenario)
        ax.set_title(model_name.replace('_',' ').title()); ax.set_xlabel("Policy-iteration pass"); ax.set_ylabel("States whose greedy action changed")
        ax.legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(figdir/"04_policy_iteration_convergence.png",dpi=200); fig.savefig(figdir/"04_policy_iteration_convergence.pdf"); plt.close(fig)

    # Figure 5: threshold tuning curves
    tune=pd.concat(thresholds_all,ignore_index=True)
    plt.figure(figsize=(8.5,5.3))
    for sc in scen_order:
        q=tune[tune.scenario==sc]
        plt.plot(q.threshold,q.mean_training_reward_rate,marker='o',label=sc)
    plt.xlabel("Fixed leaving threshold"); plt.ylabel("Training reward rate"); plt.title("Fixed-threshold baseline tuning")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(figdir/"05_threshold_tuning.png",dpi=200); plt.savefig(figdir/"05_threshold_tuning.pdf"); plt.close()

    # Save one trace per model from first scenario for inspection.
    for mname,trace in traces_saved.items(): pd.DataFrame(trace).to_csv(outdir/f"example_trace_{mname.lower().replace(' ','_')}.csv",index=False)

    report=make_report(cfg,summary,diffs,hist,thresholds_all)
    (outdir/"REPORT.md").write_text(report,encoding="utf-8")
    # manifest
    (outdir/"run_config.json").write_text(json.dumps({**cfg.__dict__,"scenarios":SCENARIOS},indent=2),encoding="utf-8")
    zip_path=outdir.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in outdir.rglob('*'):
            if p.is_file(): zf.write(p,p.relative_to(outdir.parent))
    print(f"\nResults saved to: {outdir.resolve()}\nZIP package: {zip_path.resolve()}")
    return summary,diffs,zip_path


def make_report(cfg,summary,diffs,hist,thresholds_all):
    scen_order=[x["name"] for x in SCENARIOS]
    lines=[
      "# Two-action event-aware foraging experiment", "",
      "## Summary", "",
      "This simulation compares event-aware and event-blind average-reward MDP policies, a Zeraati-style event-free policy-iteration baseline adapted to this environment, an MVT-inspired threshold, and a tuned fixed threshold. All five agents use exactly two actions: **exploit** and **leave**.", "",
      "The Zeraati-style baseline is **not an exact replication** of Zeraati et al. (2025). It adapts the paper's event-free state logic (current patch plus resource state) to this task's discretized resources and latent threat/recovery dynamics. The source paper instead models patch reward probabilities with its own depletion/replenishment dynamics and assumes known environmental statistics when solving the optimal policy.", "",
      "## State and action spaces", "",
      "The event-aware state is `s=(patch, resource_A, resource_B, resource_C, phase)`: 3 patches × 3 levels per patch × 5 phases = 405 states. The event-blind observation replaces the persistent phase with a transient boundary cue (`threat onset`, `recovery onset`, or `none`), yielding 243 observations. The Zeraati-style event-free observation is `(patch, resource_A, resource_B, resource_C)`, yielding 81 observations. The action set in every model is `{exploit, leave}`; leaving advances to the next patch in a fixed cycle A→B→C→A.", "",
      "## Constructing the three MDPs", "",
      "The event-aware MDP uses the full latent state and its known transition matrix. The event-blind and Zeraati-style models are constructed by aggregating latent states with the same observation. For each observation, compatible latent phases are mixed using a scenario-specific stationary phase prior calculated under a neutral 50:50 exploit/leave mixture, to obtain an approximate transition, reward, and duration model. The event-blind aggregation retains transient onset/recovery cues; the Zeraati-style model has no event cue. Because hidden phase affects dynamics and reward, these aggregated observations need not satisfy the Markov property exactly. This is an explicit approximation, not a property guaranteed by the original Zeraati paper.", "",
      "## Average-reward policy iteration", "",
      "For each policy, evaluation solves `h(s)=R_pi(s)-rho_pi*T_pi(s)+sum P_pi(s,s')h(s')`. Improvement selects `argmax_a [R(s,a)-rho_pi*T(s,a)+sum P(s'|s,a)h(s')]`. Iteration stops when the greedy action map stops changing. The same solver and stopping rule are used for all three MDP policies; only their state representations and transition/reward models differ. Threshold baselines do not use policy iteration.", "",
      "## Environmental event mechanism", "",
      "Threat onset is generated stochastically and independently of the immediate food-reward outcome. During threat onset/ongoing phases, exploitation incurs an illustrative danger cost; leaving advances to the next patch and makes the threat more likely to resolve. These are task-design assumptions, not empirical estimates.", "",
      "## Methods and comparison", "",
      f"The run used {cfg.n_seeds} evaluation seeds per model and scenario, {cfg.horizon} decision steps per seed, and {cfg.warmup} warm-up steps excluded from evaluation. Fixed thresholds were tuned on separate training seeds. The MVT-inspired threshold is based on the Zeraati-style event-free model's estimated average reward rate, clipped to the prespecified interval [0.05, 0.80].", "",
      "## Results", "",
      "Mean reward per unit time and approximate 95% confidence intervals across evaluation seeds:", "",
      summary[["scenario","model","mean_reward_rate","ci95_approx_low","ci95_approx_high","policy_iterations"]].to_markdown(index=False), "",
      "### Paired reward-rate differences", "",
      "Positive values indicate that event-aware PI had a higher reward rate than the comparison policy:", "",
      diffs.to_markdown(index=False), "",
      "### Policy-iteration convergence", "",
      "| Scenario | Event-aware passes | Event-blind passes | Zeraati-style event-free passes |", "|---|---:|---:|---:|"]
    for sc in scen_order:
        aw=summary[(summary.scenario==sc)&(summary.model=="Event-aware PI")].policy_iterations.iloc[0]
        bl=summary[(summary.scenario==sc)&(summary.model=="Event-blind PI")].policy_iterations.iloc[0]
        ze=summary[(summary.scenario==sc)&(summary.model=="Zeraati-style event-free PI")].policy_iterations.iloc[0]
        lines.append(f"| {sc} | {int(aw)} | {int(bl)} | {int(ze)} |")
    lines += ["", "## Interpretation and limitations", "",
      "The most direct contrasts are event-aware PI versus event-blind PI, and event-aware PI versus the Zeraati-style event-free PI adapted here. The latter has no persistent phase and receives no boundary cue. However, it is not a replication of the original Zeraati task: it shares the exploit/leave action set and broad average-reward policy-iteration approach, but uses the present task's resource discretization, hidden threat/recovery process, reward costs, action durations, and fixed cyclic patch order.", "",
      "The event-aware agent receives the true persistent phase label rather than learning an event detector. The event-blind and event-free MDPs use phase-prior aggregation based on a neutral 50:50 action mixture, an approximation that can discard information and can make their observation-level process non-Markovian. Because the phase prior is not conditioned on the policy being evaluated, the event-free baseline is not an exact solution to the underlying partially observable problem. Therefore, the results test the benefit of explicit event-state access under these assumptions; they do not prove that a sufficiently expressive history-sensitive or belief-state world model could not infer the same situation. The next key control is a history-sensitive event-free baseline trained with the same observations and interaction budget.", "",
      "Approximate confidence intervals use a normal critical value (1.96); inferential claims should be strengthened with more seeds, paired statistical tests, and sensitivity analyses over hazard rates, risk costs, replenishment parameters, event-cue reliability, state discretization, and aggregation priors.", "",
      "## Figures", "",
      "- `figures/01_reward_rate_comparison.png`: reward-rate comparison for all five policies.",
      "- `figures/02_event_awareness_advantage.png`: event-aware minus event-blind and event-free PI reward-rate differences.",
      "- `figures/03_action_fractions_by_phase.png`: exploit/leave choices by true ecological phase for all three PI policies.",
      "- `figures/04_policy_iteration_convergence.png`: changed greedy states by iteration for the three PI policies.",
      "- `figures/05_threshold_tuning.png`: fixed-threshold tuning on separate seeds.",
      "- `scenario_phase_priors.csv`: scenario-specific phase-prior weights used for approximate latent-state aggregation.", "",
      "## Reference", "",
      "Zeraati, R., Oña Jodar, T., Safavi, S., Cruz, B., Poo, C., & Dayan, P. (2025). *Optimal Foraging by Learning the World Model*. CCN 2025 conference abstract.", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='event_aware_two_action_results')
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--horizon',type=int,default=3000)
    parser.add_argument('--warmup',type=int,default=200)
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    cfg=Config(output_dir=args.output_dir,n_seeds=args.seeds,horizon=args.horizon,warmup=args.warmup,verbose=not args.quiet)
    run_experiment(cfg)

if __name__=='__main__': main()
