# Reward iterations

How the reward function for `ServoArmReach-v0` reached its shipped form, and
what each intermediate version got wrong. Numbers come from
`tests/scripted_expert.py` (the audit) and `train/train_ppo.py` (the runs), so
every claim here is reproducible by name.

## The task

Drive the tip of a 4-DOF servo arm to a target sampled anywhere in its
reachable workspace. Success = tip within **2.5 cm** of the target for **5
consecutive control steps** (a settled hold, not a fly-through). 20 Hz
control, 4 s episodes, domain randomisation on throughout.

## Auditing a reward before spending compute on it

The expensive way to discover a bad reward is to train for a few hours and
find a flat curve. The cheap way is to score a handful of reference policies
under it and check the ordering. `tests/scripted_expert.py` uses a Jacobian
resolved-rate controller as the reference "good" policy, plus deliberately
corrupted versions of it, and applies two tests:

1. **Well-ordering** — `expert > random` and `expert > do_nothing`.
   Necessary, and where most reward audits stop.
2. **Learnability** — `clumsy > do_nothing`, where `clumsy` is the expert plus
   σ=0.5 Gaussian noise, standing in for a partially-trained policy.

Test 2 is the one that earns its keep. A reward can be perfectly well-ordered
and still be unlearnable: if a *clumsy attempt* at the task scores worse than
standing still, then the first thing an exploring policy discovers is that
moving is punished, and it converges to inaction long before it ever finds the
target.

### Audit results

All figures are mean return over 40 episodes with domain randomisation on.

| variant | do_nothing | random | clumsy σ=0.5 | decent σ=0.2 | expert | verdict |
|---|---|---|---|---|---|---|
| `v1_sparse` | 0.00 | 0.03 | 8.53 | 27.05 | 34.08 | usable |
| `v2_distance` | −14.79 | −15.72 | −6.09 | −4.94 | −4.47 | usable |
| `v3_potential_raw` | 1.75 | 1.86 | 19.26 | 56.27 | 70.35 | usable |
| `v4_potential` | 1.75 | −25.85 | **−2.98** | 48.06 | 65.11 | **unlearnable** |
| `v5_balanced` | 1.75 | −4.02 | 14.53 | 54.46 | 69.16 | usable |

`v4` passes test 1 comfortably (expert 65.11 ≫ do_nothing 1.75) and fails test
2: a clumsy attempt scores −2.98 against 1.75 for standing still.

### What v4 got wrong

`v4` carried `w_action=0.02`, `w_jerk=0.10`, `w_qvel=0.004`. Those look small.
They are not, because they are paid **every step for 80 steps** while the task
reward is only collected on arrival. An exploring policy emits near-maximal
command churn, so the jerk term alone costs it roughly 0.32 per step — about
25 return per episode — against a shaping signal worth ~2 for the entire
traversal. The reward was, in effect, paying the agent to hold still.

PPO found that optimum exactly as it should have. **The algorithm was not
failing; it was succeeding at a badly specified objective.** Over 150k steps
the mean final distance *grew* from 18.7 cm to 24.1 cm as the policy learned
to stop moving.

### Sizing the regularisation instead of guessing it

The audit also supplies the fix. Under `v3` (identical, minus all penalties) a
clumsy policy scores 19.26 and doing nothing scores 1.75, so there is a
**penalty budget of ~17.5 return per episode** before clumsy attempts stop
being worth making. `v4` spent 22.2 and went bankrupt. `v5_balanced` cuts the
weights 5x (`w_action=0.004`, `w_jerk=0.02`, `w_qvel=0.001`) to spend ~4.4 — a
4x margin — and still shapes the command. Clumsy recovers to 14.53.

## Why potential-based shaping

`v2_distance` uses the reflex answer, `r = -distance`. It is well-ordered and
it trains, but it quietly changes the task: a constant per-step distance
penalty makes a *longer* episode strictly worse, so the agent is paid to end
episodes rather than to hold position — which is precisely the opposite of a
settled reach.

`v3` onward instead use potential-based shaping, `γφ(s') − φ(s)` with
`φ = −distance`. Ng et al. (1999) proves this class of shaping leaves the
optimal policy unchanged, so it accelerates learning without redefining the
task. Two details this depends on:

* **`gamma` must match.** The shaping term uses γ=0.99 and PPO is configured
  with `gamma=0.99`. If they differ, the invariance guarantee does not hold
  and the shaping is silently changing the objective again.
* **The bonus goes inside φ, not beside it.** `v6`/`v8` add a proximity bump
  `φ(d) = −d + w_prox · exp(−d/σ)`. Placed inside the potential it stays
  policy-invariant; added as a standalone reward term it would not be.

## Variant definitions

| variant | shaping | distance | success | action | jerk | qvel | prox |
|---|---|---|---|---|---|---|---|
| `v1_sparse` | — | — | 1.0 (sparse) | — | — | — | — |
| `v2_distance` | — | 1.0 | — | — | — | — | — |
| `v3_potential_raw` | 12 | — | 2.0 | — | — | — | — |
| `v4_potential` | 12 | — | 2.0 | 0.02 | 0.10 | 0.004 | — |
| `v5_balanced` | 12 | — | 2.0 | 0.004 | 0.02 | 0.001 | — |
| `v6_proximity` | 12 | — | 2.0 | 0.004 | 0.02 | 0.001 | 0.35 |
| `v7_strong` | 40 | — | 2.0 | 0.004 | 0.02 | 0.001 | — |
| `v8_strong_prox` | 40 | — | 2.0 | 0.004 | 0.02 | 0.001 | 0.35 |

---

## Training results

**Variant comparison.** 200k steps, 2 seeds each, domain randomisation on.
Mean ± spread across seeds.

| variant | success | final distance (cm) | distance 25k → 200k |
|---|---|---|---|
| `v1_sparse` | 0.03 ± 0.01 | 10.8 ± 0.7 | 17.9 → 11.1 |
| `v2_distance` | 0.04 ± 0.02 | **5.9 ± 0.1** | 13.4 → 6.1 |
| `v3_potential_raw` | **0.12 ± 0.01** | 8.5 ± 0.2 | 13.5 → 9.0 |
| `v4_potential` | **0.00 ± 0.00** | 22.0 ± 3.1 | **20.1 → 22.2** |
| `v5_balanced` | 0.06 ± 0.03 | 8.7 ± 0.0 | 15.8 → 9.1 |
| `v7_strong` | 0.05 ± 0.02 | 8.3 ± 0.7 | 14.5 → 8.6 |

`v4` is the only variant that gets *worse* over training — mean final distance
grows from 20.1 cm to 22.2 cm as the policy learns to stop moving. That is the
learnability audit's prediction, confirmed at a cost of 400k wasted steps
rather than the several hours it would have taken to find by tuning PPO.

`v1_sparse` does learn, slowly. Sparse rewards are not hopeless on a task this
small; they are just wasteful.

**Long runs.** 800k steps, single seed.

| variant | success @ 800k |
|---|---|
| `v7_strong` (shaping ×40) | **0.57** |
| `v8_strong_prox` (+ proximity bump) | 0.41 |

The proximity bump was my idea and it did not work — `v8` is 16 points *worse*
than plain `v7`. The reasoning behind it was sound (pure `−d` shaping has a
constant gradient, so the pull toward the goal is no stronger at 2 cm than at
20 cm, and the last 2 cm are where the success bonus lives) and the
implementation was correct (inside `φ`, so still policy-invariant). It simply
did not survive contact with the measurement. Recorded here rather than
quietly deleted, because a variant that was worth testing is worth reporting.

## The fine-tune: a penalty budget that expires

The `v5` rebalance solved learnability by cutting the regularisation 5x, and
that has a cost: the resulting policy reaches the target and then **chatters**.
Measured jerk 3.49 against 0.19 for the scripted expert — 18x rougher. The
command trace (`media/command_smoothness.png`) shows the shoulder servo being
driven as a full-amplitude ±1 square wave at ~10 Hz for the entire episode.
The success metric is blind to this, and the hardware would not be.

The fix follows from re-reading what the penalty budget actually is. It was
derived from a *clumsy* policy's return — an agent that is still exploring and
not yet collecting task reward. A policy that already succeeds ~60% of the
time banks a large success bonus every episode, so it can afford penalties
that would have starved a random policy. **The budget constrains exploration,
not competence, so it expires once the task is learned.**

`v9_smooth_finetune` therefore warm-starts from the trained `v7` policy and
raises the penalties past even `v4`'s levels (`w_jerk` 0.02 → 0.25,
`w_action` 0.004 → 0.02), at a reduced learning rate, for 300k steps:

| | success (train DR) | jerk | vs expert |
|---|---|---|---|
| before (`v7_strong`) | 0.63 | 3.490 | 18x rougher |
| after (`v9_smooth_finetune`) | 0.61 | **0.316** | 1.6x rougher |

**An 11x reduction in command jerk for 2 points of success rate.** Training
from scratch with `v9`'s weights is essentially `v4`, which does not learn at
all — so the same weights are either fatal or nearly free depending only on
when they are applied.

One implementation detail that is easy to get wrong: the fine-tune reloads the
*same* `VecNormalize` statistics the parent was trained under. A fresh
normaliser starts from zero mean and unit variance and would feed the policy
inputs on a different scale than it was trained on, destroying it before
fine-tuning begins.
