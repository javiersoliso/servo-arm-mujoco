"""Train PPO on ServoArmReach-v0.

Usage:
  MUJOCO_GL=osmesa python3 train/train_ppo.py --variant v4_potential --steps 400000
"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servoarm import ServoArmReachEnv, RewardCfg, DomainRandCfg   # noqa: E402

from stable_baselines3 import PPO                                  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.monitor import Monitor               # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback        # noqa: E402


# Reward variants. These are the exact configurations compared in
# docs/reward_iterations.md -- the doc quotes numbers produced by this file.
VARIANTS = {
    # 1. Pure sparse. The honest baseline: no shaping at all.
    "v1_sparse": RewardCfg(version="v1_sparse", sparse_only=True),

    # 2. Raw negative distance. The reflex answer, and the one that quietly
    #    changes the optimal policy: a constant per-step distance penalty makes
    #    a long episode strictly worse than a short one.
    "v2_distance": RewardCfg(version="v2_distance", w_potential=0.0, w_distance=1.0,
                             w_success=0.0, w_action=0.0, w_jerk=0.0, w_qvel=0.0),

    # 3. Potential-based shaping alone. Correct in principle, but with nothing
    #    penalising command churn.
    "v3_potential_raw": RewardCfg(version="v3_potential_raw", w_potential=12.0,
                                  w_distance=0.0, w_success=2.0,
                                  w_action=0.0, w_jerk=0.0, w_qvel=0.0),

    # 4. Shaping + regularisation, weights chosen by eye. Documented failure:
    #    passes the well-ordering audit and still cannot be learned.
    "v4_potential": RewardCfg(version="v4_potential"),

    # 5. v4's regularisation, scaled to fit inside a measured budget.
    #    The constraint comes from the learnability test: the total penalty an
    #    episode can accumulate must stay below the task reward that a clumsy
    #    but improving attempt earns. Measured under v3 (no penalties) a clumsy
    #    policy scores +19.3 while doing nothing scores +1.8, so the penalty
    #    budget is 17.5 return per episode. v4 spent 22.2 and went bankrupt.
    #    These weights spend ~4.4, a 4x margin, and still shape the command.
    "v5_balanced": RewardCfg(version="v5_balanced", w_potential=12.0,
                             w_distance=0.0, w_success=2.0,
                             w_action=0.004, w_jerk=0.02, w_qvel=0.001),

    # 6. v5 plus a proximity bump inside the potential. Same optimal policy,
    #    stronger gradient exactly where the success bonus lives.
    "v6_proximity": RewardCfg(version="v6_proximity", w_potential=12.0,
                              w_distance=0.0, w_success=2.0,
                              w_action=0.004, w_jerk=0.02, w_qvel=0.001,
                              w_prox=0.35, prox_sigma=0.03),

    # 7. v5 with a stronger shaping weight, as a control for v6: it tests
    #    whether v6's gain comes from the SHAPE of the potential or merely
    #    from a larger coefficient.
    "v7_strong": RewardCfg(version="v7_strong", w_potential=40.0,
                           w_distance=0.0, w_success=2.0,
                           w_action=0.004, w_jerk=0.02, w_qvel=0.001),

    # 9. Heavy regularisation, intended ONLY as a fine-tune stage on top of an
    #    already-competent policy. The penalty budget measured in
    #    docs/reward_iterations.md binds during EXPLORATION, when the agent is
    #    not yet earning task reward. A policy that already succeeds ~60% of
    #    the time collects a large success bonus every episode, so it can
    #    afford penalties that would have starved a random policy. Training
    #    from scratch with these weights is v4, which does not learn at all.
    "v9_smooth_finetune": RewardCfg(version="v9_smooth_finetune", w_potential=40.0,
                                    w_distance=0.0, w_success=2.0,
                                    w_action=0.02, w_jerk=0.25, w_qvel=0.005),

    # 8. Strong shaping AND the proximity bump. v6 showed the bump does
    #    nothing while the policy is still 10+ cm out (sigma is 3 cm), so it
    #    is only expected to pay off late in training, once the endgame is
    #    what limits success rate. Tested over a long run, not a short one.
    "v8_strong_prox": RewardCfg(version="v8_strong_prox", w_potential=40.0,
                                w_distance=0.0, w_success=2.0,
                                w_action=0.004, w_jerk=0.02, w_qvel=0.001,
                                w_prox=0.35, prox_sigma=0.03),
}


class SuccessCallback(BaseCallback):
    """Track rolling success rate and command smoothness during training."""

    def __init__(self, window=100, record_every=5000):
        super().__init__()
        self.window = window
        self.record_every = record_every
        self._next_record = record_every
        self.successes, self.dists, self.jerks = [], [], []
        self.history = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.successes.append(float(info.get("is_success", 0.0)))
                self.dists.append(float(info.get("distance", np.nan)))
                if "r_jerk" in info:
                    self.jerks.append(-float(info["r_jerk"]))
        if self.successes:
            sr = float(np.mean(self.successes[-self.window:]))
            self.logger.record("task/success_rate", sr)
            self.logger.record("task/final_distance_cm",
                               float(np.mean(self.dists[-self.window:])) * 100)
            if self.jerks:
                self.logger.record("task/jerk_cost", float(np.mean(self.jerks[-self.window:])))
            # num_timesteps advances by n_envs per call, so a modulo window
            # test fires several times in a row and floods the curve with
            # near-duplicate points. Use a monotone threshold instead.
            if self.num_timesteps >= self._next_record:
                self._next_record += self.record_every
                self.history.append(dict(
                    steps=int(self.num_timesteps), success=round(sr, 4),
                    distance_cm=round(float(np.mean(self.dists[-self.window:])) * 100, 2)))
        return True


def make_env(rcfg, dr, seed, rank):
    def _f():
        env = ServoArmReachEnv(reward_cfg=rcfg,
                               dr_cfg=DomainRandCfg(enabled=dr),
                               seed=seed + rank)
        return Monitor(env, info_keywords=("is_success", "distance"))
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v4_potential", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a previous run directory (policy + "
                         "observation normalisation statistics)")
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    rcfg = VARIANTS[a.variant]
    dr = not a.no_dr
    out = a.out or f"runs/{a.variant}_seed{a.seed}"
    os.makedirs(out, exist_ok=True)

    venv = DummyVecEnv([make_env(rcfg, dr, a.seed * 1000, i) for i in range(a.n_envs)])
    # Reward normalisation is deliberately OFF: the whole point of the reward
    # study is to compare terms across variants, and a running reward scaler
    # would make those numbers incomparable between runs.
    if a.init_from:
        # Reload the SAME observation statistics the parent policy was trained
        # under. A fresh VecNormalize would start from zero mean / unit
        # variance and feed the policy inputs on a different scale than it was
        # trained on, which silently destroys it before fine-tuning begins.
        venv = VecNormalize.load(os.path.join(a.init_from, "vecnormalize.pkl"), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv, seed=a.seed, verbose=0,
        learning_rate=a.lr,
        n_steps=256,                 # 256 x 8 envs = 2048 transitions per update
        batch_size=256,
        n_epochs=10,
        # gamma MUST equal the gamma used in the potential shaping term, or the
        # policy-invariance guarantee of Ng et al. (1999) does not hold.
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.004,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        tensorboard_log=out,
    )

    if a.init_from:
        parent = PPO.load(os.path.join(a.init_from, "policy"), device="cpu")
        model.policy.load_state_dict(parent.policy.state_dict())
        print(f"warm-started from {a.init_from}")

    cb = SuccessCallback()
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb, progress_bar=False)
    el = time.time() - t0

    model.save(os.path.join(out, "policy"))
    venv.save(os.path.join(out, "vecnormalize.pkl"))

    final_sr = float(np.mean(cb.successes[-100:])) if cb.successes else 0.0
    final_d = float(np.mean(cb.dists[-100:])) * 100 if cb.dists else float("nan")
    final_j = float(np.mean(cb.jerks[-100:])) if cb.jerks else None
    summary = dict(variant=a.variant, seed=a.seed, steps=a.steps,
                   init_from=a.init_from, learning_rate=a.lr,
                   domain_randomisation=dr, wall_clock_s=round(el, 1),
                   train_success_rate=round(final_sr, 3),
                   train_final_distance_cm=round(final_d, 2),
                   train_jerk_cost=round(final_j, 4) if final_j is not None else None,
                   reward_cfg=asdict(rcfg), curve=cb.history)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
