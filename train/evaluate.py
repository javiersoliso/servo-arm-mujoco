"""Evaluate a trained policy on ServoArmReach-v0.

Reports what matters for deciding whether a policy is worth putting on
hardware, not just the training-time success rate:

  * success rate against the Jacobian expert (the achievable ceiling) and
    against doing nothing (the floor)
  * command smoothness, because a policy that succeeds by chattering is
    useless on real gears
  * robustness OUTSIDE the training randomisation range -- the closest thing
    to a sim-to-real number available without hardware
  * a breakdown by target distance, so a headline average cannot hide a
    systematic failure on far targets
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from servoarm import ServoArmReachEnv, RewardCfg, DomainRandCfg   # noqa: E402
from scripted_expert import JacobianExpert                        # noqa: E402
from stable_baselines3 import PPO                                 # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402


def load_policy(run_dir):
    model = PPO.load(os.path.join(run_dir, "policy"), device="cpu")
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    vn = None
    if os.path.exists(vn_path):
        dummy = DummyVecEnv([lambda: ServoArmReachEnv()])
        vn = VecNormalize.load(vn_path, dummy)
        vn.training = False
    return model, vn


def rollout(env, act_fn, episodes, seed0=90000):
    succ, finals, jerks, torques, dists0 = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        dists0.append(float(np.linalg.norm(env._target - env._home_tip)))
        prev = np.zeros(env.model.nu)
        jsum = 0.0
        tmax = 0.0
        info = {}
        for _ in range(env.max_steps):
            a = act_fn(obs, env)
            obs, _, _, trunc, info = env.step(a)
            jsum += float(np.sum(np.square(a - prev)))
            prev = a
            tmax = max(tmax, float(np.abs(env.data.actuator_force).max()))
            if trunc:
                break
        succ.append(float(info.get("is_success", 0.0)))
        finals.append(float(info.get("distance", np.nan)))
        jerks.append(jsum / env.max_steps)
        torques.append(tmax)
    return dict(success=float(np.mean(succ)),
                final_cm=float(np.mean(finals)) * 100,
                final_cm_p90=float(np.percentile(finals, 90)) * 100,
                jerk=float(np.mean(jerks)),
                peak_torque=float(np.mean(torques)),
                succ_list=succ, dist0_list=dists0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model, vn = load_policy(a.run)
    rcfg = RewardCfg()

    def policy_fn(obs, env):
        o = vn.normalize_obs(obs.reshape(1, -1)) if vn is not None else obs.reshape(1, -1)
        act, _ = model.predict(o, deterministic=True)
        return np.asarray(act).reshape(-1)

    results = {}

    # -- conditions ---------------------------------------------------------
    NOM = DomainRandCfg(enabled=False)
    TRAIN = DomainRandCfg(enabled=True)
    # 2x the training ranges: tests whether the policy generalises past what
    # it saw, which is the property sim-to-real actually depends on.
    WIDE = DomainRandCfg(enabled=True, mass=(0.70, 1.30), damping=(0.40, 1.60),
                         armature=(0.60, 1.40), kp=(0.60, 1.40),
                         obs_noise_qpos=0.008, obs_noise_qvel=0.04,
                         max_action_latency=2)

    for label, dr in [("nominal", NOM), ("train_DR", TRAIN), ("wide_DR_2x", WIDE)]:
        env = ServoArmReachEnv(reward_cfg=rcfg, dr_cfg=dr)
        results[f"policy/{label}"] = rollout(env, policy_fn, a.episodes)

    # -- baselines ----------------------------------------------------------
    for label, dr in [("nominal", NOM), ("train_DR", TRAIN)]:
        env = ServoArmReachEnv(reward_cfg=rcfg, dr_cfg=dr)
        ex = JacobianExpert(env)
        results[f"expert/{label}"] = rollout(env, lambda o, e: ex.act(), a.episodes)
        env2 = ServoArmReachEnv(reward_cfg=rcfg, dr_cfg=dr)
        results[f"do_nothing/{label}"] = rollout(env2, lambda o, e: np.zeros(4), a.episodes)

    # -- report -------------------------------------------------------------
    print(f"\nEvaluation of {a.run}   ({a.episodes} episodes per condition)\n")
    print(f"{'agent / condition':28s} {'success':>8s} {'final cm':>9s} {'p90 cm':>8s} "
          f"{'jerk':>8s} {'peak Nm':>8s}")
    for k, r in results.items():
        print(f"{k:28s} {r['success']:8.2f} {r['final_cm']:9.2f} {r['final_cm_p90']:8.2f} "
              f"{r['jerk']:8.4f} {r['peak_torque']:8.3f}")

    # -- breakdown by target distance --------------------------------------
    r = results["policy/train_DR"]
    d0 = np.array(r["dist0_list"]) * 100
    sc = np.array(r["succ_list"])
    print(f"\nsuccess by target distance from the home pose (train_DR):")
    edges = [0, 10, 15, 20, 25, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d0 >= lo) & (d0 < hi)
        if m.sum():
            print(f"  {lo:3d}-{hi:3d} cm  n={m.sum():4d}  success {sc[m].mean():.2f}")

    ceiling = results["expert/train_DR"]["success"]
    got = results["policy/train_DR"]["success"]
    print(f"\npolicy reaches {100*got/ceiling:.0f}% of the scripted-expert ceiling "
          f"({got:.2f} vs {ceiling:.2f} under the same randomisation)")
    ex_j = results["expert/train_DR"]["jerk"]
    po_j = results["policy/train_DR"]["jerk"]
    print(f"command smoothness: policy jerk {po_j:.4f} vs expert {ex_j:.4f} "
          f"({'smoother' if po_j < ex_j else 'rougher'} by {abs(po_j-ex_j)/max(ex_j,1e-9)*100:.0f}%)")

    out = a.out or os.path.join(a.run, "evaluation.json")
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_list")}
             for k, v in results.items()}
    with open(out, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
