"""A scripted expert, and the reward sanity check it makes possible.

Before spending compute on RL, verify the obvious thing: that the reward
function is actually maximised by the behaviour you want. A Jacobian
resolved-rate controller solves this task without any learning, so its return
is a lower bound on what a good policy should score. If a do-nothing policy
outscores the expert, the reward is broken and no amount of PPO tuning will
fix it.
"""
from __future__ import annotations
import os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servoarm import ServoArmReachEnv, RewardCfg, DomainRandCfg   # noqa: E402


class JacobianExpert:
    """Damped least-squares resolved-rate control toward the target.

    Damping (lambda) is not optional: the arm passes through singular
    configurations where J J^T loses rank, and an undamped pseudo-inverse
    produces unbounded joint velocities exactly there.
    """

    # gain: the env's action is a CHANGE in setpoint, so this proportional
    # Jacobian law is really a discrete integrator,
    #     setpoint_{k+1} = setpoint_k + gain * J^+ e_k,
    # and its stability bounds the gain. Measured in simulation: everything up
    # to 1.0 settles (92% success), 1.5 degrades to 34%, and 3.0 oscillates
    # away from a target it has already reached (0%). 0.8 keeps margin.
    # This bit me: the first reward audit used gain 3.0 and "proved" the
    # shipped reward was broken. The reward was fine; the reference policy
    # was oscillating. An audit is only as trustworthy as its reference.
    def __init__(self, env: ServoArmReachEnv, gain=0.8, damping=0.06):
        self.env, self.gain, self.damping = env, gain, damping
        self.jacp = np.zeros((3, env.model.nv))

    def act(self):
        e = self.env
        err = e._target - e._tip
        mujoco.mj_jacSite(e.model, e.data, self.jacp, None, e.tip_sid)
        J = self.jacp[:, :4]
        JJt = J @ J.T + (self.damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err * self.gain)
        # convert desired joint step into the env's normalised delta action
        return np.clip(dq / e.max_delta, -1.0, 1.0)


def evaluate(policy_fn, rcfg, episodes=40, dr=True, seed=1234):
    env = ServoArmReachEnv(reward_cfg=rcfg, dr_cfg=DomainRandCfg(enabled=dr))
    expert = JacobianExpert(env)
    rets, succ, dists = [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        R = 0.0
        info = {}
        for _ in range(env.max_steps):
            a = policy_fn(env, expert)
            _, r, _, trunc, info = env.step(a)
            R += r
            if trunc:
                break
        rets.append(R)
        succ.append(float(info.get("is_success", 0.0)))
        dists.append(float(info.get("distance", np.nan)))
    return dict(ret=float(np.mean(rets)), ret_sd=float(np.std(rets)),
                success=float(np.mean(succ)), dist_cm=float(np.mean(dists)) * 100)


def _noisy(sigma):
    """The expert, corrupted. This stands in for a partially-trained policy."""
    def f(env, ex):
        return np.clip(ex.act() + env._np_random.normal(0, sigma, 4), -1, 1)
    return f


POLICIES = {
    "do_nothing":    lambda env, ex: np.zeros(4),
    "random":        lambda env, ex: env.action_space.sample(),
    "clumsy(s=0.5)": _noisy(0.5),
    "decent(s=0.2)": _noisy(0.2),
    "expert":        lambda env, ex: ex.act(),
}

VARIANTS = {
    "v1_sparse": RewardCfg(version="v1_sparse", sparse_only=True),
    "v2_distance": RewardCfg(version="v2_distance", w_potential=0.0, w_distance=1.0,
                             w_success=0.0, w_action=0.0, w_jerk=0.0, w_qvel=0.0),
    "v3_potential_raw": RewardCfg(version="v3_potential_raw", w_potential=12.0,
                                  w_distance=0.0, w_success=2.0,
                                  w_action=0.0, w_jerk=0.0, w_qvel=0.0),
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
}

if __name__ == "__main__":
    print("Reward sanity check: does each variant rank the policies correctly?")
    print("A usable reward must satisfy  expert > random  and  expert > do_nothing.\n")
    print(f"{'variant':18s} {'policy':12s} {'return':>10s} {'+-sd':>8s} {'success':>9s} {'dist_cm':>9s}")
    verdicts = {}
    for vn, rc in VARIANTS.items():
        scores = {}
        for pn, pf in POLICIES.items():
            r = evaluate(pf, rc)
            scores[pn] = r["ret"]
            print(f"{vn:18s} {pn:12s} {r['ret']:10.2f} {r['ret_sd']:8.2f} "
                  f"{r['success']:9.2f} {r['dist_cm']:9.1f}")
        # Test 1 -- well-ordering at the endpoints. Necessary, and the usual
        # place people stop.
        ordered = scores["expert"] > max(scores["random"], scores["do_nothing"])

        # Test 2 -- the landscape between them. A reward can be perfectly
        # well-ordered and still be unlearnable: if a clumsy attempt at the
        # task scores WORSE than standing still, then the first thing an
        # exploring policy discovers is that moving is punished, and it
        # converges to inaction long before it ever finds the target. This is
        # exactly what happened here -- the shipped reward passed test 1 and
        # PPO still flatlined for 150k steps.
        learnable = scores["clumsy(s=0.5)"] > scores["do_nothing"]

        verdicts[vn] = dict(ordered=ordered, learnable=learnable)
        msg = ("USABLE" if ordered and learnable else
               "BROKEN: expert is not the best policy" if not ordered else
               "WELL-ORDERED BUT UNLEARNABLE: clumsy attempts score below "
               "doing nothing, so exploration collapses to inaction")
        print(f"{'':18s} --> {msg}\n")

    print("summary:")
    for k, v in verdicts.items():
        tag = "usable" if all(v.values()) else ("unlearnable" if v["ordered"] else "broken")
        print(f"  {k:18s} ordered={str(v['ordered']):5s} learnable={str(v['learnable']):5s}  -> {tag}")
