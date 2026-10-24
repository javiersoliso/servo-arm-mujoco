"""Command-trace comparison: what the regularisation fine-tune actually did.

Success rate alone cannot show this. Two policies with the same success rate
can differ by an order of magnitude in how much they thrash the servos, and
only one of them is safe to put on hardware.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from servoarm import ServoArmReachEnv, RewardCfg, DomainRandCfg   # noqa: E402
from scripted_expert import JacobianExpert                        # noqa: E402
from train.evaluate import load_policy                            # noqa: E402

RUNS = [("before fine-tune (v7_strong)", "runs/long_v7_strong", "#e4572e"),
        ("after fine-tune (v9_smooth)",  "runs/finetune_smooth", "#2e86ab")]
EPISODE_SEED = 70001
JOINT = 1          # shoulder


def trace(run_dir):
    model, vn = load_policy(run_dir)
    env = ServoArmReachEnv(reward_cfg=RewardCfg(), dr_cfg=DomainRandCfg(enabled=True))
    obs, _ = env.reset(seed=EPISODE_SEED)
    acts, dists = [], []
    for _ in range(env.max_steps):
        o = vn.normalize_obs(obs.reshape(1, -1)) if vn is not None else obs.reshape(1, -1)
        a, _ = model.predict(o, deterministic=True)
        a = np.asarray(a).reshape(-1)
        obs, _, _, trunc, info = env.step(a)
        acts.append(a.copy()); dists.append(info["distance"] * 100)
        if trunc: break
    return np.array(acts), np.array(dists)


def expert_trace():
    env = ServoArmReachEnv(reward_cfg=RewardCfg(), dr_cfg=DomainRandCfg(enabled=True))
    ex = JacobianExpert(env)
    env.reset(seed=EPISODE_SEED)
    acts, dists = [], []
    for _ in range(env.max_steps):
        a = ex.act()
        _, _, _, trunc, info = env.step(a)
        acts.append(a.copy()); dists.append(info["distance"] * 100)
        if trunc: break
    return np.array(acts), np.array(dists)


fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                         gridspec_kw=dict(height_ratios=[2, 1]))
fig.patch.set_facecolor("white")

ea, ed = expert_trace()
axes[0].plot(ea[:, JOINT], color="#8d99ae", lw=1.4, ls="--",
             label=f"Jacobian expert  (jerk {np.mean(np.sum(np.diff(ea,axis=0)**2,axis=1)):.3f})")
axes[1].plot(ed, color="#8d99ae", lw=1.4, ls="--")

for label, run, col in RUNS:
    a, d = trace(run)
    j = float(np.mean(np.sum(np.diff(a, axis=0) ** 2, axis=1)))
    axes[0].plot(a[:, JOINT], color=col, lw=1.6, label=f"{label}  (jerk {j:.3f})")
    axes[1].plot(d, color=col, lw=1.6)

axes[0].set_ylabel("shoulder command\n(normalised setpoint delta)")
axes[0].set_title("Same task, same episode: what the smoothness fine-tune changed",
                  fontsize=12, weight="bold")
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
axes[0].grid(alpha=0.25); axes[0].axhline(0, color="#adb5bd", lw=0.8)

axes[1].axhline(2.5, color="#2a9d8f", lw=1.2, ls=":", label="2.5 cm success threshold")
axes[1].set_ylabel("tip to target (cm)"); axes[1].set_xlabel("control step (20 Hz)")
axes[1].legend(frameon=False, fontsize=9); axes[1].grid(alpha=0.25)

for ax in axes:
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

plt.tight_layout()
os.makedirs("media", exist_ok=True)
plt.savefig("media/command_smoothness.png", dpi=150, facecolor="white")
print("wrote media/command_smoothness.png")
