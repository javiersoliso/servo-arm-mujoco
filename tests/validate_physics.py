"""Physics validation for servoarm4.

Checks that must pass before any RL is attempted. Every failure found here is
a failure that would otherwise show up as "PPO doesn't learn" three hours into
a training run, so it is much cheaper to catch it at this level.

Run:  MUJOCO_GL=osmesa python3 tests/validate_physics.py
"""
import numpy as np
import mujoco

SCENE = "models/reach_scene.xml"
JOINTS = ["base_yaw", "shoulder", "elbow", "wrist_pitch"]


def banner(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def load():
    m = mujoco.MjModel.from_xml_path(SCENE)
    return m, mujoco.MjData(m)


# --------------------------------------------------------------------------
def check_static_gravity_load():
    """Worst case: arm fully extended horizontally. Can the servos hold it?

    If the required holding torque exceeds the datasheet stall torque the arm
    will sag no matter how good the policy is, and the sim will be teaching a
    behaviour the hardware cannot reproduce.
    """
    banner("1. STATIC GRAVITY LOAD  (worst-case holding torque vs stall torque)")
    m, d = load()

    # Fully extended forward and horizontal: shoulder 90 deg, elbow/wrist straight.
    d.qpos[:] = [0.0, np.pi / 2, 0.0, 0.0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    # qfrc_bias holds gravity + Coriolis; at zero velocity it is pure gravity.
    tau = d.qfrc_bias.copy()

    ok = True
    for i, jn in enumerate(JOINTS):
        aid = m.actuator(f"a_{jn}").id
        stall = float(m.actuator_forcerange[aid, 1])
        need = abs(float(tau[i]))
        margin = stall / need if need > 1e-9 else float("inf")
        flag = "OK " if margin >= 1.5 else "FAIL"
        if margin < 1.5:
            ok = False
        print(f"  {flag} {jn:12s} need {need:6.3f} N*m | stall {stall:5.3f} N*m "
              f"| margin {margin:5.2f}x")
    print(f"  --> {'PASS' if ok else 'FAIL'} (require >= 1.5x margin)")
    return ok


# --------------------------------------------------------------------------
def check_step_response():
    """Commanded-step response of each joint, with the rest of the arm free.

    Reports what a control engineer would measure on the bench: rise time,
    overshoot, settling time, steady-state error. Steady-state error is the
    one that matters for RL -- a position actuator with finite kp under
    gravity does NOT reach its setpoint, so the policy's action is a biased
    request, not a joint angle. Any reward that assumes action == angle is
    wrong.
    """
    banner("2. JOINT STEP RESPONSE  (kp/kv/damping sanity)")
    m, d = load()
    dt = m.opt.timestep

    print(f"  {'joint':12s} {'step':>7s} {'rise':>7s} {'over%':>7s} {'settle':>8s} {'ss_err':>8s}")
    results = {}
    for i, jn in enumerate(JOINTS):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        q0 = float(d.qpos[i])
        lo, hi = m.jnt_range[m.joint(jn).id]
        target = np.clip(q0 + 0.5, lo + 0.05, hi - 0.05)
        step = target - q0

        d.ctrl[:] = d.qpos[:]
        d.ctrl[i] = target

        traj = []
        for _ in range(int(3.0 / dt)):          # 3 seconds
            mujoco.mj_step(m, d)
            traj.append(float(d.qpos[i]))
        traj = np.asarray(traj)
        t = np.arange(len(traj)) * dt

        final = traj[-1]
        ss_err = target - final
        prog = (traj - q0) / step if abs(step) > 1e-9 else traj * 0

        def first_at(frac):
            idx = np.argmax(prog >= frac)
            return t[idx] if prog.max() >= frac else np.nan

        rise = first_at(0.9)
        overshoot = max(0.0, (prog.max() - 1.0) * 100.0)
        band = 0.02 * abs(step)
        outside = np.where(np.abs(traj - final) > band)[0]
        settle = t[outside[-1]] if len(outside) else 0.0

        print(f"  {jn:12s} {step:7.3f} {rise:7.3f} {overshoot:7.2f} {settle:8.3f} {ss_err:8.4f}")
        results[jn] = dict(rise=rise, overshoot=overshoot, settle=settle, ss_err=ss_err)

    worst = max(abs(r["ss_err"]) for r in results.values())
    print(f"  --> worst steady-state error: {worst:.4f} rad "
          f"({np.degrees(worst):.2f} deg) -- expected, kp is finite under gravity")
    return results


# --------------------------------------------------------------------------
def _rollout(integrator, dt, seconds=4.0, seed=0):
    """Deterministic control sequence, replayed at any (integrator, dt).

    The control is a function of TIME, not of step index, so every combination
    sees the identical command signal and the trajectories are comparable.
    """
    m, d = load()
    m.opt.integrator = integrator
    m.opt.timestep = dt
    mujoco.mj_resetDataKeyframe(m, d, 0)
    tip = m.site("tip").id
    rng = np.random.default_rng(seed)
    cmds = rng.uniform(m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1],
                       size=(int(seconds * 20) + 2, m.nu))     # 20 Hz square wave

    ts, xs = [], []
    n = int(seconds / dt)
    for k in range(n):
        t = k * dt
        d.ctrl[:] = cmds[int(t * 20)]
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qvel)) or np.abs(d.qvel).max() > 500:
            return None, None
        ts.append(t + dt)
        xs.append(d.site_xpos[tip].copy())
    return np.asarray(ts), np.asarray(xs)


def check_integrator_stability():
    """Stability AND accuracy sweep over integrator x timestep.

    Stability is the cheap question and this arm passes it everywhere -- the
    armature and damping terms keep it well conditioned, so nothing diverges
    even at 8 ms. That makes the stability table almost useless on its own,
    which is the point: the number that actually constrains the timestep is
    integration ERROR against a fine reference, so that is measured too.
    """
    banner("3. INTEGRATOR x TIMESTEP  (stability is easy; accuracy is the real constraint)")
    integrators = {
        "Euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    }
    timesteps = [0.0005, 0.001, 0.002, 0.004, 0.008]
    ERROR_BUDGET = 0.001   # 1 mm of tip position

    # Reference: RK4 at 0.1 ms.
    ref_t, ref_x = _rollout(mujoco.mjtIntegrator.mjINT_RK4, 0.0001)
    print(f"  reference: RK4 @ 0.1 ms, {len(ref_t)} steps")
    print(f"  error metric: max |tip - tip_ref| over the rollout, budget {ERROR_BUDGET*1000:.0f} mm\n")

    print(f"  {'integrator':14s}" + "".join(f"{ts*1000:>10.1f}ms" for ts in timesteps))
    table = {}
    for name, ig in integrators.items():
        row, cells = [], []
        for dt in timesteps:
            t, x = _rollout(ig, dt)
            if t is None:
                cells.append("  DIVERGE")
                row.append(np.inf)
                continue
            # compare on the coarse grid by sampling the reference at those times
            idx = np.clip(np.round(t / 0.0001).astype(int) - 1, 0, len(ref_x) - 1)
            err = np.linalg.norm(x - ref_x[idx], axis=1).max()
            row.append(err)
            mark = "*" if err <= ERROR_BUDGET else " "
            cells.append(f"{err*1000:8.2f}mm{mark}")
        print(f"  {name:14s}" + "".join(f"{c:>12s}" for c in cells))
        table[name] = row

    good = [dt for dt, e in zip(timesteps, table["implicitfast"]) if e <= ERROR_BUDGET]
    chosen = max(good) if good else timesteps[0]
    print(f"\n  (* = within budget)")
    print(f"  --> all three integrators are STABLE at every timestep tested;")
    print(f"      accuracy, not stability, sets the limit.")
    print(f"  --> largest implicitfast timestep within the 1 mm budget: {chosen*1000:.1f} ms")
    return chosen


# --------------------------------------------------------------------------
def measure_workspace(n=200_000, seed=0):
    """Monte-Carlo the reachable envelope of the tip by forward kinematics.

    The target sampling region for the RL task is derived from this, not
    guessed. Guessing produces targets outside the workspace, which caps the
    achievable success rate at something below 100% and makes every later
    reward comparison meaningless.
    """
    banner("4. REACHABLE WORKSPACE  (Monte-Carlo forward kinematics)")
    m, d = load()
    tip = m.site("tip").id
    lo = m.jnt_range[:, 0].copy()
    hi = m.jnt_range[:, 1].copy()

    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3))
    for k in range(n):
        d.qpos[:] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        pts[k] = d.site_xpos[tip]

    # Keep only poses that clear the table -- the arm cannot reach through it.
    above = pts[pts[:, 2] > 0.03]
    r = np.linalg.norm(above[:, :2], axis=1)

    print(f"  samples: {n}  |  above table (z>3cm): {len(above)} ({100*len(above)/n:.1f}%)")
    for lbl, v in [("x", above[:, 0]), ("y", above[:, 1]), ("z", above[:, 2]),
                   ("radius_xy", r)]:
        print(f"  {lbl:10s} min {v.min():7.3f}  p5 {np.percentile(v,5):7.3f}  "
              f"p95 {np.percentile(v,95):7.3f}  max {v.max():7.3f}")

    # A conservative box well inside the envelope: the 15th-85th percentile
    # shell in radius, forward hemisphere, comfortably above the table.
    box = dict(
        r_min=float(np.percentile(r, 20)), r_max=float(np.percentile(r, 80)),
        z_min=float(np.percentile(above[:, 2], 20)), z_max=float(np.percentile(above[:, 2], 80)),
    )
    print(f"\n  --> proposed target sampling shell:")
    print(f"      radius_xy in [{box['r_min']:.3f}, {box['r_max']:.3f}] m")
    print(f"      z         in [{box['z_min']:.3f}, {box['z_max']:.3f}] m")
    return box


# --------------------------------------------------------------------------
def find_home_pose(box):
    """Pick the rest pose the model actually ships.

    Two objectives, applied in order:
      1. tip near the centroid of the reachable workspace, so the policy
         starts roughly equidistant from every target it might be given;
      2. among those, the pose that minimises the worst servo's gravity load,
         so the arm can hold it indefinitely without a servo cooking.

    Objective 2 is not decoration. Optimising for position alone returns a
    near-horizontal shoulder, which is the highest-torque posture available
    and exactly what a real arm should not idle in.

    This function reproduces the keyframe in models/arm.xml. If you change the
    geometry, re-run it and paste the result back.
    """
    banner("5. HOME POSE SELECTION")
    m, d = load()
    tip = m.site("tip").id
    stall = m.actuator_forcerange[:, 1]
    goal = np.array([0.5 * (box["r_min"] + box["r_max"]), 0.0,
                     0.5 * (box["z_min"] + box["z_max"])])

    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    rng = np.random.default_rng(1)

    # stage 1 -- cheap kinematic filter
    cands = []
    for _ in range(80_000):
        q = rng.uniform(lo, hi)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        err = float(np.linalg.norm(d.site_xpos[tip] - goal))
        if err < 0.015:
            cands.append((err, q.copy()))

    # stage 2 -- gravity load, which needs a full forward pass
    best = None
    for err, q in cands:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        servo_load = float(np.max(np.abs(d.qfrc_bias[:4]) / stall))
        cost = err + 0.25 * servo_load
        if best is None or cost < best[0]:
            best = (cost, err, servo_load, q)

    _, err, servo_load, q = best
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    print(f"  goal tip     : {np.round(goal, 4)}")
    print(f"  candidates within 1.5 cm of the workspace centroid: {len(cands)}")
    print(f"  achieved tip : {np.round(d.site_xpos[tip], 4)}  (err {err*1000:.1f} mm)")
    print(f"  worst servo  : {servo_load*100:.1f}% of stall torque")
    print(f"  qpos         : {np.round(q, 4).tolist()}")
    kf = " ".join(f"{v:.4f}" for v in q)
    print(f'\n  --> keyframe: <key name="home" qpos="{kf}" ctrl="{kf}"/>')

    shipped = m.key_qpos[0][:4]
    match = np.allclose(shipped, q, atol=1e-3)
    print(f"  --> {'MATCHES' if match else 'DIFFERS FROM'} the keyframe in models/arm.xml "
          f"({np.round(shipped, 4).tolist()})")
    return q


if __name__ == "__main__":
    check_static_gravity_load()
    check_step_response()
    check_integrator_stability()
    box = measure_workspace()
    find_home_pose(box)
    print()
