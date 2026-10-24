"""ServoArmReach-v0 — a Gymnasium environment for the servoarm4 model.

Design notes live next to the code they justify. The three things this file
has to get right are the observation space, the action space and the reward,
so each gets an explicit section rather than being scattered through reset()
and step().
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "models", "reach_scene.xml")

# Datasheet slew rates. MG996R: 0.17 s / 60 deg -> 6.16 rad/s.
# SG90:   0.10 s / 60 deg -> 10.47 rad/s.
SLEW_RAD_S = np.array([6.16, 6.16, 6.16, 10.47], dtype=np.float64)


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardCfg:
    """Reward terms, versioned so every claim in docs/reward_iterations.md is
    reproducible by name rather than by 'the numbers I had that afternoon'.

    Terms:
      w_potential : potential-based shaping, gamma*phi(s') - phi(s) with
                    phi = -distance. Ng et al. (1999) proves this class of
                    shaping leaves the optimal policy unchanged, which is why
                    it is preferred here over a raw -distance penalty: it
                    accelerates learning without redefining the task.
      w_distance  : raw -distance. Kept configurable only so the docs can show
                    what it does wrong (it pays the agent to loiter near, but
                    not at, the target).
      w_success   : bonus while inside the tolerance ball.
      w_action    : penalises command magnitude (delta actions -> penalises
                    motion), for energy and for servo wear.
      w_jerk      : penalises change in command between steps. This is the
                    term that decides whether the policy is transferable: a
                    chattering command is fine in sim and destroys real gears.
      w_qvel      : penalises joint velocity, encourages settling at the goal
                    rather than orbiting it.
    """
    version: str = "v4_potential"
    w_potential: float = 12.0
    w_distance: float = 0.0
    w_success: float = 2.0
    w_action: float = 0.02
    w_jerk: float = 0.10
    w_qvel: float = 0.004
    # w_prox / prox_sigma: an extra bump INSIDE the potential,
    #     phi(d) = -d + w_prox * exp(-d / prox_sigma).
    # Pure -distance shaping has a constant gradient, so the pull toward the
    # goal is no stronger at 2 cm than at 20 cm -- yet the last 2 cm are where
    # the success bonus actually lives. Adding the bump to phi (rather than as
    # a standalone reward term) keeps the shaping potential-based, so the
    # optimal policy is still unchanged.
    w_prox: float = 0.0
    prox_sigma: float = 0.03
    sparse_only: bool = False


@dataclass
class DomainRandCfg:
    """Domain randomisation ranges, as multiplicative factors on the nominal
    model unless stated otherwise.

    Every range is tied to a real source of uncertainty rather than chosen to
    look thorough. The widths themselves are design assumptions, not
    measurements: the physical arm has not been built yet, and the nominal
    values are to be identified on it.
      mass      : link masses are estimates (assumed material density, fill
                  and fastener mass)
      damping   : gear grease viscosity varies strongly with temperature
      armature  : gearbox unit-to-unit variation
      kp        : servo control board tolerance and supply-voltage sag
      obs noise : hobby servo feedback pot is ~10-bit over 180 deg,
                  i.e. ~0.003 rad quantisation; 0.004 rad std is conservative
      latency   : microcontroller -> servo command path (assumed, not
                  measured), 0 or 1 control steps (0-50 ms)
    """
    enabled: bool = True
    mass: tuple = (0.85, 1.15)
    damping: tuple = (0.70, 1.30)
    armature: tuple = (0.80, 1.20)
    kp: tuple = (0.80, 1.20)
    obs_noise_qpos: float = 0.004      # rad
    obs_noise_qvel: float = 0.02       # rad/s
    max_action_latency: int = 1        # control steps


class ServoArmReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    # -- task constants ----------------------------------------------------
    CONTROL_HZ = 20                    # policy rate
    EPISODE_SECONDS = 4.0
    SUCCESS_RADIUS = 0.025             # m, ~ the tip pad diameter
    SETTLE_STEPS = 5                   # consecutive steps inside the ball

    def __init__(self, reward_cfg: RewardCfg | None = None,
                 dr_cfg: DomainRandCfg | None = None,
                 render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.rcfg = reward_cfg or RewardCfg()
        self.dcfg = dr_cfg or DomainRandCfg()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)

        dt = self.model.opt.timestep
        period = 1.0 / self.CONTROL_HZ
        # Control aliasing guard. The physics sweep in docs/physics_debugging.md
        # showed integration error jumping ~15x when the control period is not
        # an integer number of steps, for every integrator. Assert, don't hope.
        ratio = period / dt
        assert abs(ratio - round(ratio)) < 1e-9, (
            f"control period {period*1e3:.1f} ms is not an integer multiple of "
            f"timestep {dt*1e3:.1f} ms -- this aliases the command and inflates "
            f"integration error. Pick a compatible pair.")
        self.frame_skip = int(round(ratio))
        self.max_steps = int(self.EPISODE_SECONDS * self.CONTROL_HZ)

        self.tip_sid = self.model.site("tip").id
        self.target_bid = self.model.body("target").id
        self.mocap_id = self.model.body_mocapid[self.target_bid]
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Action = per-step change in servo setpoint, capped at what the servo
        # can travel in one control period at its datasheet slew rate. This makes an
        # unreachable command unrepresentable rather than merely discouraged.
        self.max_delta = SLEW_RAD_S * period

        # Nominal model parameters, kept for domain randomisation.
        self._nom = dict(
            body_mass=self.model.body_mass.copy(),
            body_inertia=self.model.body_inertia.copy(),
            dof_damping=self.model.dof_damping.copy(),
            dof_armature=self.model.dof_armature.copy(),
            actuator_gainprm=self.model.actuator_gainprm.copy(),
            actuator_biasprm=self.model.actuator_biasprm.copy(),
        )

        # ---- spaces -------------------------------------------------------
        # Observation (21):
        #   [0:4]   joint positions, normalised to [-1,1] over the joint range
        #   [4:8]   joint velocities / 10 rad/s
        #   [8:11]  tip position (m)
        #   [11:14] target position (m)
        #   [14:17] target - tip  (the term the policy actually needs; giving
        #           it explicitly rather than making the net subtract two
        #           near-equal vectors measurably speeds up early learning)
        #   [17:21] previous action
        # No absolute time is included: the policy must be time-invariant so
        # it keeps holding the target rather than learning the episode clock.
        # Finite bounds, not +/-inf: they document the expected scale of every
        # channel and turn an out-of-range observation into a visible bug.
        obs_hi = np.concatenate([
            np.full(4, 1.5),        # normalised joint pos (slack for obs noise)
            np.full(4, 3.0),        # joint vel / 10
            np.full(3, 0.5),        # tip position
            np.full(3, 0.5),        # target position
            np.full(3, 1.0),        # target - tip
            np.full(4, 1.0),        # previous action
        ])
        self.observation_space = spaces.Box(-obs_hi, obs_hi, (21,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)

        # A single reusable scratch state for forward-kinematics queries.
        # Allocating a fresh MjData per reset works, but it hands the garbage
        # collector a stream of native objects to free at arbitrary points in
        # the interpreter's life, which is a genuinely hard failure mode to
        # debug (see docs/physics_debugging.md, "the training segfault").
        # One long-lived scratch buffer removes the problem and is faster.
        self._scratch = mujoco.MjData(self.model)

        self._np_random = np.random.default_rng(seed)
        self._renderer = None
        self._reset_internal()

    # -- helpers -----------------------------------------------------------
    @property
    def _tip(self):
        return self.data.site_xpos[self.tip_sid].copy()

    @property
    def _target(self):
        return self.data.mocap_pos[self.mocap_id].copy()

    def _norm_q(self):
        lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        return 2.0 * (self.data.qpos[:4] - lo) / (hi - lo) - 1.0

    def _obs(self):
        q = self._norm_q()
        v = self.data.qvel[:4] / 10.0
        if self.dcfg.enabled:
            lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
            q = q + self._np_random.normal(0, self.dcfg.obs_noise_qpos, 4) * 2.0 / (hi - lo)
            v = v + self._np_random.normal(0, self.dcfg.obs_noise_qvel, 4) / 10.0
        tip, tgt = self._tip, self._target
        return np.concatenate([q, v, tip, tgt, tgt - tip, self._prev_action]).astype(np.float64)

    # -- target sampling ---------------------------------------------------
    def _sample_target(self):
        """Sample a target that is reachable BY CONSTRUCTION.

        A geometric box around the arm would contain unreachable points, which
        silently caps the achievable success rate and makes every reward
        comparison in the docs incomparable. Instead: sample a joint
        configuration inside the limits and take its forward kinematics. The
        target is then guaranteed to have at least one exact solution.
        """
        lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        scratch = self._scratch
        for _ in range(200):
            scratch.qpos[:4] = self._np_random.uniform(lo, hi)
            mujoco.mj_kinematics(self.model, scratch)
            p = scratch.site_xpos[self.tip_sid].copy()
            # keep it off the table and non-trivially far from the home pose
            if p[2] < 0.06:
                continue
            if np.linalg.norm(p - self._home_tip) < 0.06:
                continue
            if np.linalg.norm(p[:2]) < 0.04:      # not directly over the mast
                continue
            return p
        return self._home_tip + np.array([0.05, 0.0, 0.05])

    # -- domain randomisation ---------------------------------------------
    def _randomise(self):
        if not self.dcfg.enabled:
            return
        r, n = self._np_random, self._nom
        f_m = r.uniform(*self.dcfg.mass, size=self.model.nbody)
        self.model.body_mass[:] = n["body_mass"] * f_m
        self.model.body_inertia[:] = n["body_inertia"] * f_m[:, None]
        self.model.dof_damping[:] = n["dof_damping"] * r.uniform(*self.dcfg.damping, size=self.model.nv)
        self.model.dof_armature[:] = n["dof_armature"] * r.uniform(*self.dcfg.armature, size=self.model.nv)
        f_kp = r.uniform(*self.dcfg.kp, size=self.model.nu)
        # position actuator: gainprm[0] = kp, biasprm[1] = -kp
        self.model.actuator_gainprm[:, 0] = n["actuator_gainprm"][:, 0] * f_kp
        self.model.actuator_biasprm[:, 1] = n["actuator_biasprm"][:, 1] * f_kp
        self._latency = int(r.integers(0, self.dcfg.max_action_latency + 1))

    # -- gym API -----------------------------------------------------------
    def _reset_internal(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._home_tip = self._tip
        self._prev_action = np.zeros(self.model.nu)
        self._latency = 0
        self._pending = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        # restore nominal model before re-randomising, so factors never compound
        for k, v in self._nom.items():
            getattr(self.model, k)[:] = v
        self._randomise()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # small pose jitter so the policy cannot memorise one start state
        self.data.qpos[:4] += self._np_random.normal(0, 0.02, 4)
        self.data.qpos[:4] = np.clip(self.data.qpos[:4],
                                     self.model.jnt_range[:, 0],
                                     self.model.jnt_range[:, 1])
        self.data.qvel[:] = 0
        self.data.ctrl[:] = self.data.qpos[:4]
        mujoco.mj_forward(self.model, self.data)

        self._home_tip = self._tip
        self.data.mocap_pos[self.mocap_id] = self._sample_target()
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._inside_streak = 0
        self._prev_action = np.zeros(self.model.nu)
        self._pending = [np.zeros(self.model.nu)] * self._latency
        self._prev_dist = float(np.linalg.norm(self._target - self._tip))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # command latency: what the servo receives now was issued _latency
        # control steps ago
        self._pending.append(action)
        applied = self._pending.pop(0) if self._latency else action

        setpoint = np.clip(self.data.ctrl[:4] + applied * self.max_delta,
                           self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[:4] = setpoint
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        dist = float(np.linalg.norm(self._target - self._tip))
        inside = dist < self.SUCCESS_RADIUS
        self._inside_streak = self._inside_streak + 1 if inside else 0
        settled = self._inside_streak >= self.SETTLE_STEPS

        reward, terms = self._reward(action, dist, inside)

        self._prev_dist = dist
        self._prev_action = action
        self._step_count += 1

        truncated = self._step_count >= self.max_steps
        info = dict(distance=dist, is_success=bool(settled),
                    inside=bool(inside), **terms)
        return self._obs(), reward, False, truncated, info

    def _reward(self, action, dist, inside):
        c = self.rcfg
        if c.sparse_only:
            r = 1.0 if inside else 0.0
            return r, dict(r_sparse=r)

        gamma = 0.99

        def phi(d):
            p = -d
            if c.w_prox:
                p += c.w_prox * np.exp(-d / c.prox_sigma)
            return p

        # potential-based shaping: gamma*phi(s') - phi(s)
        shaping = c.w_potential * (gamma * phi(dist) - phi(self._prev_dist))
        r_dist = -c.w_distance * dist
        r_succ = c.w_success if inside else 0.0
        r_act = -c.w_action * float(np.sum(np.square(action)))
        r_jerk = -c.w_jerk * float(np.sum(np.square(action - self._prev_action)))
        r_qvel = -c.w_qvel * float(np.sum(np.square(self.data.qvel[:4])))

        total = shaping + r_dist + r_succ + r_act + r_jerk + r_qvel
        return total, dict(r_shaping=shaping, r_dist=r_dist, r_succ=r_succ,
                           r_act=r_act, r_jerk=r_jerk, r_qvel=r_qvel)

    # -- rendering ---------------------------------------------------------
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 720, 960)
        self._renderer.update_scene(self.data, camera="showcase")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
