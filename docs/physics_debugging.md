# Physics and infrastructure debugging notes

Four problems that cost real time on this project, what the symptom looked
like, and what the cause turned out to be. Three of the four had a first
diagnosis that was wrong, which is the part worth reading.

---

## 1. The timestep limit is set by accuracy, not stability

**What I assumed.** I wrote in `models/arm.xml` that an explicit Euler
integrator would need `timestep <= 5e-4` to stay stable with this arm's
armature values, and picked `implicitfast` on that basis. This is the standard
folklore for position-controlled arms and I did not check it.

**What the measurement said.** `tests/validate_physics.py` drives the arm with
a 20 Hz random square wave — the command profile an untrained policy actually
emits, which is far harsher than any scripted trajectory — and looks for
divergence:

| integrator | 0.5 ms | 1 ms | 2 ms | 4 ms | 8 ms |
|---|---|---|---|---|---|
| Euler | ok | ok | ok | ok | ok |
| implicitfast | ok | ok | ok | ok | ok |
| RK4 | ok | ok | ok | ok | ok |

Nothing diverges anywhere. The claim in the model file was simply false. This
arm is small, well damped and has substantial armature, so it is not a stiff
system and stability is not the binding constraint.

**What the constraint actually is.** Integration *error*, measured as maximum
tip-position deviation from an RK4 / 0.1 ms reference over the same rollout:

| dt (ms) | 50 ms / dt | aligned? | Euler | implicitfast | RK4 |
|---|---|---|---|---|---|
| 0.5 | 100 | yes | 0.84 mm | 0.79 mm | **0.08 mm** |
| 1.0 | 50 | yes | 1.68 mm | 1.54 mm | 0.12 mm |
| 2.0 | 25 | yes | 3.36 mm | 3.05 mm | **0.08 mm** |
| 2.5 | 20 | yes | 4.24 mm | 3.73 mm | 0.20 mm |
| 4.0 | 12.5 | **no** | 16.20 mm | 15.93 mm | 12.67 mm |
| 5.0 | 10 | yes | 8.34 mm | 7.28 mm | 1.22 mm |
| 8.0 | 6.25 | **no** | 25.08 mm | 21.73 mm | 17.90 mm |
| 10.0 | 5 | yes | 16.84 mm | 14.83 mm | 0.47 mm |

Euler and implicitfast are first order: their error doubles as dt doubles,
exactly as expected (0.84 → 1.68 → 3.36).

## 2. The control-aliasing trap

Look at the RK4 column again. **8 ms gives 17.90 mm and 10 ms gives 0.47 mm** —
a *larger* timestep with 38x less error. That is not a numerical-analysis
result; it is a bug in the experiment that turned out to be a real design
constraint.

The command is held for 50 ms (20 Hz). When `50 ms / dt` is not an integer,
the command changes *inside* a step and the rollout stops being comparable to
the reference. Every non-integer ratio in the table (4 ms, 8 ms) is bad across
all three integrators; every integer ratio is fine. The integrator was never
the variable.

The practical consequence is a constraint on the environment, not on the
model: **the control period must be an integer multiple of the timestep.**
`ServoArmReachEnv.__init__` asserts it rather than trusting whoever edits the
constants next:

```python
ratio = period / dt
assert abs(ratio - round(ratio)) < 1e-9, (
    f"control period {period*1e3:.1f} ms is not an integer multiple of "
    f"timestep {dt*1e3:.1f} ms -- this aliases the command ...")
```

**Choosing the integrator.** With stability off the table, the decision is
accuracy per unit compute (2 vCPU, wall-clock per simulated second):

| integrator | dt | speed | error |
|---|---|---|---|
| Euler | 0.5 ms | 80x realtime | 0.84 mm |
| implicitfast | 0.5 ms | 78x realtime | 0.79 mm |
| **RK4** | **2.0 ms** | **93x realtime** | **0.08 mm** |

RK4 at 2 ms is both faster *and* ten times more accurate than first order at
0.5 ms — one 4th-order step buys more than four 1st-order steps. Shipped:
`RK4 @ 2 ms`, control at 20 Hz, `frame_skip = 25`.

---

## 3. The training segfault

**Symptom.** Every training run died with `SIGSEGV` inside libc `free()`. The
Python traceback pointed at `torch.optim.Adam.__init__` → `torch._compile` →
importing `torch._dynamo`, i.e. squarely at PyTorch, which is where I wasted
the most time.

**Why the traceback was misleading.** A heap-corruption crash surfaces
wherever the allocator next touches the corrupted arena, which has nothing to
do with who corrupted it. `free()` in the traceback means "someone scribbled
on the heap earlier", not "the optimiser is broken".

**Bisection.** The reproducer collapsed to two lines that never touch this
project's code at all:

```python
import mujoco
from stable_baselines3 import PPO
PPO("MlpPolicy", "CartPole-v1")     # SIGSEGV
```

Removing the `import mujoco` line made it pass. So the environment code was
innocent — worth establishing early, because I had been rewriting env internals
on the assumption that the bug was mine.

**Three wrong fixes, and why each looked right.** `LD_PRELOAD`-ing the system
`libgomp`, and separately downgrading `mujoco` 3.12 → 3.3.5, each made the
reproducer pass. Both were false positives: **the crash is nondeterministic**,
so a single passing run carries almost no information. The same command that
had just passed failed on the next invocation.

**Actual cause.** Installing the CUDA build of torch leaves `nvidia-*` and
`cuda-*` packages behind. A later CPU-only torch still loads ~49
`cuda.bindings` modules from them, and in a process that also holds MuJoCo
that combination corrupts the heap. On a GPU-less box none of it is doing
anything useful in the first place.

**Fix.** A clean virtualenv containing only CPU wheels, so no CUDA package is
present to load (`requirements.txt`). Verified over **10 consecutive trials,
10/10 pass, 0 cuda modules loaded** — the verification protocol matters as
much as the fix, since a 1-in-3 flake passes a single check a third of the
time.

**The transferable lesson:** for a nondeterministic failure, "I ran it and it
worked" is not evidence. Decide the trial count *before* testing the fix.

---

## 4. Model validation, before any learning

`tests/validate_physics.py` runs four checks that all have to pass before RL
is worth starting. Every number in this section is simulated with the nominal
model; the physical arm has not been built yet. Each check catches a failure
that would otherwise appear hours later disguised as "PPO won't learn".

**Static gravity load** — worst case, arm fully extended and horizontal:

| joint | required | stall | margin |
|---|---|---|---|
| shoulder | 0.238 N·m | 0.980 N·m | 4.1x |
| elbow | 0.082 N·m | 0.980 N·m | 12.0x |
| wrist | 0.011 N·m | 0.180 N·m | 16.9x |

If a servo cannot hold the arm up, no policy can either, and the sim would be
teaching a behaviour the hardware cannot reproduce.

**Step response** — measured per joint on a 0.5 rad command step:

| joint | rise (s) | overshoot | settle (s) | steady-state error |
|---|---|---|---|---|
| base_yaw | 0.184 | 0.0% | 0.238 | 0.0002 rad |
| shoulder | 0.146 | 0.0% | 0.186 | **0.0332 rad (1.90°)** |
| elbow | 0.182 | 0.9% | 0.244 | −0.0044 rad |
| wrist | 0.232 | 0.8% | 0.292 | −0.0038 rad |

The shoulder's steady-state error is the number that matters. A position
actuator with finite `kp` under gravity does *not* reach its setpoint, so the
policy's action is a biased **request**, not a joint angle — any reward or
observation that assumes `action == angle` is wrong. Confirmed at the tip:
commanding the home pose and holding for 4 s settles **3.7 mm** below the
commanded position, with residual velocity 2.3e-4 rad/s (no drift, it is a
static offset rather than a slow fall).

**Reachable workspace** — 200k Monte-Carlo forward-kinematics samples; 97.1%
clear the table, maximum reach 0.255 m, maximum height 0.335 m. This is what
target sampling is derived from. Sampling targets from a hand-drawn geometric
box instead would include unreachable points, silently capping the maximum
achievable success rate and making every reward comparison in
`reward_iterations.md` incomparable. The environment therefore samples a joint
configuration and takes its forward kinematics, so every target has at least
one exact solution *by construction*.

**Home pose** — chosen, not guessed: of the 51 sampled configurations whose
tip lands within 1.5 cm of the workspace centroid, the one minimising
worst-servo gravity load. Result: tip at (0.110, 0.001, 0.215) m with the
most-loaded servo at **15.1% of stall**, so the arm can hold it indefinitely.
Optimising for tip position alone returns a near-horizontal shoulder — the
highest-torque posture available, and exactly what a real arm should not idle
in. `find_home_pose()` asserts that its result matches the keyframe actually
shipped in `models/arm.xml`, so the model and the script that justifies it
cannot drift apart silently.
