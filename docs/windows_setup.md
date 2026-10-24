# Windows setup

The project was developed on a headless Linux container. Nothing in it is
Linux-specific, but two things change on Windows.

## 1. Use a fresh venv, NOT your Anaconda base environment

```powershell
cd <this folder>
python -m venv venv
venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
venv\Scripts\pip install -r requirements.txt
```

This is not tidiness. The hardest bug in this project (written up in
`docs/physics_debugging.md`, "the training segfault") was caused by leftover
`nvidia-*` / `cuda-*` packages from a previous CUDA PyTorch install. A later
CPU-only PyTorch still loaded ~49 `cuda.bindings` modules from them, and in a
process that also held MuJoCo that corrupted the heap. The crash landed inside
libc `free()` with a traceback pointing at PyTorch, and it was
**nondeterministic** -- so it passed roughly one run in three, which is worse
than failing every time.

A long-lived Anaconda base environment is exactly where that kind of residue
accumulates. Install into the clean venv above.

If you do have an NVIDIA GPU and want the CUDA build of PyTorch, that is fine
-- just install it into an environment that has never held a mismatched set,
and verify with:

```powershell
venv\Scripts\python -c "import torch, sys; print(len([m for m in sys.modules if m.startswith('cuda')]))"
```

Run it after `import torch`. On a healthy CPU-only setup it prints `0`.

## 2. Rendering

Leave `MUJOCO_GL` unset. Windows will use the native GL backend and your GPU.

`run.sh` sets `MUJOCO_GL=osmesa` (software rendering) because the development
container had no display and no GPU; that was the only backend that worked
there, and it is much slower. `run.ps1` does not set it.

## Running things

```powershell
.\run.ps1 tests\validate_physics.py
.\run.ps1 tests\scripted_expert.py
.\run.ps1 train\evaluate.py --run runs\finetune_smooth
.\run.ps1 train\render_rollout.py --run runs\finetune_smooth
```

The last two need no training -- the trained policy ships in
`runs\finetune_smooth\`.

## Interactive viewer

The repo only renders offscreen to video, because the container had no
display. On Windows you can open the model in MuJoCo's interactive viewer,
which is by far the fastest way to build intuition for what the MJCF is
describing -- drag the joints, perturb bodies, watch the contacts:

```powershell
venv\Scripts\python -m mujoco.viewer --mjcf=models\reach_scene.xml
```
