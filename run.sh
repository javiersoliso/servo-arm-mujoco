#!/usr/bin/env bash
# Thin wrapper that pins the runtime environment for every entry point.
#   MUJOCO_GL=osmesa  -- software GL, the only backend that works headless here
#   OMP_NUM_THREADS=1 -- PPO on 4 DOF is latency-bound, not FLOP-bound; extra
#                        OpenMP threads only add contention on a small box
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER=1
# Use the virtualenv inside the repository (same as run.ps1 and the README).
exec "$(cd "$(dirname "$0")" && pwd)/venv/bin/python" -u "$@"
