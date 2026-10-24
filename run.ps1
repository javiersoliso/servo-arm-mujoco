# Windows entry point. Mirrors run.sh.
#
#   .\run.ps1 tests\validate_physics.py
#   .\run.ps1 train\evaluate.py --run runs\finetune_smooth
#
# MUJOCO_GL is deliberately NOT set here. On Windows the default backend uses
# the real GPU and works. The osmesa setting in run.sh exists only because the
# machine this was developed on is a headless Linux container with no display
# and no GPU -- software rendering was the only option there, and forcing it
# on Windows would make rendering far slower for no reason.
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
$env:PYTHONFAULTHANDLER = "1"
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Error "No virtualenv found at $py. See docs\windows_setup.md first."
  exit 1
}
& $py -u @args
