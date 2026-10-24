# Included run artefacts

`finetune_smooth/` is the shipped policy: weights, the observation-normalisation
statistics it was trained under (required -- loading the policy without them
feeds it inputs on the wrong scale), and its evaluation output. It can be
evaluated or rendered without retraining anything:

    ./run.sh train/evaluate.py --run runs/finetune_smooth
    ./run.sh train/render_rollout.py --run runs/finetune_smooth

`study/` holds the summary JSON for every run behind the tables in
docs/reward_iterations.md -- 6 variants x 2 seeds at 200k steps, plus the two
800k runs. Weights are omitted for size; the curves and final metrics are all
there.
