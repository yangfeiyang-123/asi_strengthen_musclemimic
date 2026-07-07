# Forehand Net Lift PostTrain Interface Design

## Goal

Create a reusable PostTrain experiment interface for badminton skills, starting with ForehandNetLift. The interface should let a user define one YAML experiment spec, generate reproducible Hydra configs, and run or dry-run prepare/train/eval/render stages without hand-editing fullbody configs for every action.

## Recommended Approach

Use a spec-driven runner instead of adding another one-off shell script. The runner reads a YAML spec with action metadata, train/validation/stress motions, resume checkpoint, training defaults, evaluation defaults, and experiment arms. `prepare` materializes Hydra configs and a report. `train`, `eval`, and `render` build commands from those generated artifacts and only execute them when `--execute` is supplied.

This keeps the first version safe: it gives exact commands and reproducible configs, but does not accidentally launch a long GPU training job.

## Components

- `experiments/posttrain/forehand_net_lift_v1.yaml`: default ForehandNetLift PostTrain spec.
- `musclemimic/badminton/scripts/run_posttrain_experiment.py`: reusable runner for `prepare`, `train`, `eval`, `render`, and `all`.
- `tests/unit/test_run_posttrain_experiment.py`: unit tests for spec loading, config generation, and command construction.

## Data Flow

1. Load spec YAML.
2. Normalize motion paths by removing a trailing `.npz`.
3. For each posttrain arm, generate a Hydra config under:
   - `outputs/posttrain/<Action>/<experiment_id>/configs/<arm>.yaml`
   - `fullbody/config_specific_task/posttrain/<Action>/<experiment_id>/<arm>.yaml`
4. Generate train/eval/render command files and a short Markdown report under the output directory.
5. Execute commands only when the user passes `--execute`.

## ForehandNetLift v1 Arms

- `E0_baseline`: evaluate the current best baseline checkpoint only.
- `E1_root_hand_focus`: root forward tracking plus right-hand endpoint focus.
- `E2_root_hand_foot`: E1 plus stricter foot/root stability.
- `E3_smooth_control`: E2 plus lighter action entropy and smoother control.

## Success Criteria

- A future action can reuse the same runner by copying one YAML spec and changing `action`, `resume_from`, reference motions, and arms.
- `prepare` is deterministic and does not require GPU execution.
- `train/eval/render` can be dry-run first and executed explicitly later.
- Unit tests cover config generation and command construction.

## Self-Review

No placeholders remain. The scope is intentionally limited to a safe experiment interface and does not include launching full long-running experiments automatically.
