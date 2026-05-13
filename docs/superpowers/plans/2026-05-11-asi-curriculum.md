# ASI / Curriculum Implementation Plan

## Goal

Add SFV-style adaptive start-state initialization (ASI) as an explicit PPO training switch. When `asi.enabled: false` or the key is absent, the existing mimic PPO behavior must remain unchanged, including trajectory-reset sampling and RNG use.

## Constraints

- ASI must be opt-in.
- Existing adaptive trajectory sampling and curriculum code must keep working.
- Reset-time ASI state must be stored in carry/trajectory state so rollout transitions can attribute episode outcomes to the chosen start bucket.
- Tests must cover the disabled path before integration is considered complete.

## Tasks

1. Add focused unit tests for ASI probability computation and update behavior.
2. Add reset-path tests proving disabled ASI matches the old random trajectory/start-frame formula.
3. Implement pure ASI helpers under `musclemimic/algorithms/common`.
4. Extend `TrajState` and `TrajectoryHandler.reset_state()` with optional ASI bucket sampling while preserving the old disabled branch.
5. Add carry/env-state update helpers for ASI frame probabilities and minimum remaining steps.
6. Wire PPO runner config:
   - `asi.enabled`
   - `num_buckets`
   - `uniform_mix`
   - `temperature`
   - `alpha`
   - `baseline_beta`
   - `logit_clip`
   - `min_remaining_steps`
   - `early_termination_penalty`
7. Add default config blocks with `enabled: false`.
8. Run targeted tests and report any remaining implementation risks.

## Validation

- `pytest tests/unit/test_asi.py`
- `pytest tests/unit/test_trajectory_handler_asi.py`
- If practical, run existing adaptive sampling tests to catch interaction regressions.
