# Distillation Review Round 3 TODO

Source review:
`doc/修改建议/蒸馏策略第三轮修改意见/README (3).md`

## Completed in this round

- [x] Include `BadmintonMimic*` in setuptools package discovery.
- [x] Remove or wire previously misleading Forehand wrapper arguments.
- [x] Switch subprocess commands to `python -m ...` module execution.
- [x] Add BC warm-start through `--init_ckpt` and `--resume-student`.
- [x] Document current `freeze_run_stats` persistence semantics.
- [x] Add `--motion_path` support to generic teacher and DAgger collection.
- [x] Fail evaluation parsing when required metrics are missing.
- [x] Add `musclemimic-distill-inspect-dataset`.
- [x] Add ForehandClear smoke-test runbook.

## Still Requires Real Checkpoints

- [ ] Run the smoke runbook with the actual ForehandClear teacher checkpoint.
- [ ] Confirm `fullbody/eval.py` prints all required rollout metrics.
- [ ] Verify warm-started DAgger BC improves closed-loop metrics.
