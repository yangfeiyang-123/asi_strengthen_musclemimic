# Distillation Review Round 1 TODO

Source review:
`doc/修改建议/蒸馏策略一轮修改建议/README (1).md`

## Completed in this round

- [x] Add schema version to distillation shard metadata.
- [x] Make `drop_goal_lookahead` an explicit student obs filter option.
- [x] Validate BC dataset `student_obs_dim` against the configured student env.
- [x] Freeze `run_stats` during BC evaluation loss.
- [x] Use `step_with_transition()` in DAgger collection.
- [x] Add official ForehandClear student configs for PPO fine-tune and BC eval.
- [x] Add iterative DAgger orchestration CLI.
- [x] Add teacher-vs-student Markdown report generation.
- [x] Add console script entrypoints for distillation commands.
- [x] Update command docs and acceptance criteria.

## Runtime workflow

1. Collect teacher data with `fullbody/distill_collect.py`.
2. Train BC with `fullbody/distill_train_bc.py`.
3. Run iterative DAgger with `fullbody/distill_run_dagger.py`.
4. Fine-tune with `fullbody/experiment.py --config-name=config_specific_task/conf_fullbody_badminton_student_gmr`.
5. Compare with `fullbody/distill_compare.py` or `BadmintonMimic/scripts/evaluate_teacher_student_distill.py`.

## Remaining experiment work

- [ ] Run the workflow on the real ForehandClear teacher checkpoint.
- [ ] Fill `comparison_metrics.json`, `comparison_table.csv`, and `summary.md` with real metrics.
- [ ] Tune acceptance thresholds after measuring teacher baselines on the target clips.
