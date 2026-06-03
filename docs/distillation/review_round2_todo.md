# Distillation Review Round 2 TODO

Source review:
`doc/修改建议/蒸馏策略二轮修改意见/README (2).md`

## Completed in this round

- [x] Force teacher collection rollout to use full lookahead observations.
- [x] Add `collector_obs_mode: teacher_full_obs` metadata.
- [x] Add `freeze_run_stats` support to teacher and DAgger collection.
- [x] Add StudentObsContainer compatibility methods.
- [x] Save DAgger `rollout_action` and `used_teacher_action`.
- [x] Save teacher log-prob diagnostics for student and rollout actions.
- [x] Lazy-export public distillation APIs without reintroducing circular imports.
- [x] Add split shard writer for `train_*.npz` and `val_*.npz`.
- [x] Add diagonal Gaussian KL loss support for BC/KD.
- [x] Add ForehandClear task scripts and observation diagnostic tool.
- [x] Add tests for the above code paths.

## Still Requires Real Checkpoints

- [ ] Run teacher dataset smoke collection on the real ForehandClear teacher.
- [ ] Train a small BC checkpoint and reload it with `fullbody/eval.py`.
- [ ] Run DAgger collection on student rollout states.
- [ ] Run PPO fine-tune smoke with `conf_fullbody_badminton_student_gmr`.
- [ ] Fill teacher-vs-student metrics artifacts from real rollout evaluation.
