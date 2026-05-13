# ASI / Curriculum Training Commands

Run from the repository root:

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
```

## Fullbody Demo

The three modes below are progressive:

1. **Baseline**: original mimic PPO. ASI is disabled, so reset/start-state sampling follows the original code path.
2. **ASI only**: keeps the original PPO objective and curriculum settings, but adaptively changes the trajectory start frame distribution.
3. **ASI + curriculum**: builds on ASI by also adapting the early-termination threshold and reward weights during training.

Original mimic PPO baseline, ASI disabled:

```bash
wandb.mode=disabled \
.venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_demo \
  experiment.asi.enabled=false
```

ASI only:

```bash
wandb.mode=disabled \
.venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_demo \
  experiment.asi.enabled=true
```

ASI + adaptive termination curriculum + reward curriculum:

```bash
wandb.mode=disabled \
.venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_demo \
  experiment.asi.enabled=true \
  experiment.adaptive_termination.enabled=true \
  experiment.reward_curriculum.enabled=true
```

## Fullbody GMR / Larger Run

```bash
.venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet \
  experiment.asi.enabled=true \
  experiment.adaptive_termination.enabled=true \
  experiment.reward_curriculum.enabled=true
```

## Notes

- `experiment.asi.enabled=false` is the default and keeps the original mimic PPO reset path.
- Remove `wandb.mode=disabled` if you want online Weights & Biases logging.
- If using Warp/GPU and CUDA compatibility scripts are needed, prefix the command with:

```bash
scripts/run_with_cuda_compat.sh
```
