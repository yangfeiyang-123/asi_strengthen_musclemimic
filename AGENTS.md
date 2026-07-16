# MuscleMimic training launch contract

These instructions apply to every local MuscleMimic training launch or restart.
Do not rely on an interactive shell having the required paths already exported.

## Canonical environment

Always launch from the repository root through
`scripts/run_fullbody_training.sh`.  Never invoke `python`, `.venv/bin/python`,
or `uv run fullbody/experiment.py` directly for a production run.

The launcher must provide all of the following:

- `source configs/env.sh`, which binds the repository datasets, local AMASS
  paths, local GMR cache, SMPL model path, and CUDA compatibility directory;
- one explicit physical GPU through `CUDA_VISIBLE_DEVICES`;
- `MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4` and
  `MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4` unless deliberately overridden;
- a task-specific `JAX_COMPILATION_CACHE_DIR` under
  `/data3/yangfeiyang/WorkSpace/ENV/jax-cache`;
- `scripts/run_with_cuda_compat.sh uv run`, not a direct Python invocation;
- an append-only combined stdout/stderr log via `tee -a`.

Required launch variables:

```bash
export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
export MUSCLEMIMIC_JAX_CACHE_KEY=<stable_task_cache_key>
export MUSCLEMIMIC_TRAIN_LOG=<absolute_or_repo_relative_log_path>
```

Canonical command:

```bash
scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/<stage>/<config> \
  wandb.mode=online
```

For a non-launching environment check, add `MUSCLEMIMIC_DRY_RUN=1`.

For the ChinaJump root-control 640M run, the exact values are:

```bash
export CUDA_VISIBLE_DEVICES=2
export MUSCLEMIMIC_JAX_CACHE_KEY=chinajump_stage1
export MUSCLEMIMIC_TRAIN_LOG=datasets/ChinaJump/training/logs/chinajump_root_control_v2_stage1_body_640m.log
scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  wandb.mode=online
```

## tmux

Use a named session on an explicit socket.  For ChinaJump:

```bash
tmux -S /data3/yangfeiyang/tmp/tmux_chinajump.sock new-session -d \
  -s chinajump_root_v2_640m \
  -c /data3/yangfeiyang/WorkSpace/musclemimic
```

Then send the canonical environment exports and launcher command to that pane.
Do not reuse a pane reported as `Pane is dead`; create a new named session.

## Mandatory pre-flight and launch verification

Before launch:

1. Run the focused config/reward/terminal tests.
2. Resolve the Hydra config and verify the requested `total_timesteps`, unique
   `run_id`, reward weights, terminal thresholds, and promotion behavior.
3. Check GPU processes with `nvidia-smi`; do not confuse CUDA-visible index 0
   inside the process with the physical GPU selected by `CUDA_VISIBLE_DEVICES`.
4. For a changed reward or termination contract, use a new run id and a fresh
   optimizer.  Never resume an incompatible checkpoint.

After launch, do not report success until all of these are true:

1. The log says every training and validation trajectory was loaded as an
   existing local retargeted file.  A Hugging Face download attempt indicates
   that `configs/env.sh` was not loaded and the run must be stopped.
2. The checkpoint run manifest exists and records the intended config hash,
   `run_id`, `total_timesteps`, promotion behavior, rewards, and terminal limits.
3. W&B has a live run id and URL.
4. `nvidia-smi` shows the new Python PID on the intended physical GPU.
5. The log reaches `Starting training...` and has no fatal traceback.

When stopping a run, send one Ctrl-C through its tmux pane, wait for the Python
PID and CUDA context to disappear, and preserve the latest finalized checkpoint.
