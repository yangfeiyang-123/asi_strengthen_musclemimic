# Racket-Holding Trajectory Imitation Env (`MjxMyoFullBodyRacket`)

A GPU/JAX(MJX)-trainable environment where the muscle-actuated MyoFullBody humanoid
rigidly holds a badminton racket while imitating retargeted swing trajectories. This
is the "hold-racket" counterpart of the free-hand Stage-1 trajectory imitation, and a
bridge to the downstream hitting / residual-grip curriculum.

## What it is

- `musclemimic/environments/humanoids/myofullbody_racket.py`
  - `MyoFullBodyRacket` (CPU) and `MjxMyoFullBodyRacket` (jax/warp), registered under
    those names.
  - The rigid racket asset (`environment/racket/assets/badminton_racket_rigid.xml`) is
    attached as a **jointless child body** of the right-hand palm body `thirdmc_r`
    (the Overall scene's hand-racket weld `body1`). Its free joint is removed.
  - **Zero added DOF**: `qpos`/`qvel`/`nu` are identical to `MyoFullBody(disable_fingers=True)`
    (89 / 88 / 354). Retargeted free-hand trajectories, observation/action spaces, and
    trained body policies all transfer unchanged; the muscles now carry the racket's
    0.09 kg mass/inertia while tracking the swing.
  - Racket collision geoms are moved to an isolated collision bit (4), so the racket
    never contacts the body. `mjx.put_model` succeeds on both `jax` and `warp`.

- Retargeting reuse: the racket env declares `retarget_as = "MyoFullBody"`. The GMR
  pipeline (`loco_mujoco.smpl.retargeting._resolve_retarget_env_name`) maps it to the
  MyoFullBody robot conf and `gmr_cache/MyoFullBody/gmr` cache, so it reuses the
  existing retargeted clips with no re-retargeting and no new `*.yaml` robot conf.

- Config: `fullbody/config_specific_task/conf_fullbody_badminton_racket_gmr.yaml`
  inherits `conf_fullbody_badminton_gmr` and only swaps `env_name: MjxMyoFullBodyRacket`.

## Why rigid (matches the downstream hitting/residual modules)

The downstream `soft_weld_schedule` starts at `strong_weld` (weld_strength 1.0,
solref 0.002), which is physically ~rigid. The rigid hold is exactly that starting
condition, so a body policy pretrained here drops into the Overall hitting scene at
its first curriculum stage. Keeping obs/action/qpos identical to the free-hand env
preserves checkpoint compatibility with the existing forehand-clear runner and the
downstream `LatentBodyPolicy` (body-muscle only). The free-body + soft-weld
representation is only needed once the curriculum weakens the weld, and that scene
(`environment/overall_environment`) already provides it.

## Run GPU training

```bash
source BadmintonMimic/configs/env.sh      # cleans LD_LIBRARY_PATH + mounts cuda-compat-12.4 (warp)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_racket_gmr
```

Quick GPU smoke (KIT locomotion clips already cached, tiny + no validation/wandb):

```bash
source BadmintonMimic/configs/env.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python fullbody/experiment.py \
  --config-name=conf_fullbody_gmr \
  experiment.env_params.env_name=MjxMyoFullBodyRacket \
  experiment.env_params.mjx_backend=warp \
  experiment.env_params.num_envs=64 experiment.num_envs=64 \
  experiment.total_timesteps=80000 experiment.ppo_config.num_steps=20 \
  experiment.ppo_config.num_minibatches=4 \
  experiment.validation.active=false experiment.validation.num=1 \
  experiment.save_checkpoints=false experiment.auto_resume=false \
  wandb.mode=disabled
```

## Tests

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_myofullbody_racket.py -q
```

Covers registration, dim-parity with `MyoFullBody`, racket parented to `thirdmc_r`
(mass 0.09), reset/step finiteness, zero racket-body contact pairs, `enable_racket=False`
fallback, `mjx.put_model` (jax), and the `retarget_as` alias.

## Knobs (env kwargs)

- `enable_racket` (default `True`) — drop the racket to recover plain MyoFullBody.
- `racket_attach_body` (default `thirdmc_r`).
- `racket_grip_pos` / `racket_grip_quat` — racket butt pose in the attach-body local
  frame; defaults are derived from `configs/right_hand_racket_grip_reference.json` so the
  grip pose matches the Overall hitting scene. Override to refine the grip.
- `racket_collision_bit` (default `4`) — disjoint from the human body's bit 1.
- `racket_xml_path` — alternate racket asset.
