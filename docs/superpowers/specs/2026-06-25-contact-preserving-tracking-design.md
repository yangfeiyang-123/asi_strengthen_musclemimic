# Contact-Preserving Tracking Training Mode Design

## Overview

Extend the existing MimicReward to support foot contact tracking rewards, enabling ground-contact-aware RL training using reference bundles from optimized_wham. Includes a curriculum scheduler that progressively introduces contact reward terms.

## Architecture

```
reference_bundle (optimized_wham)
    │
    ├─ motion.npz ──── materialize ──→ AMASS npz ──→ GMR retarget ──→ Trajectory (qpos/qvel/sites)
    │                                                                        │
    └─ contact_schedule.npz ──→ ContactTrackingData ─────────────────────────┤
         (stance_mask, foot_points, body_laplacian)                          │
                                                                             ▼
                                                            MimicReward.__call__()
                                                                ├─ existing: rpos, rquat, qpos, qvel...
                                                                └─ new: foot_height, foot_velocity, body_graph
                                                                             │
                                                                    CurriculumScheduler
                                                                    (update count → stage → weights)
```

## Component Design

### 1. ContactTrackingData — loads contact schedule alongside trajectory

**File**: `musclemimic/badminton/asi/contact_tracking_data.py` (new)

```python
@dataclass(frozen=True)
class ContactTrackingData:
    stance_mask: np.ndarray       # (T, K) bool — which foot points are in stance
    foot_points: np.ndarray       # (T, K, 3) float32 — reference foot point positions
    body_laplacian: np.ndarray | None  # (T, J, 3) float32 — body graph laplacian coords
    foot_labels: list[str]
    reference_fps: float
    control_dt: float
    effective_ref_stride: float

    def frame_at_traj_step(self, traj_step: int) -> int:
        """Map trajectory step index to reference frame index."""
        frame = int(round(traj_step * self.effective_ref_stride))
        return min(frame, len(self.stance_mask) - 1)
```

**Loading**: From tracking cache npz or directly from reference bundle.

**Lifecycle**: Created once at env init, stored on the MimicReward instance. Indexed by trajectory step during reward computation.

### 2. MimicReward extensions

**File**: `musclemimic/core/reward/trajectory_based.py` (modify existing)

New kwargs in `__init__`:
```python
# Contact tracking weights (all default 0.0 → disabled unless configured)
self._foot_contact_height_w_sum = kwargs.get("foot_contact_height_w_sum", 0.0)
self._foot_contact_height_w_exp = kwargs.get("foot_contact_height_w_exp", 80.0)
self._foot_contact_velocity_w_sum = kwargs.get("foot_contact_velocity_w_sum", 0.0)
self._foot_contact_velocity_w_exp = kwargs.get("foot_contact_velocity_w_exp", 8.0)
self._body_graph_w_sum = kwargs.get("body_graph_w_sum", 0.0)
self._body_graph_w_exp = kwargs.get("body_graph_w_exp", 20.0)

# Contact data reference (set externally after trajectory loads)
self._contact_tracking_data: ContactTrackingData | None = None
```

New block in `__call__` (after existing reward calculations, before total_reward):
```python
# Contact tracking rewards (only when contact data is available)
foot_height_reward = 0.0
foot_velocity_reward = 0.0
body_graph_reward = 0.0

if self._contact_tracking_data is not None:
    ctd = self._contact_tracking_data
    ref_frame = ctd.frame_at_traj_step(carry.traj_state.subtraj_step_no)

    # Foot contact height: stance feet should match reference height
    stance = ctd.stance_mask[ref_frame]  # (K,) bool
    if backend.any(stance):
        # Get actual foot positions from simulation
        # (foot sites added to env spec, IDs stored during init)
        actual_feet_z = data.site_xpos[self._foot_site_ids, 2]  # Z-up
        ref_feet_z = ctd.foot_points[ref_frame, :, 2]
        height_err = backend.mean(backend.abs(actual_feet_z[stance] - ref_feet_z[stance]))
        foot_height_reward = backend.exp(-self._foot_contact_height_w_exp * height_err)

    # Foot contact velocity: stance feet should be stationary
    if backend.any(stance) and ref_frame > 0:
        prev_feet = data.site_xpos[self._foot_site_ids]  # current
        # velocity approximated from consecutive frames
        foot_vel = backend.linalg.norm(prev_feet - self._last_foot_pos, axis=-1)
        stance_vel = backend.mean(foot_vel[stance])
        foot_velocity_reward = backend.exp(-self._foot_contact_velocity_w_exp * stance_vel)

    # Body graph laplacian error
    if ctd.body_laplacian is not None and self._body_graph_w_sum > 0:
        ref_lap = ctd.body_laplacian[ref_frame]
        actual_lap = _compute_body_laplacian(data, self._body_keypoint_ids, self._body_graph)
        graph_err = backend.mean(backend.linalg.norm(actual_lap - ref_lap, axis=-1))
        body_graph_reward = backend.exp(-self._body_graph_w_exp * graph_err)
```

Add to total_reward aggregation:
```python
contact_reward = (
    carry.foot_contact_height_w_sum * foot_height_reward
    + carry.foot_contact_velocity_w_sum * foot_velocity_reward
    + carry.body_graph_w_sum * body_graph_reward
)
total_reward = total_reward + contact_reward
```

Using `carry.*_w_sum` (not `self._*_w_sum`) allows the curriculum to adjust weights dynamically.

### 3. CurriculumScheduler — update-based weight scheduling

**File**: `musclemimic/badminton/asi/contact_curriculum.py` (new)

```python
@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_update: int
    foot_contact_height_w: float
    foot_contact_velocity_w: float
    body_graph_w: float

def default_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage("body_only", 0, 0.0, 0.0, 0.0),
        CurriculumStage("joint_tracking", 2000, 0.0, 0.0, 0.0),
        CurriculumStage("contact_intro", 5000, 0.20, 0.20, 0.10),
        CurriculumStage("contact_full", 10000, 0.45, 0.45, 0.20),
        CurriculumStage("finetune", 20000, 0.45, 0.45, 0.20),
    ]

def weights_at_update(stages: list[CurriculumStage], update: int) -> CurriculumStage:
    """Return the active stage for the given PPO update count."""
    active = stages[0]
    for stage in stages:
        if update >= stage.start_update:
            active = stage
    return active
```

**Integration point**: In the PPO runner's update loop, after each PPO update, call `weights_at_update(update_count)` and update the carry's dynamic weights. This follows the same pattern as the existing `reward_curriculum` for qvel_w_sum.

### 4. Carry extensions — dynamic contact weights

Add to the carry dataclass (same pattern as existing `qvel_w_sum`, `root_vel_w_sum`):

```python
foot_contact_height_w_sum: float = 0.0
foot_contact_velocity_w_sum: float = 0.0
body_graph_w_sum: float = 0.0
```

These are set to 0.0 at init and updated by the curriculum scheduler each update.

### 5. Foot site registration — env knows where feet are

During MimicReward init, if contact tracking is enabled:
1. Look up foot site IDs from the MyoFullBody model (e.g., `left_toes_mimic`, `right_toes_mimic`, `left_ankle_mimic`, `right_ankle_mimic`)
2. Store as `self._foot_site_ids`
3. These map to the `foot_labels` in the reference bundle

### 6. Config file

**File**: `fullbody/config_specific_task/conf_fullbody_contact_tracking_gmr.yaml` (new)

```yaml
# @package _global_
defaults:
  - /conf_fullbody_gmr
  - _self_

wandb:
  tags: ["fullbody", "gmr", "contact_tracking"]

experiment:
  env_params:
    reward_params:
      # Existing weights (slightly reduced to make room for contact)
      rpos_w_sum: 0.45
      rquat_w_sum: 0.01
      rvel_w_sum: 0.05
      root_pos_w_sum: 0.10
      # Contact tracking (initial weights, overridden by curriculum)
      foot_contact_height_w_sum: 0.0
      foot_contact_height_w_exp: 80.0
      foot_contact_velocity_w_sum: 0.0
      foot_contact_velocity_w_exp: 8.0
      body_graph_w_sum: 0.0
      body_graph_w_exp: 20.0

  # Contact tracking config
  contact_tracking:
    enabled: true
    reference_root: null  # Path to reference bundles (or use tracking_cache_dir)
    tracking_cache_dir: null  # Pre-built cache directory
    foot_sites: ["left_toes_mimic", "right_toes_mimic", "left_ankle_mimic", "right_ankle_mimic"]
    curriculum:
      stages:
        - {name: body_only, start_update: 0, foot_h: 0.0, foot_v: 0.0, graph: 0.0}
        - {name: joint_tracking, start_update: 2000, foot_h: 0.0, foot_v: 0.0, graph: 0.0}
        - {name: contact_intro, start_update: 5000, foot_h: 0.20, foot_v: 0.20, graph: 0.10}
        - {name: contact_full, start_update: 10000, foot_h: 0.45, foot_v: 0.45, graph: 0.20}
        - {name: finetune, start_update: 20000, foot_h: 0.45, foot_v: 0.45, graph: 0.20}

  task_factory:
    params:
      amass_dataset_conf:
        dataset_group: null
        rel_dataset_path: []  # Filled per experiment
        retargeting_method: gmr
        gmr_config:
          src_human: smplh
          target_fps: 60
          solver: daqp
          damping: 0.5
          offset_to_ground: false
          use_fitted_shape: true
```

### 7. Engine integration — loading contact data

In `run_experiment()` (engine.py), after environment instantiation:

```python
if config.experiment.get("contact_tracking", {}).get("enabled", False):
    contact_data = load_contact_tracking_data(config.experiment.contact_tracking)
    env.reward_function._contact_tracking_data = contact_data
```

For the curriculum, in the PPO update callback:
```python
def on_ppo_update(update_count, carry):
    if curriculum_stages:
        stage = weights_at_update(curriculum_stages, update_count)
        carry = carry.replace(
            foot_contact_height_w_sum=stage.foot_contact_height_w,
            foot_contact_velocity_w_sum=stage.foot_contact_velocity_w,
            body_graph_w_sum=stage.body_graph_w,
        )
    return carry
```

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `musclemimic/badminton/asi/contact_tracking_data.py` | NEW | ContactTrackingData dataclass + loader |
| `musclemimic/badminton/asi/contact_curriculum.py` | NEW | CurriculumScheduler with stage-based weight scheduling |
| `musclemimic/core/reward/trajectory_based.py` | MODIFY | Add contact reward terms to MimicReward |
| `musclemimic/runner/engine.py` | MODIFY | Load contact data, wire curriculum callback |
| `fullbody/config_specific_task/conf_fullbody_contact_tracking_gmr.yaml` | NEW | Training config |

## Files NOT changed (existing, already working)

- `musclemimic/badminton/data/reference_bundle.py` — loads bundles correctly
- `musclemimic/badminton/asi/tracking_cache.py` — builds cache correctly
- `musclemimic/badminton/asi/rewards.py` — standalone reward functions (reuse logic, not import directly)
- `musclemimic/badminton/scripts/build_contact_tracking_manifest.py` — manifest builder

## Backward Compatibility

All new reward terms default to `w_sum=0.0`. Existing training configs are unaffected — contact rewards are only active when explicitly configured with non-zero weights. The curriculum carry fields also default to 0.0.

## Testing Strategy

1. Unit test: ContactTrackingData loading from cache npz
2. Unit test: CurriculumScheduler stage transitions
3. Unit test: MimicReward with contact terms (mock contact data, verify reward computation)
4. Integration test: Full training config loads and runs 1 PPO update without crash
5. Smoke test: 100-update training run, verify contact rewards appear in wandb logs after stage transition

## Usage

```bash
# 1. Ensure reference bundles are exported and materialized
python musclemimic/badminton/scripts/prepare_ppo_training_source.py \
  --source-mode reference_bundle \
  --reference-root /path/to/optimized_wham/output/forehand_clear/InsufficientArmExtension \
  --namespace forehand_clear/insufficient_arm_extension_refbundle \
  --fps 60

# 2. Build tracking cache (optional, for faster startup)
python musclemimic/badminton/scripts/build_tracking_reference_cache.py \
  --manifest /path/to/reference_bundle/manifest.json \
  --out-dir /path/to/cache \
  --control-dt 0.01

# 3. Train with contact tracking
python fullbody/experiment.py \
  --config-name conf_fullbody_contact_tracking_gmr \
  experiment.task_factory.params.amass_dataset_conf.rel_dataset_path='[
    "forehand_clear/insufficient_arm_extension_refbundle/001_6_23_1_-6_e7963b04",
    "forehand_clear/insufficient_arm_extension_refbundle/002_6_23_1_-7_0d26d382",
    "forehand_clear/insufficient_arm_extension_refbundle/003_6_23_1_-12_50cb8967"
  ]' \
  experiment.contact_tracking.reference_root=/path/to/optimized_wham/output/forehand_clear/InsufficientArmExtension
```
