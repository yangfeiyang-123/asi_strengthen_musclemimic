from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.distill.body_obs_schema import build_body_obs_schema
from musclemimic.distill.dataset import (
    SequenceDistillDataset,
    motion_split_datasets,
    write_split_shard,
)
from musclemimic.distill.motion_identity import (
    MotionIdentityMap,
    RolloutIdentityTracker,
    normalize_motion_path,
    select_transition_traj_no,
    stable_motion_uid,
    validate_environment_motion_identity,
)
from musclemimic.distill.obs_filter import StudentObsSpec
from musclemimic.latent_muscle.normalization import ObservationNormalizer


def _strict_data(*, motion_uid: int, rollout_uid: int, obs_offset: float = 0.0, n: int = 4):
    action = np.zeros((n, 2), dtype=np.float32)
    return {
        "student_obs": np.arange(obs_offset, obs_offset + n * 3, dtype=np.float32).reshape(n, 3),
        "reference_features": np.ones((n, 2), dtype=np.float32),
        "teacher_action": action,
        "teacher_mu": action.copy(),
        "traj_no": np.zeros(n, dtype=np.int32),
        "subtraj_step_no": np.arange(n, dtype=np.int32),
        "motion_uid": np.full(n, motion_uid, dtype=np.int64),
        "rollout_uid": np.full(n, rollout_uid, dtype=np.int64),
        "rollout_step": np.arange(n, dtype=np.int32),
        "env_index": np.zeros(n, dtype=np.int32),
    }


def test_motion_uid_is_stable_across_local_traj_numbering():
    train = MotionIdentityMap.from_paths(["skill/forehand/a", "skill/forehand/b"])
    val = MotionIdentityMap.from_paths(["skill/forehand/b"])

    assert int(train.map_traj_no(np.array([1]))[0]) == int(val.map_traj_no(np.array([0]))[0])
    assert stable_motion_uid("skill/forehand/b") == stable_motion_uid("./skill/forehand/b")
    assert normalize_motion_path("skill\\forehand\\b") == "skill/forehand/b"


def test_rollout_identity_tracker_separates_vector_lanes_and_episode_resets():
    tracker = RolloutIdentityTracker(num_envs=2, collection_uid=123)
    uid0, step0, env0 = tracker.current()
    tracker.advance(np.array([True, False]))
    uid1, step1, env1 = tracker.current()

    assert uid0[0] != uid0[1]
    assert uid1[0] != uid0[0]
    assert uid1[1] == uid0[1]
    np.testing.assert_array_equal(step0, [0, 0])
    np.testing.assert_array_equal(step1, [0, 1])
    np.testing.assert_array_equal(env0, env1)


def test_terminal_transition_and_environment_motion_count_fail_fast():
    selected = select_transition_traj_no(
        np.array([4, 5], dtype=np.int32),
        np.array([True, False]),
        final_traj_no=np.array([1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(selected, [1, 5])

    identity = MotionIdentityMap.from_paths(["a", "b"])

    class Env:
        class TH:
            n_trajectories = 2

        th = TH()

    validate_environment_motion_identity(Env(), identity)
    Env.th.n_trajectories = 3
    with pytest.raises(ValueError, match="n_trajectories=3"):
        validate_environment_motion_identity(Env(), identity)


def test_explicit_train_val_split_uses_motion_uid_and_train_only_normalizer(tmp_path):
    train_uid = stable_motion_uid("motions/train")
    val_uid = stable_motion_uid("motions/val")
    write_split_shard(tmp_path, _strict_data(motion_uid=train_uid, rollout_uid=10), split="train")
    write_split_shard(
        tmp_path,
        _strict_data(motion_uid=val_uid, rollout_uid=20, obs_offset=10_000.0),
        split="val",
    )

    train, val, manifest = motion_split_datasets(
        tmp_path,
        dataset_cls=SequenceDistillDataset,
        require_stable_ids=True,
    )
    normalizer = ObservationNormalizer.fit(train.arrays["student_obs"])

    assert val is not None
    assert manifest["train_motion_ids"] == [train_uid]
    assert manifest["val_motion_ids"] == [val_uid]
    assert normalizer.count == train.num_samples
    assert float(np.max(normalizer.mean)) < 100.0
    assert float(np.min(normalizer.normalize_numpy(val.arrays["student_obs"]))) == normalizer.clip


def test_explicit_train_val_without_motion_uid_is_rejected(tmp_path):
    legacy = _strict_data(motion_uid=1, rollout_uid=10)
    legacy.pop("motion_uid")
    write_split_shard(tmp_path, legacy, split="train")
    write_split_shard(tmp_path, legacy, split="val")

    with pytest.raises(ValueError, match="stable motion_uid"):
        motion_split_datasets(tmp_path, dataset_cls=SequenceDistillDataset)


def test_train_val_cannot_overwrite_ordered_action_schema_metadata(tmp_path):
    train = _strict_data(motion_uid=stable_motion_uid("train"), rollout_uid=10)
    val = _strict_data(motion_uid=stable_motion_uid("val"), rollout_uid=20)
    write_split_shard(
        tmp_path,
        train,
        split="train",
        metadata={"actuator_names": ["hip", "shoulder"]},
    )
    with pytest.raises(ValueError, match="ABI metadata mismatch.*actuator_names"):
        write_split_shard(
            tmp_path,
            val,
            split="val",
            metadata={"actuator_names": ["shoulder", "hip"]},
        )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["actuator_names"] == ["hip", "shoulder"]
    assert "train" in metadata["split_metadata"]


def test_body_obs_semantic_hash_excludes_checkpoint_provenance():
    spec = StudentObsSpec(
        raw_obs_dim=3,
        goal_indices=np.array([2]),
        state_indices=np.array([0, 1]),
        student_indices=np.array([0, 1, 2]),
        phase_index=2,
    )

    class Env:
        root_free_joint_xml_name = "root"

    channels = [
        {
            "name": f"state[{index}]",
            "entry": "q_all_pos" if index < 2 else "GoalTrajMimic",
            "entry_type": "test",
            "entry_offset": index,
            "student_index": index,
            "joint_name": "hip" if index < 2 else None,
        }
        for index in range(3)
    ]
    first = build_body_obs_schema(
        env=Env(),
        spec=spec,
        actuator_names=["hip_muscle"],
        channels=channels,
        provenance={"teacher_ckpt": "/old/path"},
    )
    second = build_body_obs_schema(
        env=Env(),
        spec=spec,
        actuator_names=["hip_muscle"],
        channels=channels,
        provenance={"teacher_ckpt": "/new/path"},
    )

    assert first["semantic_hash"] == second["semantic_hash"]
    assert first["provenance"] != second["provenance"]


def test_sequence_windows_never_mix_parallel_rollouts_or_cross_gaps(tmp_path):
    # Time-major rows from two vector lanes. rollout 10 has a missing step 1,
    # while rollout 20 is contiguous; only rollout 20 can form horizon=3.
    n = 6
    data = {
        "student_obs": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        "reference_features": np.ones((n, 1), dtype=np.float32),
        "teacher_action": np.zeros((n, 2), dtype=np.float32),
        "teacher_mu": np.zeros((n, 2), dtype=np.float32),
        "traj_no": np.zeros(n, dtype=np.int32),
        "subtraj_step_no": np.array([0, 0, 2, 1, 3, 2], dtype=np.int32),
        "motion_uid": np.full(n, stable_motion_uid("motions/a"), dtype=np.int64),
        "rollout_uid": np.array([10, 20, 10, 20, 10, 20], dtype=np.int64),
        "rollout_step": np.array([0, 0, 2, 1, 3, 2], dtype=np.int32),
        "env_index": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
    }
    write_split_shard(tmp_path, data, split="train")
    dataset = SequenceDistillDataset(tmp_path, require_stable_ids=True)
    batch = next(dataset.iter_sequence_batches(batch_size=1, horizon=3, shuffle=False))

    np.testing.assert_array_equal(batch["rollout_uid"], [[20, 20, 20]])
    np.testing.assert_array_equal(batch["rollout_step"], [[0, 1, 2]])
    np.testing.assert_array_equal(batch["env_index"], [[1, 1, 1]])


def test_production_latent_yaml_enforces_plan_defaults():
    from omegaconf import OmegaConf

    path = "fullbody/config_specific_task/distill/latent_forehandclear_lab.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["latent_distill"]

    assert cfg["dataset_dir"].endswith("latent_stage2_racket_raw_smooth_v1")
    assert cfg["output_dir"].endswith("latent_stage2_racket_raw_smooth_v1")
    assert cfg["latent_dim"] == 16
    assert cfg["horizon"] == 8
    assert cfg["smooth_weight"] == pytest.approx(0.01)
    assert cfg["val_fraction"] == 0.2
    assert cfg["strict_motion_identity"] is True
    assert cfg["require_dataset_provenance"] is True
    assert cfg["action_mask"]["preset"] == "myofullbody_finger_partition"
    assert json.dumps(cfg["promotion_gates"])


def test_latent_cli_accepts_pipeline_teacher_and_direct_bc_metrics(tmp_path):
    from fullbody.latent_train import _direct_bc_action_mse, build_parser

    metrics_path = tmp_path / "direct_bc.json"
    # Match train_bc.py's real distill_metadata.json layout and prove the
    # held-out value, rather than the lower-value training tensor, is used.
    metrics_path.write_text(
        json.dumps(
            {
                "train": {
                    "mse_to_teacher_action": 0.001,
                    "mse_to_teacher_mu": 0.101,
                },
                "val": {
                    "mse_to_teacher_action": 0.012,
                    "mse_to_teacher_mu": 0.112,
                },
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            "fullbody/config_specific_task/distill/latent_forehandclear_lab.yaml",
            "--direct_bc_metrics",
            str(metrics_path),
            "--teacher_ckpt",
            "/checkpoints/stage2",
            "--teacher_promotion_manifest",
            "/evidence/stage2.json",
        ]
    )

    assert args.teacher_ckpt == "/checkpoints/stage2"
    assert args.teacher_promotion_manifest == "/evidence/stage2.json"
    assert _direct_bc_action_mse(metrics_path) == pytest.approx(0.012)


def test_production_action_mask_preserves_real_416_model_order():
    pytest.importorskip("mujoco")
    from musclemimic_models import load

    from musclemimic.latent_muscle.train_latent import _build_action_mask
    from musclemimic.runner.export_metadata import model_actuator_names
    from musclemimic.utils.finger_isolation import FingerActuatorPartition

    model, _data = load("myofullbody")
    runtime_names = model_actuator_names(model)
    partition = FingerActuatorPartition.from_actuator_names(runtime_names)
    mask = _build_action_mask(
        {"preset": "myofullbody_finger_partition"},
        354,
        list(partition.body_actuator_names),
    )

    assert mask.all_actuator_names == runtime_names
    assert mask.body_actuator_names == list(partition.body_actuator_names)
    assert mask.correction_actuator_names == list(partition.right_grip_actuator_names)
    assert mask.neutral_actuator_names == list(partition.left_neutral_actuator_names)
    assert mask.body_size + mask.correction_size + mask.neutral_size == 416
