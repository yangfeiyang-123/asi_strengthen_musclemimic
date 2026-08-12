"""Tests for teacher rollout collection helpers."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import musclemimic.distill as distill
from musclemimic.distill.collect_teacher import (
    _build_emg_reference_capture_spec,
    _resolve_actuator_names,
    _resolve_emg_action_indices,
    build_teacher_rollout_config,
    collect_teacher_dataset,
)
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.dagger import build_dagger_shard_data
from musclemimic.distill.dataset import load_metadata, write_distill_shard
from musclemimic.distill.motion_identity import MotionIdentityMap
from musclemimic.distill.obs_filter import StudentObsSpec, extract_reference_features


def test_build_teacher_rollout_config_disables_student_filter():
    exp = OmegaConf.create(
        {
            "num_envs": 1,
            "normalize_env": False,
            "gamma": 0.99,
            "student_obs_filter": {"enabled": True},
        }
    )

    rollout_cfg = build_teacher_rollout_config(exp, num_envs=8)

    assert rollout_cfg.num_envs == 8
    assert rollout_cfg.student_obs_filter.enabled is False


def test_teacher_collector_accepts_synergy_mode_and_reaches_wrapper_construction(tmp_path):
    agent_conf = SimpleNamespace(
        config=OmegaConf.create(
            {
                "experiment": {
                    "len_obs_history": 1,
                    "action_representation": {"mode": "fixed_synergy"},
                }
            }
        )
    )

    # A bare object cannot construct the MuJoCo-backed wrapper, but the
    # collector must no longer fail at the former blanket synergy rejection.
    with pytest.raises(ValueError, match="MuJoCo model"):
        collect_teacher_dataset(
            env=object(),
            agent_conf=agent_conf,
            agent_state=None,
            output_dir=tmp_path,
            num_envs=1,
            num_steps=1,
        )


def test_collector_actuator_schema_prefers_policy_interface_names_over_raw_model(tmp_path):
    class RawEnv:
        raw_actuator_names = tuple(
            [f"body_{index}" for index in range(354)] + [f"finger_{index}" for index in range(62)]
        )

    class PolicyInterface:
        env = RawEnv()
        policy_actuator_names = tuple(f"body_{index}" for index in range(354))

    interface = PolicyInterface()
    names = _resolve_actuator_names(interface, None)
    assert names == list(interface.policy_actuator_names)
    assert len(interface.env.raw_actuator_names) == 416
    assert interface.env.raw_actuator_names[-1] == "finger_61"
    assert _resolve_actuator_names(PolicyInterface(), ["explicit"]) == ["explicit"]

    action = np.zeros((2, 354), dtype=np.float32)
    write_distill_shard(
        tmp_path / "shard_000000.npz",
        {
            "student_obs": np.zeros((2, 3), dtype=np.float32),
            "teacher_action": action,
        },
        metadata={"actuator_names": names},
    )
    metadata = load_metadata(tmp_path)
    assert metadata["actuator_names"] == names
    assert metadata["actuator_names"][-1] == "body_353"
    assert metadata["action_dim"] == 354


def test_distill_collect_cli_defaults_to_teacher_mean_actions():
    from fullbody.distill_collect import build_parser

    args = build_parser().parse_args(
        [
            "--teacher_ckpt",
            "/ckpt/teacher",
            "--output_dir",
            "/tmp/distill",
        ]
    )

    assert args.teacher_action_target == "mean"
    assert args.deterministic_teacher is True
    assert args.save_physical_muscle_state is False
    assert args.save_event_features is False
    assert args.physical_racket_site_name is None
    assert args.teacher_promotion_stage == "stage2"
    assert args.teacher_promotion_role is None


def test_distill_collect_cli_requires_an_explicit_stage1_body_role_contract():
    from fullbody.distill_collect import build_parser

    args = build_parser().parse_args(
        [
            "--teacher_ckpt",
            "/ckpt/stage1",
            "--output_dir",
            "/tmp/distill",
            "--teacher-promotion-manifest",
            "/evidence/stage1.json",
            "--teacher-promotion-stage",
            "stage1",
            "--teacher-promotion-role",
            "body_only",
        ]
    )

    assert args.teacher_promotion_stage == "stage1"
    assert args.teacher_promotion_role == "body_only"


def _minimal_agent_conf():
    """Smallest conf that reaches the EMG flag guards without a MuJoCo model."""

    return SimpleNamespace(
        config=OmegaConf.create({"experiment": {"len_obs_history": 1}}),
    )


def _verified_trial_qc_provenance(
    *,
    action: str,
    channels: tuple[str, ...],
    mapping_sha256: str,
) -> dict[str, object]:
    """Minimal immutable human-QC binding accepted by formal test tubes."""

    return {
        "trial_qc_review": {
            "schema_version": "emg_trial_channel_qc_review_v1",
            "action": action,
            "review_status": "verified",
            "training_enabled": True,
            "source_path": "/controlled/evidence/emg_trial_qc_review.json",
            "review_sha256": "d" * 64,
            "mapping_sha256": mapping_sha256,
            "reviewer_id": "domain_expert",
            "reviewed_at": "2025-07-20T00:00:00Z",
            "review_evidence": ["controlled-review-record"],
            "trial_decisions": [
                {
                    "trial_id": "trial_001",
                    "decision": "include",
                    "reason": "reviewed synthetic fixture",
                    "mvc_normalized_emg_sha256": "1" * 64,
                    "preprocessing_qc_sha256": "2" * 64,
                }
            ],
            "channel_decisions": [
                {
                    "emg_channel": name,
                    "decision": "include_after_review",
                    "reason": "reviewed synthetic fixture",
                }
                for name in channels
            ],
            "risk_decisions": [
                {
                    "risk_id": risk_id,
                    "decision": "mitigated",
                    "reason": "reviewed synthetic fixture",
                    "evidence": ["controlled-review-record"],
                }
                for risk_id in ("s9_progressive_near_flatline", "super_mvc")
            ],
        }
    }


def test_distill_collect_cli_defaults_disable_emg_reference_capture():
    from fullbody.distill_collect import build_parser

    args = build_parser().parse_args(["--teacher_ckpt", "/ckpt/teacher", "--output_dir", "/tmp/distill"])

    assert args.save_emg_reference is False
    assert args.emg_reference_cache is None
    assert args.save_sim_anchor_activation is False


def test_distill_collect_cli_accepts_kebab_emg_flags():
    from fullbody.distill_collect import build_parser

    args = build_parser().parse_args(
        [
            "--teacher_ckpt",
            "/ckpt/teacher",
            "--output_dir",
            "/tmp/distill",
            "--save-emg-reference",
            "--emg-reference-cache",
            "/tmp/emg",
            "--stage1-peasd-promotion-manifest",
            "/tmp/stage1_peasd_promotion.json",
            "--save-sim-anchor-activation",
        ]
    )

    assert args.save_emg_reference is True
    assert args.emg_reference_cache == "/tmp/emg"
    assert args.stage1_peasd_promotion_manifest == "/tmp/stage1_peasd_promotion.json"
    assert args.save_sim_anchor_activation is True


def test_emg_reference_cache_without_save_flag_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="emg_reference_cache requires save_emg_reference"):
        collect_teacher_dataset(
            env=object(),
            agent_conf=_minimal_agent_conf(),
            agent_state=None,
            output_dir=tmp_path,
            num_envs=1,
            num_steps=1,
            emg_reference_cache=tmp_path,
        )


def test_save_emg_reference_without_cache_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="save_emg_reference=True requires emg_reference_cache"):
        collect_teacher_dataset(
            env=object(),
            agent_conf=_minimal_agent_conf(),
            agent_state=None,
            output_dir=tmp_path,
            num_envs=1,
            num_steps=1,
            save_emg_reference=True,
        )


def test_sim_anchor_activation_requires_physical_muscle_state(tmp_path):
    with pytest.raises(ValueError, match=r"save_sim_anchor_activation.*save_physical_muscle_state"):
        collect_teacher_dataset(
            env=object(),
            agent_conf=_minimal_agent_conf(),
            agent_state=None,
            output_dir=tmp_path,
            num_envs=1,
            num_steps=1,
            save_emg_reference=True,
            emg_reference_cache=tmp_path,
            save_sim_anchor_activation=True,
        )


def test_single_action_emg_tube_uses_row_zero_after_registry_action_binding():
    identity = MotionIdentityMap.from_paths(
        ["forehandClear_standard/muscle_trajectory/raw_smooth_v1/video10"]
    )
    spec = {
        "tube": SimpleNamespace(
            action_count=1,
            action_ids=("forehand_high_clear",),
            reference_id="P002_forehand_high_clear_16ch_v1",
        )
    }

    indices = _resolve_emg_action_indices(
        spec,
        motion_uid=identity.motion_uids,
        motion_identity_map=identity,
        batch_size=1,
    )

    np.testing.assert_array_equal(indices, np.zeros((1,), dtype=np.int64))


def test_single_action_emg_tube_cannot_be_borrowed_by_another_action():
    identity = MotionIdentityMap.from_paths(
        ["ChinaJump/muscle_trajectory/optimized/forehandJump-8"]
    )
    spec = {
        "tube": SimpleNamespace(
            action_count=1,
            action_ids=("forehand_high_clear",),
            reference_id="P002_forehand_high_clear_16ch_v1",
        )
    }

    with pytest.raises(ValueError, match="does not match collected motion action"):
        _resolve_emg_action_indices(
            spec,
            motion_uid=identity.motion_uids,
            motion_identity_map=identity,
            batch_size=1,
        )


def test_emg_capture_rejects_bundled_mapping_sha_mismatch(tmp_path, monkeypatch):
    mapping_path = tmp_path / "emg_observation_mapping.json"
    mapping_path.write_text(json.dumps({"mapping_id": "mapping-v1"}), encoding="utf-8")
    actual_sha = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    tube = SimpleNamespace(
        mapping_binding={
            "mapping_id": "mapping-v1",
            "mapping_sha256": "0" * 64,
            "mapping_review_status": "verified",
        },
        channel_names=("S2",),
        review_status="verified",
        training_enabled=True,
        reference_id="reference-v1",
        action_ids=("forehand_high_clear",),
        provenance=_verified_trial_qc_provenance(
            action="forehand_high_clear",
            channels=("S2",),
            mapping_sha256="0" * 64,
        ),
        anchor_valid=np.ones((1, 1, 1), dtype=bool),
        synergy_valid=np.ones((1, 1, 1), dtype=bool),
    )
    monkeypatch.setattr(
        "musclemimic.distill.collect_teacher.load_emg_phase_reference_tube",
        lambda _path: tube,
    )

    with pytest.raises(ValueError, match=r"mapping SHA-256 mismatch") as exc_info:
        _build_emg_reference_capture_spec(
            tmp_path,
            ["DELT1_R"],
            include_sim_anchor_activation=False,
        )

    assert actual_sha in str(exc_info.value)


def test_emg_capture_rejects_provisional_reference_before_writing_training_shards(
    tmp_path,
    monkeypatch,
):
    tube = SimpleNamespace(
        mapping_binding={"mapping_review_status": "provisional"},
        review_status="provisional",
        training_enabled=False,
        reference_id="diagnostics-only",
        anchor_valid=np.ones((1, 1, 1), dtype=bool),
        synergy_valid=np.ones((1, 1, 1), dtype=bool),
    )
    monkeypatch.setattr(
        "musclemimic.distill.collect_teacher.load_emg_phase_reference_tube",
        lambda _path: tube,
    )

    with pytest.raises(ValueError, match="mapping review must complete"):
        _build_emg_reference_capture_spec(
            tmp_path,
            ["DELT1_R"],
            include_sim_anchor_activation=False,
        )


def test_emg_capture_accepts_exact_bundled_mapping_sha(tmp_path, monkeypatch):
    mapping_path = tmp_path / "emg_observation_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "mapping_id": "mapping-v1",
                "channels": [
                    {
                        "mapping_status": "mapped",
                        "emg_channel": "S2",
                        "simulation_actuators": ["DELT1_R"],
                        "weights": [1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    actual_sha = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    tube = SimpleNamespace(
        mapping_binding={
            "mapping_id": "mapping-v1",
            "mapping_sha256": actual_sha,
            "mapping_review_status": "verified",
        },
        channel_names=("S2",),
        review_status="verified",
        training_enabled=True,
        anchor_mean=np.zeros((1, 2, 1), dtype=np.float32),
        anchor_scale=np.ones((1, 2, 1), dtype=np.float32),
        anchor_valid=np.ones((1, 2, 1), dtype=bool),
        synergy_mean=np.zeros((1, 2, 1), dtype=np.float32),
        synergy_scale=np.ones((1, 2, 1), dtype=np.float32),
        synergy_valid=np.ones((1, 2, 1), dtype=bool),
        reference_id="reference-v1",
        reference_fingerprint="f" * 64,
        action_ids=("forehand_high_clear",),
        provenance=_verified_trial_qc_provenance(
            action="forehand_high_clear",
            channels=("S2",),
            mapping_sha256=actual_sha,
        ),
        phase_bin_count=2,
        channel_count=1,
        synergy_count=1,
    )
    monkeypatch.setattr(
        "musclemimic.distill.collect_teacher.load_emg_phase_reference_tube",
        lambda _path: tube,
    )

    spec = _build_emg_reference_capture_spec(
        tmp_path,
        ["DELT1_R"],
        include_sim_anchor_activation=False,
    )

    np.testing.assert_array_equal(spec["projection"], np.ones((1, 1)))
    assert spec["metadata"]["mapping_sha256"] == actual_sha
    assert spec["metadata"]["trial_qc_review_sha256"] == "d" * 64


def test_distill_package_lazy_exports_public_functions():
    assert distill.bc_loss is not None
    assert distill.collect_teacher_dataset is not None
    assert distill.collect_dagger_dataset is not None
    assert distill.train_bc is not None


def test_apply_collection_overrides_supports_motion_group_and_fixed_start():
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {},
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["old_motion"],
                            "dataset_group": "OLD",
                        }
                    }
                },
            }
        }
    )

    apply_collection_overrides(
        config,
        motion_group="KIT_TEST",
        traj_index=3,
        traj_start_step=12,
    )

    dataset_conf = config.experiment.task_factory.params.amass_dataset_conf
    assert dataset_conf.dataset_group == "KIT_TEST"
    assert dataset_conf.rel_dataset_path == ["old_motion"]
    assert config.experiment.env_params.th_params.fixed_start_conf == [3, 12]
    assert config.experiment.env_params.th_params.random_start is False


def test_apply_collection_overrides_motion_path_takes_precedence_over_group():
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {},
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["old_motion"],
                            "dataset_group": "OLD",
                        }
                    }
                },
            }
        }
    )

    apply_collection_overrides(config, motion_path=["new_motion"], motion_group="IGNORED")

    dataset_conf = config.experiment.task_factory.params.amass_dataset_conf
    assert dataset_conf.rel_dataset_path == ["new_motion"]
    assert dataset_conf.dataset_group is None


def test_extract_reference_features_uses_dropped_goal_lookahead_without_phase():
    spec = StudentObsSpec(
        raw_obs_dim=8,
        goal_indices=np.array([4, 5, 6, 7]),
        state_indices=np.array([0, 1, 2, 3]),
        student_indices=np.array([0, 1, 2, 3, 7]),
        phase_index=7,
    )
    obs = np.arange(16, dtype=np.float32).reshape(2, 8)

    reference = extract_reference_features(obs, spec)

    np.testing.assert_array_equal(reference, obs[:, [4, 5, 6]])


def test_build_dagger_shard_data_can_include_reference_features():
    spec = StudentObsSpec(
        raw_obs_dim=6,
        goal_indices=np.array([3, 4, 5]),
        state_indices=np.array([0, 1, 2]),
        student_indices=np.array([0, 1, 2, 5]),
        phase_index=5,
    )
    full_obs = np.arange(12, dtype=np.float32).reshape(2, 6)

    data = build_dagger_shard_data(
        full_obs=full_obs,
        teacher_mu=np.zeros((2, 2), dtype=np.float32),
        student_action=np.zeros((2, 2), dtype=np.float32),
        reward=np.zeros((2,), dtype=np.float32),
        done=np.zeros((2,), dtype=bool),
        absorbing=np.zeros((2,), dtype=bool),
        info={},
        spec=spec,
        save_reference_features=True,
    )

    np.testing.assert_array_equal(data["reference_features"], full_obs[:, [3, 4]])
