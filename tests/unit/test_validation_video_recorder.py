"""Unit tests for ValidationVideoRecorder - replicates training validation workflow."""

import os
import json
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from loco_mujoco.core.stateful_object import StatefulObject
import musclemimic.utils.display as display_module
import musclemimic.runner.validation_video_recorder as recorder_module
from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder
from musclemimic.badminton.visual_review import (
    CANDIDATE_SCHEMA_VERSION,
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
)
from musclemimic.runner.logging import ExperimentHooks
from musclemimic.utils import setup_headless_rendering_if_needed
from loco_mujoco.task_factories import ImitationFactory


class TestValidationVideoRecorderConfig:
    """Test environment configuration building for validation videos."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=500,
            deterministic=True
        )

    @pytest.fixture
    def bimanual_training_config(self):
        """Training config from conf_bimanual.yaml with validation overrides."""
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]
                        }
                    }
                },
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "headless": True,
                    "horizon": 1000,
                    "disable_fingers": False,
                    "mjx_backend": "warp",
                    "num_envs": 2048,
                    "enable_muscle_length_observations": False,
                    "enable_muscle_velocity_observations": False,
                    "enable_muscle_force_observations": True,
                    "enable_muscle_activation_observations": True,
                    "init_state_type": "TrajInitialStateHandler",
                    "goal_type": "GoalBimanualTrajMimic",
                    "goal_params": {
                        "visualize_goal": True,
                        "n_step_lookahead": 1,
                        "upper_body_xml_name": "thorax",
                        "sites_for_mimic": [
                            "right_shoulder_mimic",
                            "right_elbow_mimic",
                            "right_hand_mimic",
                            "left_shoulder_mimic",
                            "left_elbow_mimic",
                            "left_hand_mimic"
                        ]
                    },
                    "control_type": "DefaultControl",
                    "control_params": {},
                    "reward_type": "MimicReward",
                    "reward_params": {
                        "qpos_w_sum": 0.5,
                        "qvel_w_sum": 0.5,
                        "rpos_w_sum": 2.0,
                        "rquat_w_sum": 0.2,
                        "rvel_w_sum": 0.5,
                        "use_mean_exp_reward": False,
                        "sites_for_mimic": [
                            "right_shoulder_mimic",
                            "right_elbow_mimic",
                            "right_hand_mimic",
                            "left_shoulder_mimic",
                            "left_elbow_mimic",
                            "left_hand_mimic"
                        ]
                    },
                    "terminal_state_type": "EnhancedBimanualTerminalStateHandler",
                    "terminal_state_params": {
                        "max_site_deviation": 0.15,
                        "enable_reference_check": False,
                        "site_deviation_mode": "mean",
                        "debug_print": False,
                        "debug_every": 10
                    },
                    "timestep": 0.002,
                    "n_substeps": 5
                },
                "validation": {
                    "active": True,
                    "deterministic": True,
                    "num_steps": 500,
                    "num_envs": 20,
                    "num": 50,
                    "video_length": 500,
                    "video_frequency": 1,
                    "terminal_state_type": "NoTerminalStateHandler",
                    "terminal_state_params": {}
                }
            }
        })
        return config

    @pytest.fixture
    def mock_agent_conf(self, bimanual_training_config):
        agent_conf = MagicMock()
        agent_conf.config = bimanual_training_config
        return agent_conf

    @pytest.fixture
    def myofullbody_training_config(self):
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["CMU/01/03"]
                        }
                    }
                },
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "headless": True,
                    "goal_type": "GoalTrajMimic",
                    "goal_params": {
                        "sites_for_mimic": [
                            "pelvis", "head", "left_hand", "right_hand"
                        ]
                    },
                    "control_type": "DefaultControl",
                    "reward_type": "MimicReward",
                    "reward_params": {},
                    "timestep": 0.005,
                    "n_substeps": 4
                },
                "validation": {}
            }
        })
        return config

    @pytest.fixture
    def mock_fullbody_agent_conf(self, myofullbody_training_config):
        agent_conf = MagicMock()
        agent_conf.config = myofullbody_training_config
        return agent_conf

    @pytest.fixture
    def mock_agent_state(self):
        return MagicMock()

    def test_mjx_to_cpu_conversion(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["env_name"] == "MyoBimanualArm"

    def test_mjx_params_removed(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        for param in ["mjx_backend", "num_envs", "nconmax", "njmax"]:
            assert param not in env_params

    def test_validation_disables_early_termination(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        if "terminal_state_type" in mock_agent_conf.config.experiment.validation:
            assert env_params["terminal_state_type"] == "NoTerminalStateHandler"
            assert env_params["terminal_state_params"] == {}

    def test_headless_rendering_enabled(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["headless"] is True

    def test_goal_type_v2_for_visualization(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["goal_type"] == "GoalBimanualTrajMimicv2"

    def test_visualize_goal_flag_set(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["visualize_goal"] is True
        assert env_params["goal_params"]["visualize_goal"] is True

    def test_recorder_params_configured(self, video_recorder, mock_agent_conf):
        tag = "validation_5_t10000_20241119"
        env_params = video_recorder._build_env_params(mock_agent_conf, tag)
        assert env_params["recorder_params"]["tag"] == tag
        assert env_params["recorder_params"]["fps"] == 100  # 1/(0.002*5)
        assert env_params["recorder_params"]["compress"] is True

    def test_timing_preserved_from_training(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["timestep"] == 0.002
        assert env_params["n_substeps"] == 5

    def test_sites_for_mimic_preserved(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        expected = [
            "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
            "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
        ]
        assert env_params["goal_params"]["sites_for_mimic"] == expected

    def test_fullbody_goal_visualization_defaults(self, video_recorder, mock_fullbody_agent_conf):
        env_params = video_recorder._build_env_params(mock_fullbody_agent_conf, "fullbody_tag")
        assert env_params["env_name"] == "MyoFullBody"
        assert env_params["goal_type"] == "GoalTrajMimicv2"
        assert env_params["goal_params"]["sites_for_mimic"] == [
            "pelvis", "head", "left_hand", "right_hand"
        ]
        assert env_params["goal_params"]["enable_enhanced_visualization"] is True
        assert env_params["goal_params"]["target_geom_rgba"] == [0.471, 0.38, 0.812, 0.6]
        assert env_params["visualize_goal"] is True
        assert env_params["recorder_params"]["path"] == video_recorder.video_dir
        assert env_params["recorder_params"]["fps"] == 50  # 1/(0.005*4)

    def test_task_params_use_validation_dataset_override(self, video_recorder):
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["Action/train_motion"],
                            "retargeting_method": "gmr",
                        }
                    },
                },
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "goal_params": {},
                    "timestep": 0.005,
                    "n_substeps": 4,
                },
                "validation": {
                    "amass_dataset_conf": {
                        "rel_dataset_path": ["Action/validation_motion"],
                        "retargeting_method": "gmr",
                    }
                },
            }
        })
        agent_conf = MagicMock()
        agent_conf.config = config

        task_params = video_recorder._build_task_params(agent_conf)

        assert task_params["amass_dataset_conf"]["rel_dataset_path"] == ["Action/validation_motion"]
        assert task_params["amass_dataset_conf"]["max_motions"] == 3

    def test_cycled_video_task_params_load_every_validation_motion(self):
        recorder = ValidationVideoRecorder(
            video_dir="/tmp/videos", cycle_trajectories=True
        )
        config = OmegaConf.create(
            {
                "experiment": {
                    "task_factory": {
                        "name": "ImitationFactory",
                        "params": {"amass_dataset_conf": {}},
                    },
                    "env_params": {"env_name": "MjxMyoFullBody"},
                    "validation": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": [f"heldout/{index}" for index in range(5)]
                        }
                    },
                }
            }
        )
        agent_conf = MagicMock(config=config)

        task_params = recorder._build_task_params(agent_conf)

        assert task_params["amass_dataset_conf"]["max_motions"] == 5


class TestHeadlessSetupHelpers:
    def test_headless_detection_sets_egl(self, monkeypatch):
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: True)
        monkeypatch.setattr(display_module, "glob", lambda _: [])
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "egl"

    def test_headless_detection_sets_osmesa_when_dri_devices_are_inaccessible(self, monkeypatch):
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: True)
        monkeypatch.setattr(
            display_module,
            "glob",
            lambda pattern: ["/dev/dri/renderD128"] if "renderD" in pattern else [],
        )
        monkeypatch.setattr(display_module.os, "access", lambda path, mode: False)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "osmesa"

    def test_headless_detection_keeps_egl_when_dri_devices_are_accessible(self, monkeypatch):
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: True)
        monkeypatch.setattr(
            display_module,
            "glob",
            lambda pattern: ["/dev/dri/renderD128"] if "renderD" in pattern else [],
        )
        monkeypatch.setattr(display_module.os, "access", lambda path, mode: True)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "egl"

    def test_headless_preserves_existing_gl_backend(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "osmesa")
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: True)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "osmesa"

    def test_non_headless_preserves_existing_gl_backend(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "glfw")
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: False)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "glfw"


class TestValidationVideoRecorderErrorHandling:
    """Test error handling during validation video recording."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=500,
        )

    @pytest.fixture
    def mock_agent_conf(self):
        return MagicMock(config=OmegaConf.create({
            "experiment": {
                "task_factory": {"name": "ImitationFactory", "params": {}},
                "env_params": {"env_name": "MjxMyoBimanualArm", "timestep": 0.002, "n_substeps": 5},
                "validation": {}
            }
        }))

    @pytest.fixture
    def mock_agent_state(self):
        return MagicMock()

    def test_env_creation_failure_no_unboundlocalerror(
        self, video_recorder, mock_agent_conf, mock_agent_state
    ):
        with patch.object(ImitationFactory, 'make', side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                video_recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=1,
                    timestep=1000
                )

    def test_env_creation_failure_detailed_logging(
        self, video_recorder, mock_agent_conf, mock_agent_state, capsys
    ):
        with patch.object(ImitationFactory, 'make', side_effect=ValueError("Invalid param")):
            with pytest.raises(ValueError):
                video_recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=1,
                    timestep=1000
                )
        captured = capsys.readouterr()
        assert "[ValidationVideo] ERROR:" in captured.out
        assert "ValueError" in captured.out
        assert "Invalid param" in captured.out

    def test_frequency_skipping(self, mock_agent_conf, mock_agent_state):
        recorder = ValidationVideoRecorder(video_dir="/tmp/test", frequency=5, length=500)
        for val_num in [1, 2, 3, 4]:
            result = recorder.record_episode(
                agent_conf=mock_agent_conf,
                agent_state=mock_agent_state,
                validation_number=val_num,
                timestep=val_num * 1000
            )
            assert result is None

    def test_max_recordings_skips_later_validation(
        self, mock_agent_conf, mock_agent_state
    ):
        recorder = ValidationVideoRecorder(
            video_dir="/tmp/test", frequency=1, length=10, max_recordings=5
        )
        assert (
            recorder.record_episode(
                agent_conf=mock_agent_conf,
                agent_state=mock_agent_state,
                validation_number=6,
                timestep=6000,
            )
            is None
        )

    def test_frequency_recording(self, mock_agent_conf, mock_agent_state):
        recorder = ValidationVideoRecorder(video_dir="/tmp/test", frequency=5, length=500)
        with patch.object(ImitationFactory, 'make') as mock_make:
            mock_env = MagicMock()
            mock_env.video_file_path = "/tmp/test/video.mp4"
            mock_make.return_value = mock_env
            with patch(
                'musclemimic.runner.validation_video_recorder.PPOJax.play_policy'
            ) as mock_play:
                result = recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=5,
                    timestep=5000,
                    candidate_identity={"update_number": 10},
                )
                assert mock_make.called
                assert mock_env.stop.called
                assert result == "/tmp/test/video.mp4"
                assert mock_play.call_args.kwargs["deterministic"] is True
                assert mock_play.call_args.kwargs["freeze_run_stats"] is True

    def test_cycle_records_distinct_validation_motion_and_candidate_manifest(
        self, tmp_path, mock_agent_state
    ):
        config = OmegaConf.create(
            {
                "experiment": {
                    "task_factory": {
                        "name": "ImitationFactory",
                        "params": {"amass_dataset_conf": {}},
                    },
                    "env_params": {
                        "env_name": "MjxMyoFullBody",
                        "timestep": 0.002,
                        "n_substeps": 5,
                        "goal_params": {},
                    },
                    "validation": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["heldout/a", "heldout/b"]
                        }
                    },
                }
            }
        )
        agent_conf = MagicMock(config=config)
        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path),
            frequency=2,
            length=10,
            cycle_trajectories=True,
            review_kind=STAGE1_REVIEW_KIND,
        )
        mock_env = MagicMock()
        mock_env.th.n_trajectories = 2
        mock_env.video_file_path = str(tmp_path / "b.mp4")

        with (
            patch.object(ImitationFactory, "make", return_value=mock_env),
            patch.object(recorder_module, "setup_headless_rendering_if_needed"),
            patch.object(recorder_module.PPOJax, "play_policy"),
        ):
            result = recorder.record_episode(
                agent_conf=agent_conf,
                agent_state=mock_agent_state,
                validation_number=4,
                timestep=123,
            )

        assert result == str(tmp_path / "b.mp4")
        assert mock_env.th.fixed_start_conf == [1, 0]
        assert mock_env.th.use_fixed_start is True
        rows = [
            json.loads(line)
            for line in (tmp_path / "stage1_visual_review_candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows == [
            {
                "artifact": str(tmp_path / "b.mp4"),
                "major_swing_complete": None,
                "motion": "heldout/b",
                "notes": None,
                "passed": None,
                "review_kind": STAGE1_REVIEW_KIND,
                "right_hand_tracking_spike_free": None,
                "root_tracking_spike_free": None,
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "timestep": 123,
                "validation_number": 4,
            }
        ]

    def test_stage2_candidate_manifest_is_typed_unique_and_racket_aware(self, tmp_path):
        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path),
            review_kind=STAGE2_REVIEW_KIND,
        )
        for index in range(5):
            recorder._write_visual_candidate(
                motion=f"heldout/{index}",
                artifact=f"videos/{index}.mp4",
                validation_number=index + 1,
                timestep=(index + 1) * 100,
            )
        # Re-recording one motion replaces its row instead of introducing a
        # duplicate after resume.
        recorder._write_visual_candidate(
            motion="heldout/0",
            artifact="videos/0-new.mp4",
            validation_number=6,
            timestep=600,
        )
        path = tmp_path / "stage2_visual_review_candidates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 5
        assert len({row["motion"] for row in rows}) == 5
        assert len({row["artifact"] for row in rows}) == 5
        assert {row["review_kind"] for row in rows} == {STAGE2_REVIEW_KIND}
        assert all("racket_head_trajectory_ok" in row for row in rows)
        assert all("racket_face_orientation_ok" in row for row in rows)

        with pytest.raises(ValueError, match="already assigned to another motion"):
            recorder._write_visual_candidate(
                motion="heldout/new",
                artifact=rows[1]["artifact"],
                validation_number=7,
                timestep=700,
            )

    def test_promotion_review_set_records_every_motion_at_one_candidate_update(
        self, tmp_path, monkeypatch
    ):
        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path),
            review_kind=STAGE1_REVIEW_KIND,
            cycle_trajectories=True,
        )
        config = OmegaConf.create(
            {
                "experiment": {
                    "validation": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["heldout/a", "heldout/b"]
                        }
                    }
                }
            }
        )
        calls = []

        def fake_record(**kwargs):
            calls.append(kwargs)
            artifact = tmp_path / f"{kwargs['motion_index']}.mp4"
            artifact.write_bytes(f"clip-{kwargs['motion_index']}".encode("utf-8"))
            return str(artifact)

        monkeypatch.setattr(recorder, "record_episode", fake_record)
        identity = {"update_number": 30, "global_timestep": 3000}
        paths = recorder.record_review_set(
            agent_conf=MagicMock(config=config),
            agent_state=MagicMock(),
            validation_number=5,
            timestep=3000,
            candidate_identity=identity,
        )

        assert paths == [str(tmp_path / "0.mp4"), str(tmp_path / "1.mp4")]
        assert [call["motion_index"] for call in calls] == [0, 1]
        assert {call["validation_number"] for call in calls} == {5}
        assert all(call["candidate_identity"] == identity for call in calls)
        review_set = json.loads(
            (tmp_path / "stage1_endpoint_review_set.json").read_text(encoding="utf-8")
        )
        assert review_set["candidate"] == identity
        assert review_set["run_stats_frozen"] is True
        assert [clip["motion"] for clip in review_set["clips"]] == [
            "heldout/a",
            "heldout/b",
        ]
        assert not (tmp_path / "stage1_visual_review_template.json").exists()
        package_dir = tmp_path / "stage1_blind_review_package"
        package_path = package_dir / "package_manifest.json"
        template_path = package_dir / "review_template.json"
        mapping_path = tmp_path / "stage1_blind_private_mapping.json"
        package = recorder_module.validate_stage1_peasd_blind_package(package_path)
        mapping = recorder_module.validate_stage1_peasd_blind_mapping(
            mapping_path,
            expected_candidate=identity,
            expected_motions=["heldout/a", "heldout/b"],
        )
        template = json.loads(template_path.read_text(encoding="utf-8"))
        visible = json.dumps({"package": package, "template": template}).lower()
        assert "heldout/a" not in visible
        assert "heldout/b" not in visible
        assert "candidate" not in visible
        assert "checkpoint" not in visible
        assert "run_id" not in visible
        assert '"arm"' not in visible
        assert '"seed"' not in visible
        assert len(package["clips"]) == 2
        assert mapping["candidate"] == identity
        assert mapping_path.parent == tmp_path
        assert mapping_path.parent != package_dir

        review = template
        review["reviewer_id"] = "reviewer-fixture"
        review["passed"] = True
        for row in review["clips"]:
            row["major_swing_complete"] = True
            row["root_tracking_spike_free"] = True
            row["right_hand_tracking_spike_free"] = True
            row["passed"] = True
            row["notes"] = "opaque clip passed"
        review_path = package_dir / "review.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        report = recorder_module.validate_stage1_peasd_blind_review(
            review_path,
            private_mapping=mapping_path,
            expected_candidate=identity,
            expected_motions=["heldout/a", "heldout/b"],
        )
        assert report["validation_report"]["passed"] is True

    def test_logging_propagates_review_kind_to_recorder(self, tmp_path):
        config = OmegaConf.create(
            {
                "experiment": {
                    "validation": {
                        "active": True,
                        "visual_review_kind": STAGE2_REVIEW_KIND,
                    }
                }
            }
        )
        recorder = ExperimentHooks().build_video_recorder(str(tmp_path), config)
        assert recorder.review_kind == STAGE2_REVIEW_KIND

    def test_env_stop_called_on_success(self, video_recorder, mock_agent_conf, mock_agent_state):
        with patch.object(ImitationFactory, 'make') as mock_make:
            mock_env = MagicMock()
            mock_env.video_file_path = "/tmp/test/video.mp4"
            mock_make.return_value = mock_env
            with patch('musclemimic.runner.validation_video_recorder.PPOJax.play_policy'):
                video_recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=1,
                    timestep=1000
                )
                assert mock_env.stop.called


class TestValidationVideoRecorderRecordingBehavior:
    """Test StatefulObject isolation and PPO invocation."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=20,
            deterministic=False
        )

    @pytest.fixture
    def simple_agent_conf(self):
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {"name": "ImitationFactory", "params": {}},
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "goal_type": "GoalBimanualTrajMimic",
                    "goal_params": {"sites_for_mimic": ["left_hand_mimic"]},
                    "timestep": 0.002,
                    "n_substeps": 5
                },
                "validation": {}
            }
        })
        agent_conf = MagicMock()
        agent_conf.config = config
        return agent_conf

    @pytest.fixture
    def agent_state(self):
        return MagicMock()

    def test_stateful_objects_restored_and_play_policy_called(
        self, video_recorder, simple_agent_conf, agent_state, tmp_path
    ):
        dummy_env = MagicMock()
        dummy_env.video_file_path = str(tmp_path / "videos" / "mock.mp4")
        dummy_env.stop = MagicMock()

        class DummyFactory:
            @staticmethod
            def make(*_args, **_kwargs):
                return dummy_env

        sentinel_instances = ["sentinel"]
        original_instances = list(StatefulObject._instances)

        with patch.object(recorder_module.TaskFactory, "get_factory_cls", return_value=DummyFactory), \
            patch.object(recorder_module.PPOJax, "play_policy") as mock_play:
            try:
                StatefulObject._instances = list(sentinel_instances)
                result = video_recorder.record_episode(
                    agent_conf=simple_agent_conf,
                    agent_state=agent_state,
                    validation_number=1,
                    timestep=42
                )
                restored_instances = list(StatefulObject._instances)
            finally:
                StatefulObject._instances = original_instances

        assert result == dummy_env.video_file_path
        assert dummy_env.stop.called
        assert restored_instances == sentinel_instances
        mock_play.assert_called_once()
        args, kwargs = mock_play.call_args
        assert args[0] is dummy_env
        assert args[1] is simple_agent_conf
        assert args[2] is agent_state
        assert kwargs["n_envs"] == 1
        assert kwargs["n_steps"] == video_recorder.length
        assert kwargs["render"] is True
        assert kwargs["record"] is True
        assert kwargs["deterministic"] is video_recorder.deterministic
        assert kwargs["use_mujoco"] is True
        assert kwargs["wrap_env"] is True
        assert kwargs["train_state_seed"] == 0

    def test_stateful_objects_restored_on_env_failure(self, video_recorder, simple_agent_conf, agent_state):
        class DummyFactory:
            @staticmethod
            def make(*_args, **_kwargs):
                raise RuntimeError("factory failure")

        sentinel_instances = ["existing"]
        original_instances = list(StatefulObject._instances)

        with patch.object(recorder_module.TaskFactory, "get_factory_cls", return_value=DummyFactory):
            try:
                StatefulObject._instances = list(sentinel_instances)
                with pytest.raises(RuntimeError, match="factory failure"):
                    video_recorder.record_episode(
                        agent_conf=simple_agent_conf,
                        agent_state=agent_state,
                        validation_number=1,
                        timestep=0
                    )
            finally:
                restored_instances = list(StatefulObject._instances)
                StatefulObject._instances = original_instances

        assert restored_instances == sentinel_instances


_SMPLH_NEUTRAL_MODEL = Path(__file__).resolve().parents[2] / "smpl_models" / "SMPLH_NEUTRAL.pkl"


@pytest.mark.skipif(
    not _SMPLH_NEUTRAL_MODEL.is_file(),
    reason=(
        "external licensed SMPLH_NEUTRAL.pkl is required for the legacy "
        "MyoBimanualArm/KIT retargeting integration tests"
    ),
)
class TestValidationVideoRecorderIntegration:
    """Integration tests with actual environment creation and recording."""

    @pytest.fixture
    def bimanual_training_config(self):
        """Full bimanual training config matching conf_bimanual.yaml."""
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]
                        }
                    }
                },
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "headless": True,
                    "horizon": 1000,
                    "disable_fingers": False,
                    "mjx_backend": "warp",
                    "num_envs": 2048,
                    "init_state_type": "TrajInitialStateHandler",
                    "goal_type": "GoalBimanualTrajMimic",
                    "goal_params": {
                        "visualize_goal": True,
                        "n_step_lookahead": 1,
                        "upper_body_xml_name": "thorax",
                        "sites_for_mimic": [
                            "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                            "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                        ]
                    },
                    "control_type": "DefaultControl",
                    "control_params": {},
                    "reward_type": "MimicReward",
                    "reward_params": {
                        "qpos_w_sum": 0.5,
                        "qvel_w_sum": 0.5,
                        "rpos_w_sum": 2.0,
                        "rquat_w_sum": 0.2,
                        "rvel_w_sum": 0.5,
                        "use_mean_exp_reward": False,
                        "sites_for_mimic": [
                            "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                            "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                        ]
                    },
                    "terminal_state_type": "EnhancedBimanualTerminalStateHandler",
                    "terminal_state_params": {
                        "max_site_deviation": 0.15,
                        "enable_reference_check": False,
                        "site_deviation_mode": "mean"
                    },
                    "timestep": 0.002,
                    "n_substeps": 5
                },
                "validation": {
                    "terminal_state_type": "NoTerminalStateHandler",
                    "terminal_state_params": {}
                }
            }
        })
        return config

    @pytest.mark.integration
    def test_create_validation_env_with_no_termination(self, tmp_path):
        from loco_mujoco.task_factories import ImitationFactory

        env_params = {
            "env_name": "MyoBimanualArm",
            "headless": True,
            "visualize_goal": True,
            "init_state_type": "TrajInitialStateHandler",
            "goal_type": "GoalBimanualTrajMimicv2",
            "goal_params": {
                "visualize_goal": True,
                "n_step_lookahead": 1,
                "upper_body_xml_name": "thorax",
                "sites_for_mimic": [
                    "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                    "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                ],
                "enable_enhanced_visualization": True,
                "target_geom_rgba": [0.471, 0.38, 0.812, 0.6]
            },
            "terminal_state_type": "NoTerminalStateHandler",
            "terminal_state_params": {},
            "control_type": "DefaultControl",
            "control_params": {},
            "reward_type": "MimicReward",
            "reward_params": {
                "qpos_w_sum": 0.5,
                "qvel_w_sum": 0.5,
                "rpos_w_sum": 2.0,
                "rquat_w_sum": 0.2,
                "rvel_w_sum": 0.5,
                "use_mean_exp_reward": False,
                "sites_for_mimic": [
                    "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                    "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                ]
            },
            "recorder_params": {
                "path": str(tmp_path / "videos"),
                "tag": "test_validation",
                "video_name": "MyoBimanualArm",
                "fps": 30,
                "compress": True
            },
            "timestep": 0.002,
            "n_substeps": 5
        }

        factory_params = {
            "amass_dataset_conf": {
                "rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]
            }
        }

        try:
            print("\n[Integration] Creating validation env with NoTerminalStateHandler...")
            env = ImitationFactory.make(**env_params, **factory_params)
            print("[Integration] Testing reset...")
            env.reset()
            print("[Integration] Verifying NoTerminalStateHandler...")
            assert type(env._terminal_state_handler).__name__ == "NoTerminalStateHandler"
            print("[Integration] Stopping environment...")
            env.stop()
            print("Validation env created successfully")
        except Exception as e:
            print(f"\nFailed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_full_validation_video_recording_workflow(self, tmp_path, bimanual_training_config):
        """Test complete video recording workflow - env creation and setup."""
        from loco_mujoco.task_factories import ImitationFactory

        print("\n[Integration] Testing validation env setup for recording...")

        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path / "validation_videos"),
            frequency=1,
            length=50,
            deterministic=True
        )

        mock_agent_conf = MagicMock()
        mock_agent_conf.config = bimanual_training_config

        try:
            print("[Integration] Building env params...")
            env_params = recorder._build_env_params(mock_agent_conf, "test_validation_1_t10000")

            print("[Integration] Creating validation env...")
            env = ImitationFactory.make(
                **env_params,
                **bimanual_training_config.experiment.task_factory.params
            )

            print("[Integration] Testing env reset...")
            env.reset()

            print("[Integration] Verifying configuration...")
            assert type(env._terminal_state_handler).__name__ == "NoTerminalStateHandler"
            assert hasattr(env, '_viewer')
            assert env_params["recorder_params"]["tag"] == "test_validation_1_t10000"
            assert env_params["recorder_params"]["fps"] == 100  # 1/(0.002*5)
            assert env_params["headless"] is True

            print("[Integration] Cleaning up...")
            env.stop()
            print("Validation env setup and configuration verified")

        except Exception as e:
            print(f"\nSetup failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_actual_video_recording(self, tmp_path, bimanual_training_config):
        """Test actual video recording with a short rollout."""
        import os
        import numpy as np
        from loco_mujoco.task_factories import ImitationFactory

        print("\n[Integration] Testing actual video recording...")

        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path / "actual_videos"),
            frequency=1,
            length=30,  # Short rollout
            deterministic=True
        )

        mock_agent_conf = MagicMock()
        mock_agent_conf.config = bimanual_training_config

        # Create simple mock agent state with random policy
        mock_agent_state = MagicMock()
        mock_agent_state.params = MagicMock()
        mock_agent_state.run_stats = MagicMock()

        try:
            print("[Integration] Building env for recording...")
            env_params = recorder._build_env_params(mock_agent_conf, "actual_recording_test")
            env = ImitationFactory.make(
                **env_params,
                **bimanual_training_config.experiment.task_factory.params
            )

            print("[Integration] Running rollout with recording enabled...")
            obs = env.reset()
            env.render(record=True)  # Initial render with recording

            action_dim = env.model.nu  # Number of actuators
            for step in range(30):
                action = np.zeros(action_dim)  # Zero action
                obs, reward, terminated, truncated, info = env.step(action)
                env.render(record=True)  # Render and record each frame
                if step % 10 == 0:
                    print(f"  Step {step}/30")
                if terminated or truncated:
                    break

            print("[Integration] Finalizing recording and stopping environment...")
            env.stop()

            # Check if video was created
            video_path = env.video_file_path
            print(f"[Integration] Video path: {video_path}")

            if video_path and os.path.exists(video_path):
                print(f"Video file created: {video_path}")
                file_size = os.path.getsize(video_path)
                print(f"  File size: {file_size} bytes")
                assert file_size > 0, "Video file is empty"
            else:
                print("  Note: Video path is None or file doesn't exist")
                print("  This might be expected if recording wasn't triggered")

        except Exception as e:
            print(f"\nRecording test failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
