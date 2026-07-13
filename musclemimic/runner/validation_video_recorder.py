"""Validation video recording utilities for training workflows."""

import json
import time
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.badminton.visual_review import (
    CANDIDATE_SCHEMA_VERSION,
    REVIEW_KINDS,
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
)
from musclemimic.utils import setup_headless_rendering_if_needed
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.task_factories import TaskFactory


def _candidate_motion_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/").rstrip("/")
    return normalized[:-4] if normalized.endswith(".npz") else normalized


class ValidationVideoRecorder:
    """
    Host-side utility to record short evaluation rollouts during training.

    It reconstructs a standalone env (MJX by default) with headless rendering and
    invokes PPOJax.play_policy(record=True) to write a video using the built-in
    VideoRecorder. Designed to be called from the training logging callback.
    """

    def __init__(
        self,
        video_dir: str,
        frequency: int = 10,
        length: int = 500,
        deterministic: bool = True,
        cycle_trajectories: bool = False,
        max_recordings: int | None = None,
        review_kind: str | None = None,
    ):
        """
        Args:
            video_dir: Base directory where videos are written.
            frequency: Record every N validation callbacks.
            length: Number of steps to record per episode.
            deterministic: Use deterministic policy for reproducibility.
        """
        self.video_dir = video_dir
        self.frequency = max(1, int(frequency))
        self.length = max(1, int(length))
        self.deterministic = deterministic
        self.cycle_trajectories = bool(cycle_trajectories)
        self.max_recordings = (
            None if max_recordings is None else max(0, int(max_recordings))
        )
        if review_kind is not None and review_kind not in REVIEW_KINDS:
            raise ValueError(f"unsupported validation visual review kind: {review_kind!r}")
        self.review_kind = review_kind

    def _selected_validation_motion(self, agent_conf, validation_number: int):
        if not self.cycle_trajectories:
            return None
        validation = getattr(agent_conf.config.experiment, "validation", {})
        dataset = validation.get("amass_dataset_conf", {})
        paths = list(dataset.get("rel_dataset_path", []) or [])
        if not paths:
            return None
        recording_number = max(0, int(validation_number) // self.frequency - 1)
        trajectory_index = recording_number % len(paths)
        return trajectory_index, str(paths[trajectory_index])

    def _write_visual_candidate(
        self,
        *,
        motion: str,
        artifact: str,
        validation_number: int,
        timestep: int | None,
        candidate_identity: Mapping | None = None,
    ) -> None:
        if self.review_kind is None:
            return
        prefix = "stage1" if self.review_kind == STAGE1_REVIEW_KIND else "stage2"
        path = Path(self.video_dir) / f"{prefix}_visual_review_candidates.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "review_kind": self.review_kind,
            "motion": motion,
            "artifact": artifact,
            "validation_number": int(validation_number),
            "timestep": None if timestep is None else int(timestep),
            "major_swing_complete": None,
            "root_tracking_spike_free": None,
            "right_hand_tracking_spike_free": None,
            "passed": None,
            "notes": None,
        }
        if candidate_identity is not None:
            row["candidate"] = dict(candidate_identity)
        if self.review_kind == STAGE2_REVIEW_KIND:
            row.update(
                {
                    "racket_head_trajectory_ok": None,
                    "racket_face_orientation_ok": None,
                }
            )

        existing: list[dict] = []
        if path.is_file():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    previous = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid visual candidate JSONL at {path}:{line_number}"
                    ) from exc
                # Legacy candidate rows are readable historical output, but
                # cannot enter the structured production manifest.
                if (
                    isinstance(previous, dict)
                    and previous.get("schema_version") == CANDIDATE_SCHEMA_VERSION
                    and previous.get("review_kind") == self.review_kind
                ):
                    existing.append(previous)

        motion_key = _candidate_motion_key(motion)
        existing = [
            previous
            for previous in existing
            if _candidate_motion_key(str(previous.get("motion", ""))) != motion_key
        ]
        for previous in existing:
            if str(previous.get("artifact", "")).strip() == artifact.strip():
                raise ValueError(
                    "visual candidate artifact is already assigned to another motion: "
                    f"{artifact}"
                )
        existing.append(row)
        motion_keys = [_candidate_motion_key(str(item.get("motion", ""))) for item in existing]
        artifacts = [str(item.get("artifact", "")).strip() for item in existing]
        if len(set(motion_keys)) != len(motion_keys):
            raise ValueError("visual candidate motions must remain unique")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("visual candidate artifacts must remain unique")

        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in existing
            ),
            encoding="utf-8",
        )
        temp.replace(path)

    def _build_env_params(self, agent_conf, tag: str) -> dict:
        """Clone training env params and add recording-specific settings."""
        # Copy training env params.
        env_params = dict(agent_conf.config.experiment.env_params)

        # Switch to the MuJoCo CPU env for recording.
        env_name = env_params.get("env_name", "")
        if isinstance(env_name, str) and env_name.startswith("Mjx"):
            env_params["env_name"] = env_name.replace("Mjx", "", 1)
        # Drop MJX-only parameters.
        for k in ("mjx_backend", "num_envs", "nconmax", "njmax"):
            if k in env_params:
                env_params.pop(k, None)

        # Apply validation terminal-state settings.
        if hasattr(agent_conf.config.experiment, "validation"):
            validation_config = agent_conf.config.experiment.validation
            env_params["terminal_state_type"] = validation_config.get("terminal_state_type", "NoTerminalStateHandler")
            env_params["terminal_state_params"] = dict(validation_config.get("terminal_state_params", {}))
        else:
            env_params["terminal_state_type"] = "NoTerminalStateHandler"
            env_params["terminal_state_params"] = {}

        # Configure headless recording.
        env_params["headless"] = True
        # Enable goal visualization during recording.
        env_params["visualize_goal"] = True
        # Match recording FPS to the control rate.
        timestep = env_params.get("timestep", 0.002)
        n_substeps = env_params.get("n_substeps", 5)
        control_dt = timestep * n_substeps
        fps = int(round(1.0 / control_dt))
        env_params["recorder_params"] = {
            "path": self.video_dir,
            "tag": tag,
            "video_name": f"{env_params.get('env_name', 'env')}",
            "fps": fps,
            "compress": True,
        }

        # Mirror visualization settings into goal_params.
        goal_params = dict(env_params.get("goal_params", {}))
        goal_params["visualize_goal"] = True
        # Enable enhanced goal visualization when supported.
        goal_params.setdefault("enable_enhanced_visualization", True)
        goal_params.setdefault("target_geom_rgba", [0.471, 0.38, 0.812, 0.6])
        env_params["goal_params"] = goal_params

        # Use visualization-specific goal classes.
        env_name = env_params.get("env_name", "")
        sites = goal_params.get("sites_for_mimic", [])
        if "Bimanual" in env_name:
            env_params["goal_type"] = "GoalBimanualTrajMimicv2"
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites
        elif "MyoFullBody" in env_name:
            # Fullbody uses GoalTrajMimicv2.
            env_params["goal_type"] = "GoalTrajMimicv2"
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites

        # Reuse training timing parameters.
        for k in ("timestep", "n_substeps"):
            if k in agent_conf.config.experiment.env_params:
                env_params[k] = agent_conf.config.experiment.env_params[k]

        # Start each validation rollout at trajectory step 0.
        if hasattr(agent_conf.config.experiment, "validation"):
            if agent_conf.config.experiment.validation.get("start_from_beginning", False):
                if "th_params" not in env_params:
                    env_params["th_params"] = {}
                env_params["th_params"]["start_from_random_step"] = False

        return env_params

    def _build_task_params(self, agent_conf) -> dict:
        """Clone task params and apply validation-specific dataset overrides."""
        raw_task_params = agent_conf.config.experiment.task_factory.params
        if OmegaConf.is_config(raw_task_params):
            task_params = OmegaConf.to_container(raw_task_params, resolve=True)
        else:
            task_params = dict(raw_task_params) if raw_task_params else {}

        if hasattr(agent_conf.config.experiment, "validation"):
            validation_config = agent_conf.config.experiment.validation
            for key in ("amass_dataset_conf", "dataset_conf", "trajectory_dataset_conf"):
                val_dataset = validation_config.get(key, None)
                if val_dataset is not None:
                    task_params[key] = (
                        OmegaConf.to_container(val_dataset, resolve=True)
                        if OmegaConf.is_config(val_dataset)
                        else val_dataset
                    )

        amass_conf = task_params.get("amass_dataset_conf")
        if isinstance(amass_conf, dict):
            amass_conf = dict(amass_conf)
            motion_paths = list(amass_conf.get("rel_dataset_path", []) or [])
            if self.cycle_trajectories and motion_paths:
                # Every held-out motion must be present so the scheduled
                # recorder cycle can generate one candidate for each clip.
                amass_conf["max_motions"] = len(motion_paths)
            else:
                amass_conf.setdefault("max_motions", 3)
            task_params["amass_dataset_conf"] = amass_conf
        return task_params

    def record_episode(
        self,
        agent_conf,
        agent_state,
        validation_number: int,
        timestep: int | None = None,
        *,
        motion_index: int | None = None,
        candidate_identity: Mapping | None = None,
    ) -> str | None:
        """
        Record a single short rollout if the frequency condition matches.

        Args:
            agent_conf: PPO agent configuration (contains network and saved config).
            agent_state: PPO agent state; only params and run_stats are used.
            validation_number: Current validation counter (1-based).
            timestep: Global training timestep for naming.

        Returns:
            Path to the recorded video file if available, else None.
        """
        if motion_index is None and validation_number % self.frequency != 0:
            return None
        recording_number = int(validation_number) // self.frequency
        if (
            motion_index is None
            and self.max_recordings is not None
            and recording_number > self.max_recordings
        ):
            return None

        setup_headless_rendering_if_needed()

        # Always use MuJoCo CPU env for evaluation visualization
        use_mujoco = True

        # Build a recording tag.
        time_tag = time.strftime("%Y%m%d_%H%M%S")
        if motion_index is None:
            selected = self._selected_validation_motion(agent_conf, validation_number)
        else:
            validation = getattr(agent_conf.config.experiment, "validation", {})
            dataset = validation.get("amass_dataset_conf", {})
            paths = list(dataset.get("rel_dataset_path", []) or [])
            if not 0 <= int(motion_index) < len(paths):
                raise ValueError(
                    "review-set motion index is outside the validation motion list"
                )
            selected = (int(motion_index), str(paths[int(motion_index)]))
        trajectory_tag = "" if selected is None else f"_traj{selected[0]}"
        tag = (
            f"validation_{validation_number}{trajectory_tag}_"
            f"t{timestep if timestep is not None else 0}_{time_tag}"
        )

        # Build the evaluation environment.
        factory = TaskFactory.get_factory_cls(agent_conf.config.experiment.task_factory.name)
        env_params = self._build_env_params(agent_conf, tag)
        task_params = self._build_task_params(agent_conf)

        # Isolate StatefulObject indices for the recorder env.
        saved_instances = StatefulObject._instances.copy()
        StatefulObject._instances.clear()
        env = None
        try:
            print(f"[ValidationVideo] Building eval env for recording (tag={tag})...")
            # Create the recorder environment.
            try:
                env = factory.make(**env_params, **task_params)
            except Exception as e:
                print(f"[ValidationVideo] ERROR: Failed to create environment: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise

            if selected is not None:
                trajectory_index, motion_path = selected
                if not hasattr(env, "th") or env.th is None:
                    raise ValueError(
                        "cycle_trajectories requires a trajectory-backed validation environment"
                    )
                if trajectory_index >= int(env.th.n_trajectories):
                    raise ValueError(
                        "selected validation trajectory is absent from recorder environment: "
                        f"index={trajectory_index} count={env.th.n_trajectories}"
                    )
                env.th.fixed_start_conf = [int(trajectory_index), 0]
                env.th.use_fixed_start = True
                env.th.random_start = False

            fps = env_params["recorder_params"]["fps"]
            print(f"[ValidationVideo] Eval env ready; starting rollout for {self.length} steps @ {fps} fps")

            # Keep isolation active through the internal reset in play_policy.
            PPOJax.play_policy(
                env,
                agent_conf,
                agent_state,
                n_envs=1,
                n_steps=self.length,
                render=True,  # must be True for recording to emit frames
                record=True,
                deterministic=self.deterministic,
                use_mujoco=use_mujoco,
                wrap_env=True,
                train_state_seed=0,
            )
        finally:
            # Restore global state even on failure.
            if env is not None:
                env.stop()
            StatefulObject._instances = saved_instances

        # Return the recorded video path when available.
        video_path = env.video_file_path if env is not None else None
        if video_path and selected is not None:
            self._write_visual_candidate(
                motion=selected[1],
                artifact=str(video_path),
                validation_number=validation_number,
                timestep=timestep,
                candidate_identity=candidate_identity,
            )
        return video_path

    def record_review_set(
        self,
        *,
        agent_conf,
        agent_state,
        validation_number: int,
        timestep: int,
        candidate_identity: Mapping,
    ) -> list[str]:
        """Record every held-out motion from one frozen promotion candidate."""

        if self.review_kind is None:
            raise ValueError("review-set recording requires a structured review kind")
        validation = getattr(agent_conf.config.experiment, "validation", {})
        paths = list(
            validation.get("amass_dataset_conf", {}).get("rel_dataset_path", [])
            or []
        )
        if not paths:
            raise ValueError("review-set recording has no held-out validation motions")
        artifacts: list[str] = []
        for motion_index in range(len(paths)):
            artifact = self.record_episode(
                agent_conf=agent_conf,
                agent_state=agent_state,
                validation_number=validation_number,
                timestep=timestep,
                motion_index=motion_index,
                candidate_identity=candidate_identity,
            )
            if not artifact:
                raise RuntimeError(
                    f"review-set recording produced no artifact for {paths[motion_index]}"
                )
            artifacts.append(artifact)
        return artifacts
