from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


class MuscleMimicSceneEnv:
    """Run the augmented badminton XML through the original MyoFullBody env.

    The court, racket, and shuttle are present in the MuJoCo spec, but stepping,
    action preprocessing, observation construction, and substep handling come
    from the original musclemimic environment.
    """

    def __init__(
        self,
        xml: str | Path,
        *,
        disable_fingers: bool = False,
        headless: bool = True,
        timestep: float = 0.002,
        n_substeps: int = 5,
        body_checkpoint: str | Path | None = None,
        max_trajectories: int | None = 1,
    ) -> None:
        self.xml_path = Path(xml)
        self.source = "direct"
        if body_checkpoint is None:
            self.env = _make_direct_env(
                self.xml_path,
                disable_fingers=disable_fingers,
                headless=headless,
                timestep=timestep,
                n_substeps=n_substeps,
            )
        else:
            self.env = _make_imitation_env_from_checkpoint(
                self.xml_path,
                body_checkpoint,
                max_trajectories=max_trajectories,
            )
            self.source = "checkpoint_imitation_factory"
        self.model = self.env._model
        self.data = self.env._data

    @property
    def action_size(self) -> int:
        return int(self.env.info.action_space.shape[0])

    @property
    def control_substeps(self) -> int:
        return int(self.env._n_substeps)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        obs = np.asarray(self.env.reset(), dtype=float)
        return obs, self._info(obs)

    def step_action(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != self.env.info.action_space.shape:
            raise ValueError(
                f"musclemimic action must have shape {self.env.info.action_space.shape}, "
                f"got {action_array.shape}"
            )
        obs, reward, absorbing, done, info = self.env.step(action_array)
        merged_info = self._info(obs)
        merged_info.update(info)
        return np.asarray(obs, dtype=float), float(reward), bool(absorbing), bool(done), merged_info

    def sync_observation(self) -> np.ndarray:
        obs, carry = self.env._create_observation(self.model, self.data, self.env._additional_carry)
        self.env._obs = obs
        self.env._additional_carry = carry
        return np.asarray(obs, dtype=float)

    def _info(self, obs: np.ndarray) -> dict[str, Any]:
        return {
            "base_runtime": "musclemimic_myofullbody",
            "base_runtime_source": self.source,
            "base_obs_size": int(np.asarray(obs).size),
            "base_action_size": self.action_size,
            "base_control_substeps": self.control_substeps,
            "base_physics_timestep_s": float(self.model.opt.timestep),
            "base_policy_control_dt_s": float(self.model.opt.timestep * self.control_substeps),
            "has_overall_racket": mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "overall_racket",
            )
            >= 0,
        }


def _make_direct_env(
    xml_path: Path,
    *,
    disable_fingers: bool,
    headless: bool,
    timestep: float,
    n_substeps: int,
):
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    return MyoFullBody(
        spec=str(xml_path),
        disable_fingers=disable_fingers,
        headless=headless,
        timestep=timestep,
        n_substeps=n_substeps,
    )


def _make_imitation_env_from_checkpoint(
    xml_path: Path,
    checkpoint: str | Path,
    *,
    max_trajectories: int | None,
):
    from loco_mujoco.task_factories import ImitationFactory

    _set_local_dataset_defaults()
    checkpoint_path = Path(checkpoint)
    metadata_path = checkpoint_path / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint config metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    experiment = metadata["experiment"]
    env_params = dict(experiment["env_params"])
    env_params.update(
        {
            "env_name": "MyoFullBody",
            "disable_fingers": False,
            "headless": True,
            "spec": str(xml_path),
            "reward_type": "NoReward",
            "terminal_state_type": "NoTerminalStateHandler",
            "th_params": {
                "random_start": False,
                "fixed_start_conf": (0, 0),
                "start_from_random_step": False,
            },
        }
    )
    for key in (
        "mjx_backend",
        "num_envs",
        "nconmax",
        "njmax",
        "reward_params",
        "terminal_state_params",
    ):
        env_params.pop(key, None)

    task_params = dict(experiment.get("task_factory", {}).get("params", {}))
    amass_conf = task_params.get("amass_dataset_conf")
    if isinstance(amass_conf, dict) and max_trajectories is not None:
        limited_conf = dict(amass_conf)
        rel_paths = limited_conf.get("rel_dataset_path")
        if isinstance(rel_paths, list):
            limited_conf["rel_dataset_path"] = rel_paths[: max(1, int(max_trajectories))]
        task_params["amass_dataset_conf"] = limited_conf

    return ImitationFactory.make(**env_params, **task_params)


def _set_local_dataset_defaults() -> None:
    converted = REPO_ROOT / "caches" / "AMASS"
    amass = REPO_ROOT / "musclemimic" / "badminton" / "data" / "amass_npz"
    os.environ.setdefault("MUSCLEMIMIC_CONVERTED_AMASS_PATH", str(converted))
    os.environ.setdefault("CONVERTED_AMASS_PATH", str(converted))
    os.environ.setdefault("MUSCLEMIMIC_AMASS_PATH", str(amass))
    os.environ.setdefault("AMASS_PATH", str(amass))
