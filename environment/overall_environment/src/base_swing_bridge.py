"""Base-policy bridge for the incoming-hit task (Stage 3).

Wraps a distilled tracking base policy (frozen body policy export) so it can
drive the muscle body while a residual PPO head learns to hit the shuttle.
Composition follows the OverallGripHoldEnv convention (normalized action
space, clip to [-1, 1], then the env's ctrl scaling):

    a_full = clip( base(body_obs) + residual_scale * delta, -1, 1 )

Adaptations vs the mainline distilled policy:

- **No future trajectory.** The base was distilled with the goal lookahead
  dropped (StudentObservationFilterWrapper); its goal segment is just the
  motion phase (plus optional condition slots). Here phase is synthesized from
  the shuttle's time-to-intercept so the base plays its swing timed to the
  incoming ball; remaining goal slots are zero-filled.
- **Ball position** enters the residual head's observation (the hitting env
  obs), not the base body observation.

Pure NumPy on the CPU env; ``jax_arrays()`` exports the actor tensors for the
GPU training loop.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter
from environment.overall_environment.src.action_manifest import load_action_manifest
from environment.overall_environment.src.body_obs_adapter import BodyObsAdapter, BodyObsSchema
from environment.overall_environment.src.frozen_body_policy import (
    FrozenBodyPolicy,
    load_frozen_body_policy_manifest,
)


@dataclass
class SwingPhaseConfig:
    """Synthesized motion phase driven by the feed's intercept time.

    phase(t) ramps 0 -> 1 over ``swing_duration_s`` such that ``contact_phase``
    is reached exactly at the predicted intercept time. Before the swing starts
    the phase holds at 0 (ready pose), after completion it holds at 1.
    """

    swing_duration_s: float = 1.2
    contact_phase: float = 0.55

    def phase_at(self, elapsed_s: float, intercept_time_s: float) -> float:
        start = intercept_time_s - self.contact_phase * self.swing_duration_s
        return float(np.clip((elapsed_s - start) / self.swing_duration_s, 0.0, 1.0))


class BaseSwingBridge:
    """Frozen distilled base policy driving the body inside the hitting env."""

    def __init__(
        self,
        artifact_dir: str | Path,
        model: mujoco.MjModel,
        *,
        residual_scale: float = 0.3,
        phase_config: SwingPhaseConfig | None = None,
        skill: str | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.residual_scale = float(residual_scale)
        self.phase_config = phase_config if phase_config is not None else SwingPhaseConfig()

        manifest = load_frozen_body_policy_manifest(self.artifact_dir)
        self.policy = FrozenBodyPolicy.load_from_export(self.artifact_dir)

        schema_payload: dict[str, Any] = json.loads(
            (self.artifact_dir / manifest.body_obs_schema_file).read_text(encoding="utf-8")
        )
        if schema_payload.get("synthetic"):
            raise ValueError(
                "synthetic frozen-policy exports carry no body observation schema; "
                "export from a real checkpoint or provide a schema JSON"
            )
        schema_payload = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in schema_payload.items()
        }
        self.schema = BodyObsSchema(**schema_payload)
        self.obs_adapter = BodyObsAdapter(expected_obs_size=self.schema.total_size, schema=self.schema)

        # multi-skill exports append a skill one-hot after the student obs
        self.skill_actions: list[str] = []
        self.skill_onehot = np.zeros(0, dtype=float)
        skill_manifest_path = self.artifact_dir / "skill_manifest.json"
        if skill_manifest_path.is_file():
            payload = json.loads(skill_manifest_path.read_text(encoding="utf-8"))
            self.skill_actions = list(payload.get("actions", []))
            condition_size = int(payload.get("condition_size", 0))
            if condition_size:
                if skill is None:
                    raise ValueError(
                        f"multi-skill base requires skill selection from {self.skill_actions}"
                    )
                if skill not in self.skill_actions:
                    raise ValueError(f"unknown skill {skill!r}; available {self.skill_actions}")
                self.skill_onehot = np.zeros(condition_size, dtype=float)
                self.skill_onehot[self.skill_actions.index(skill)] = 1.0
        expected = self.schema.total_size + self.skill_onehot.size
        if int(self.policy.actor_spec.obs_size) != expected:
            raise ValueError(
                f"base policy obs size {self.policy.actor_spec.obs_size} != schema+condition {expected}"
            )

        action_manifest = load_action_manifest(self.artifact_dir / manifest.action_manifest_file)
        target_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
        ]
        self.action_adapter = CheckpointToFullActionAdapter(
            list(action_manifest.actuator_names), target_names
        )
        self.goal_size = int(self.schema.goal_size)
        self.nu = int(model.nu)

    # ---- inference ---------------------------------------------------------

    def synth_goal(self, phase: float) -> np.ndarray:
        """Goal segment for the distilled student: motion phase in the last slot."""
        goal = np.zeros(self.goal_size, dtype=float)
        if self.goal_size >= 1:
            goal[-1] = float(np.clip(phase, 0.0, 1.0))
        return goal

    def base_action(self, model: mujoco.MjModel, data: mujoco.MjData, *, phase: float) -> np.ndarray:
        """Full-model normalized action ([-1,1] space) from the frozen base."""
        body_obs = self.obs_adapter.build_from_mujoco(model, data, goal_obs=self.synth_goal(phase))
        if self.skill_onehot.size:
            body_obs = np.concatenate([body_obs, self.skill_onehot])
        return self.action_adapter.adapt(self.policy.act(body_obs))

    def combined_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        residual: np.ndarray,
        *,
        phase: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (full normalized action, base action). Env applies ctrl scaling."""
        base = self.base_action(model, data, phase=phase)
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (self.nu,):
            raise ValueError(f"residual must have shape ({self.nu},), got {residual.shape}")
        combined = np.clip(base + self.residual_scale * residual, -1.0, 1.0)
        return combined, base

    # ---- GPU export ---------------------------------------------------------

    def jax_arrays(self) -> dict[str, Any]:
        """Actor tensors + index maps for the JAX/MJX residual training loop."""
        spec = self.policy.actor_spec
        stats = self.policy.run_stats["RunningMeanStd_0"]
        layers = []
        actor = self.policy.params["actor"]
        for index in range(len(spec.actor_hidden_layers) + 1):
            dense = actor[f"Dense_{index}"]
            layer = {
                "kernel": np.asarray(dense["kernel"], dtype=np.float32),
                "bias": np.asarray(dense["bias"], dtype=np.float32),
            }
            if spec.use_layernorm and index < len(spec.actor_hidden_layers):
                layernorm = actor[f"LayerNorm_{index}"]
                layer["ln_scale"] = np.asarray(layernorm["scale"], dtype=np.float32)
                layer["ln_bias"] = np.asarray(layernorm["bias"], dtype=np.float32)
            layers.append(layer)
        return {
            "layers": layers,
            "activation": spec.activation,
            "use_layernorm": bool(spec.use_layernorm),
            "layernorm_eps": float(spec.layernorm_eps),
            "obs_mean": np.asarray(stats["mean"], dtype=np.float32),
            "obs_var": np.asarray(stats["var"], dtype=np.float32),
            "obs_size": int(spec.obs_size),
            "action_size": int(spec.action_size),
            "goal_size": self.goal_size,
            "skill_onehot": np.asarray(self.skill_onehot, dtype=np.float32),
        }
