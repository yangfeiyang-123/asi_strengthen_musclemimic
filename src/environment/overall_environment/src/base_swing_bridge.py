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

import hashlib
import json
import sys
from collections.abc import Mapping
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
    contact_phase: float = 0.76
    # Measured actuator/body response lag between the distilled phase command
    # and the physical racket sweep.  A positive value advances the frozen
    # swing while leaving the task's true shuttle intercept time unchanged.
    phase_advance_s: float = 0.0

    def phase_at(self, elapsed_s: float, intercept_time_s: float) -> float:
        start = intercept_time_s - self.phase_advance_s - self.contact_phase * self.swing_duration_s
        return float(np.clip((elapsed_s - start) / self.swing_duration_s, 0.0, 1.0))


def selected_correction_window(
    time_to_intercept: Any,
    *,
    open_s: float,
    close_s: float,
    smoothing_s: float,
    array_module: Any = np,
) -> Any:
    """Smoothly gate a local correction around the predicted impact.

    ``time_to_intercept`` decreases with time. The gate opens at ``open_s``
    before impact and closes at ``close_s`` (normally a small negative value)
    after impact. The implementation is shared by NumPy tests/CPU tools and
    JAX training by supplying ``jax.numpy`` as ``array_module``.
    """

    open_value = float(open_s)
    close_value = float(close_s)
    smoothing = float(smoothing_s)
    if not np.isfinite([open_value, close_value, smoothing]).all():
        raise ValueError("correction window values must be finite")
    if open_value <= close_value:
        raise ValueError("correction window open_s must be greater than close_s")
    if smoothing < 0.0 or smoothing * 2.0 > open_value - close_value:
        raise ValueError("correction window smoothing_s is outside the valid interval")

    xp = array_module
    tti = xp.asarray(time_to_intercept)
    if smoothing == 0.0:
        return ((tti <= open_value) & (tti >= close_value)).astype(tti.dtype)

    def smoothstep(value: Any) -> Any:
        clipped = xp.clip(value, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    opening = smoothstep((open_value - tti) / smoothing)
    closing = smoothstep((tti - close_value) / smoothing)
    return opening * closing


def interpolate_correction_prior(
    time_to_intercept: Any,
    *,
    knot_time_to_intercept_s: Any,
    knot_correction_raw: Any,
    array_module: Any = np,
) -> Any:
    """Linearly replay a sealed correction trajectory in intercept time.

    The CEM teacher is fundamentally a time-indexed control trajectory.  A
    neural BC head fitted only on the teacher's visited observations can have
    tiny pointwise MSE yet leave that state manifold after one control step.
    This interpolation supplies the verified open-loop trajectory as a frozen
    prior; the learned head can then represent only state-feedback deltas.

    Knots must be strictly increasing in ``time_to_intercept``.  Queries beyond
    the recorded interval clamp to the endpoint; the independent correction
    window remains responsible for smoothly turning physical authority off.
    """

    xp = array_module
    query = xp.asarray(time_to_intercept)
    knot_t = xp.asarray(knot_time_to_intercept_s)
    knot_raw = xp.asarray(knot_correction_raw)
    if knot_t.ndim != 1 or knot_raw.ndim != 2:
        raise ValueError("correction prior requires 1-D times and 2-D actions")
    if knot_raw.shape[0] != knot_t.shape[0] or knot_t.shape[0] < 2:
        raise ValueError("correction prior knot arrays have incompatible shapes")

    upper = xp.searchsorted(knot_t, query, side="right")
    upper = xp.clip(upper, 1, knot_t.shape[0] - 1)
    lower = upper - 1
    lower_t = knot_t[lower]
    upper_t = knot_t[upper]
    fraction = xp.clip(
        (query - lower_t) / xp.maximum(upper_t - lower_t, 1.0e-12),
        0.0,
        1.0,
    )
    lower_raw = knot_raw[lower]
    upper_raw = knot_raw[upper]
    return lower_raw + fraction[..., None] * (upper_raw - lower_raw)


def compose_selected_physical_correction(
    inherited_residual: Any,
    correction_raw: Any,
    *,
    selected_indices: tuple[int, ...] | list[int],
    physical_scales: Any,
    inherited_residual_scale: Any,
    window: Any,
    array_module: Any = np,
) -> Any:
    """Map an independent selected correction into the legacy residual ABI.

    The environment still evaluates ``base + residual_scale * residual``.
    This helper returns a full residual whose physical result is exactly

    ``base + residual_scale * inherited + M(window*alpha*tanh(correction))``.

    In particular, the inherited actor is squashed before this function and
    is never added to the new logits. This avoids the coupled-tanh defect of
    the former refinement adapter while keeping the environment/action ABI
    backwards compatible.
    """

    xp = array_module
    inherited = xp.asarray(inherited_residual)
    correction = xp.asarray(correction_raw)
    indices = tuple(int(index) for index in selected_indices)
    if inherited.ndim < 1 or correction.ndim < 1:
        raise ValueError("inherited_residual and correction_raw require an action axis")
    if correction.shape[-1] != len(indices):
        raise ValueError("correction_raw width does not match selected_indices")
    if len(set(indices)) != len(indices):
        raise ValueError("selected_indices must not contain duplicates")
    if indices and (min(indices) < 0 or max(indices) >= inherited.shape[-1]):
        raise ValueError("selected correction index is outside the full action")
    if inherited.shape[:-1] != correction.shape[:-1]:
        raise ValueError("inherited and correction batch shapes must match")

    scales = xp.asarray(physical_scales, dtype=inherited.dtype)
    if scales.shape != (len(indices),):
        raise ValueError("physical_scales must contain one value per selected action")
    residual_scale = xp.asarray(inherited_residual_scale, dtype=inherited.dtype)
    if residual_scale.ndim == 0:
        residual_scale = xp.broadcast_to(residual_scale, (inherited.shape[-1],))
    if residual_scale.shape != (inherited.shape[-1],):
        raise ValueError("inherited_residual_scale must be scalar or full-action sized")
    if isinstance(residual_scale, np.ndarray) and np.any(residual_scale <= 0.0):
        raise ValueError("inherited_residual_scale must be positive")

    selector = xp.eye(inherited.shape[-1], dtype=inherited.dtype)[xp.asarray(indices)]
    physical_selected = (
        xp.asarray(window, dtype=inherited.dtype)[..., None]
        * scales
        * xp.tanh(correction)
    )
    physical_full = physical_selected @ selector
    return inherited + physical_full / residual_scale


class BaseSwingBridge:
    """Frozen distilled base policy driving the body inside the hitting env."""

    def __init__(
        self,
        artifact_dir: str | Path,
        model: mujoco.MjModel,
        *,
        residual_scale: float = 0.3,
        residual_scale_overrides: Mapping[str, float] | None = None,
        residual_scale_schedule: Mapping[str, Any] | None = None,
        phase_config: SwingPhaseConfig | None = None,
        skill: str | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.residual_scale = self._validated_residual_scale(
            residual_scale,
            label="residual_scale",
        )
        self.phase_config = phase_config if phase_config is not None else SwingPhaseConfig()
        self.selected_skill = skill

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
            key: tuple(value) if isinstance(value, list) else value for key, value in schema_payload.items()
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
                    raise ValueError(f"multi-skill base requires skill selection from {self.skill_actions}")
                if skill not in self.skill_actions:
                    raise ValueError(f"unknown skill {skill!r}; available {self.skill_actions}")
                self.skill_onehot = np.zeros(condition_size, dtype=float)
                self.skill_onehot[self.skill_actions.index(skill)] = 1.0
        expected = self.schema.total_size + self.skill_onehot.size
        if int(self.policy.actor_spec.obs_size) != expected:
            raise ValueError(f"base policy obs size {self.policy.actor_spec.obs_size} != schema+condition {expected}")

        action_manifest = load_action_manifest(self.artifact_dir / manifest.action_manifest_file)
        target_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        target_name_to_id = {str(name): index for index, name in enumerate(target_names)}
        raw_overrides = {} if residual_scale_overrides is None else dict(residual_scale_overrides)
        unknown = sorted(str(name) for name in raw_overrides if str(name) not in target_name_to_id)
        if unknown:
            raise ValueError("residual_scale_overrides contains unknown model actuators: " + ", ".join(unknown))
        self.residual_scale_overrides = {
            str(name): self._validated_residual_scale(
                value,
                label=f"residual_scale_overrides[{name!r}]",
            )
            for name, value in sorted(raw_overrides.items(), key=lambda item: str(item[0]))
        }
        self.residual_scale_vector = np.full(model.nu, self.residual_scale, dtype=np.float64)
        for name, scale in self.residual_scale_overrides.items():
            self.residual_scale_vector[target_name_to_id[name]] = scale
        self.residual_override_indices = np.asarray(
            [target_name_to_id[name] for name in self.residual_scale_overrides],
            dtype=np.int32,
        )
        raw_schedule = {} if residual_scale_schedule is None else dict(residual_scale_schedule)
        if raw_schedule:
            unknown_schedule_keys = sorted(set(raw_schedule) - {"initial_scale", "ramp_steps"})
            if unknown_schedule_keys:
                raise ValueError("residual_scale_schedule contains unknown keys: " + ", ".join(unknown_schedule_keys))
            if not self.residual_scale_overrides:
                raise ValueError("residual_scale_schedule requires residual_scale_overrides")
            if "initial_scale" not in raw_schedule or "ramp_steps" not in raw_schedule:
                raise ValueError("residual_scale_schedule requires initial_scale and ramp_steps")
            self.residual_schedule_initial_scale = self._validated_residual_scale(
                raw_schedule["initial_scale"],
                label="residual_scale_schedule.initial_scale",
            )
            if isinstance(raw_schedule["ramp_steps"], bool):
                raise ValueError("residual_scale_schedule.ramp_steps must be a positive integer")
            try:
                self.residual_schedule_ramp_steps = int(raw_schedule["ramp_steps"])
            except (TypeError, ValueError) as exc:
                raise ValueError("residual_scale_schedule.ramp_steps must be a positive integer") from exc
            if self.residual_schedule_ramp_steps <= 0 or float(raw_schedule["ramp_steps"]) != float(
                self.residual_schedule_ramp_steps
            ):
                raise ValueError("residual_scale_schedule.ramp_steps must be a positive integer")
        else:
            self.residual_schedule_initial_scale = None
            self.residual_schedule_ramp_steps = 0
        self.residual_scale_initial_vector = self.residual_scale_vector.copy()
        if self.residual_schedule_initial_scale is not None:
            self.residual_scale_initial_vector[self.residual_override_indices] = self.residual_schedule_initial_scale
        self.action_adapter = CheckpointToFullActionAdapter(list(action_manifest.actuator_names), target_names)
        self.goal_size = int(self.schema.goal_size)
        self.nu = int(model.nu)
        self.control_binding = self._build_control_binding(manifest)

    @staticmethod
    def _validated_residual_scale(value: float, *, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a finite number in [0, 2]")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a finite number in [0, 2]") from exc
        if not np.isfinite(result) or not 0.0 <= result <= 2.0:
            raise ValueError(f"{label} must be a finite number in [0, 2]")
        return result

    def _build_control_binding(self, manifest: Any) -> dict[str, Any]:
        """Seal every frozen-policy input that changes residual control.

        Stage-3 checkpoints previously recorded only the direct policy/router
        ABI. A resume or evaluation could therefore silently use a different
        frozen swing (or omit it entirely) while retaining the same control
        hash. The content inventory below makes the base policy, selected
        skill, residual scale and swing clock part of that hash.
        """

        roles = {
            "manifest": "manifest.json",
            "params": manifest.params_file,
            "run_stats": manifest.run_stats_file,
            "body_obs_schema": manifest.body_obs_schema_file,
            "action_manifest": manifest.action_manifest_file,
        }
        if (self.artifact_dir / "skill_manifest.json").is_file():
            roles["skill_manifest"] = "skill_manifest.json"
        files: list[dict[str, Any]] = []
        for role, relative in roles.items():
            path = self.artifact_dir / relative
            if not path.is_file():
                raise ValueError(f"frozen base policy is missing {role}: {path}")
            files.append(
                {
                    "role": role,
                    "path": str(relative),
                    "num_bytes": int(path.stat().st_size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        content_payload = {
            "schema_version": "frozen_base_policy_content_v1",
            "files": files,
        }
        content_sha256 = hashlib.sha256(
            json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        binding = {
            "schema_version": "incoming_hit_frozen_base_residual_v1",
            "artifact_content_sha256": content_sha256,
            "source_checkpoint": str(manifest.source_checkpoint),
            "selected_skill": self.selected_skill,
            "residual_scale": float(self.residual_scale),
            "swing_duration_s": float(self.phase_config.swing_duration_s),
            "contact_phase": float(self.phase_config.contact_phase),
            "phase_advance_s": float(self.phase_config.phase_advance_s),
            "actor_obs_size": int(manifest.actor_spec.obs_size),
            "actor_action_size": int(manifest.actor_spec.action_size),
            "files": files,
        }
        # Preserve legacy checkpoint hashes when the optional per-actuator
        # authority map is not configured.  New runs seal both exact names and
        # model indices so a reordered or silently broadened override cannot
        # resume/evaluate under the same control ABI.
        if self.residual_scale_overrides:
            binding["residual_scale_overrides"] = [
                {
                    "actuator_name": name,
                    "actuator_id": int(index),
                    "scale": float(self.residual_scale_vector[index]),
                }
                for name, index in zip(
                    self.residual_scale_overrides,
                    self.residual_override_indices,
                    strict=True,
                )
            ]
            binding["residual_scale_vector_sha256"] = hashlib.sha256(
                np.asarray(self.residual_scale_vector, dtype="<f8").tobytes()
            ).hexdigest()
        if self.residual_schedule_initial_scale is not None:
            binding["residual_scale_schedule"] = {
                "schema_version": "incoming_hit_residual_authority_schedule_v1",
                "interpolation": "linear_env_steps",
                "initial_scale": float(self.residual_schedule_initial_scale),
                "ramp_steps": int(self.residual_schedule_ramp_steps),
                "scheduled_actuators": [
                    {
                        "actuator_name": name,
                        "actuator_id": int(index),
                        "initial_scale": float(self.residual_scale_initial_vector[index]),
                        "target_scale": float(self.residual_scale_vector[index]),
                    }
                    for name, index in zip(
                        self.residual_scale_overrides,
                        self.residual_override_indices,
                        strict=True,
                    )
                ],
                "initial_scale_vector_sha256": hashlib.sha256(
                    np.asarray(self.residual_scale_initial_vector, dtype="<f8").tobytes()
                ).hexdigest(),
                "target_scale_vector_sha256": hashlib.sha256(
                    np.asarray(self.residual_scale_vector, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        binding["binding_sha256"] = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return binding

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
        residual_authority_progress: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (full normalized action, base action). Env applies ctrl scaling."""
        base = self.base_action(model, data, phase=phase)
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (self.nu,):
            raise ValueError(f"residual must have shape ({self.nu},), got {residual.shape}")
        progress = float(residual_authority_progress)
        if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("residual_authority_progress must be finite and lie in [0, 1]")
        scale_vector = self.residual_scale_initial_vector + progress * (
            self.residual_scale_vector - self.residual_scale_initial_vector
        )
        combined = np.clip(base + scale_vector * residual, -1.0, 1.0)
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
            "residual_scale_vector": np.asarray(self.residual_scale_vector, dtype=np.float32),
            "residual_scale_initial_vector": np.asarray(
                self.residual_scale_initial_vector,
                dtype=np.float32,
            ),
            "residual_scale_ramp_steps": int(self.residual_schedule_ramp_steps),
            "residual_override_indices": np.asarray(self.residual_override_indices, dtype=np.int32),
        }
