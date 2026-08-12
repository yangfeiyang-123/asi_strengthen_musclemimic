"""Fail-closed physical-control producer for primitive muscle rollouts.

The primitive synergy library needs controls that were actually applied to the
compiled MuJoCo model.  Retargeted ``qpos``/``qvel`` files are targets, not
control evidence.  This module closes that gap by:

* constructing the exact resolved ChinaJump TaskFactory runtime model (whose
  environment is ``MyoFullBody`` with fingers disabled), or strictly reusing
  its previously verified content-addressed MJB artifact;
* optimizing controls in the model's physical ``data.ctrl`` coordinates;
* stepping MuJoCo directly and recording each transition's exact ``data.ctrl``
  and post-transition muscle state from ``data.act``;
* replaying those controls from the exact initial integration state; and
* publishing a trial as successful only when every configured QC gate passes.

The controller uses an exact-transition forward-shooting linearization only as
a CEM proposal.  Proposal and initialization residuals are diagnostic and can
never make a rollout successful.  Only closed-loop tracking, contact semantics,
initialization shadow evidence, and replay gates decide recorded ``success``.
No normalized action or EMG field is accepted by this producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.motion_identity import normalize_relative_motion_path, stable_motion_uid
from musclemimic.distill.physical import resolve_muscle_channel_contract, validate_unit_muscle_ctrlrange
from musclemimic.distill.provenance import checkpoint_content_fingerprint
from musclemimic.synergy.canonical_control import (
    _publish_content_addressed_directory,
    load_canonical_control_artifact,
)
from musclemimic.synergy.primitive_catalog import (
    PrimitivePhaseSchema,
    canonical_json_sha256,
    load_primitive_phase_schema,
)
from musclemimic.synergy.primitive_ingest import file_sha256, save_compiled_model_artifact
from musclemimic.synergy.primitive_recording import write_primitive_trial_npz
from musclemimic.synergy.schema import ctrlrange_schema_hash

OPTIMIZER_MANIFEST_SCHEMA_VERSION = "primitive_physical_optimizer_manifest_v1"
ROLLOUT_MANIFEST_SCHEMA_VERSION = "primitive_physical_rollout_manifest_v7"
PHASE_PLAN_SCHEMA_VERSION = "primitive_transition_phase_plan_v1"
QC_ARRAY_SCHEMA_VERSION = "primitive_physical_rollout_qc_arrays_v7"
OPTIMIZER_ALGORITHM = "contact_forward_transition_shooting_bounded_cem_v1"
_INFERRED_INITIAL_STATE_CONTRACT = "inferred_hidden_muscle_state_no_step_shadow_gated_v1"
_ZERO_INITIAL_STATE_CONTRACT = "explicit_zero_activation_control_v1"
_CANONICAL_TONIC_INITIAL_STATE_CONTRACT = "canonical_tonic_activation_control_v1"

_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_SHA256_FIELDS = {"implementation_sha256", "model_hash", "model_artifact_sha256"}
_FLOOR_GEOM_NAME = "floor"
_FOOT_GEOM_NAMES = {
    "left": (
        "l_talus",
        "l_foot",
        "l_foot_col1",
        "l_foot_col3",
        "l_foot_col4",
        "l_bofoot",
        "l_bofoot_col1",
        "l_bofoot_col2",
    ),
    "right": (
        "r_talus",
        "r_foot",
        "r_foot_col1",
        "r_foot_col3",
        "r_foot_col4",
        "r_bofoot",
        "r_bofoot_col1",
        "r_bofoot_col2",
    ),
}
_TASK_PHASE_ORDER = {
    "P01": (0,),
    "P05": (0, 1, 2, 3),
    "P06": (0, 1, 2, 3),
    "P07": (0, 1, 2, 3),
    "P08": (0, 1, 2, 3),
    "P11": (0, 1, 2, 3, 4),
    "P12": (0, 1, 2),
}
_DYNAMIC_COM_TASK_FAMILIES = frozenset({"P05", "P06", "P07", "P11", "P12"})
_ROOT_VERTICAL_SIGNAL = "root_z/root_freejoint_vz"
_COM_VERTICAL_SIGNAL = "root_subtree_com_z/delta_over_transition_duration"
_TOY_VERTICAL_SIGNAL = "unscored_explicit_toy_fixture"
_P08_AUXILIARY_VERTICAL_SIGNAL = "auxiliary_root_subtree_com_z_not_scored_for_p08"
_AXIAL_ROTATION_JOINT_NAMES = (
    "axial_rotation",
    "Abs_r3",
    "L4_L5_AR",
    "L3_L4_AR",
    "L2_L3_AR",
    "L1_L2_AR",
)
_AXIAL_ROTATION_CONTRACT_SCHEMA_VERSION = "p08_named_axial_rotation_signal_contract_v1"
_AXIAL_ROTATION_POSITION_SIGNAL = "sum_named_hinge_qpos_post_transition"
_AXIAL_ROTATION_VELOCITY_SIGNAL = "delta_position_over_transition_duration"


@dataclass(frozen=True)
class PhysicalOptimizerConfig:
    """Configuration of physical transition shooting and CEM refinement."""

    horizon: int = 2
    population: int = 12
    elite_count: int = 3
    iterations: int = 2
    initial_std: float = 0.12
    min_std: float = 0.015
    elite_momentum: float = 0.25
    position_weight: float = 1.0
    velocity_weight: float = 0.02
    p08_axial_position_weight: float = 0.0
    p08_axial_velocity_weight: float = 0.0
    p08_position_abs_weight: float = 0.0
    p08_root_orientation_weight: float = 0.0
    effort_weight: float = 1.0e-4
    rate_weight: float = 5.0e-4
    terminal_weight: float = 2.0
    initial_activation_margin: float = 0.01
    initial_forward_regularization: float = 1.0e-3
    initial_forward_solver_tolerance: float = 1.0e-10
    initial_forward_solver_max_iterations: int = 1000
    shooting_finite_difference_step: float = 0.05
    shooting_solver_tolerance: float = 1.0e-10
    shooting_solver_max_iterations: int = 1000

    def validated(self) -> PhysicalOptimizerConfig:
        integer_fields = {
            "horizon": self.horizon,
            "population": self.population,
            "elite_count": self.elite_count,
            "iterations": self.iterations,
            "initial_forward_solver_max_iterations": self.initial_forward_solver_max_iterations,
            "shooting_solver_max_iterations": self.shooting_solver_max_iterations,
        }
        if any(type(value) is not int or value <= 0 for value in integer_fields.values()):
            raise ValueError("physical optimizer integer settings must be positive integers")
        if self.elite_count > self.population:
            raise ValueError("elite_count may not exceed population")
        finite_nonnegative = {
            "initial_std": self.initial_std,
            "min_std": self.min_std,
            "position_weight": self.position_weight,
            "velocity_weight": self.velocity_weight,
            "p08_axial_position_weight": self.p08_axial_position_weight,
            "p08_axial_velocity_weight": self.p08_axial_velocity_weight,
            "p08_position_abs_weight": self.p08_position_abs_weight,
            "p08_root_orientation_weight": self.p08_root_orientation_weight,
            "effort_weight": self.effort_weight,
            "rate_weight": self.rate_weight,
            "terminal_weight": self.terminal_weight,
            "initial_forward_regularization": self.initial_forward_regularization,
        }
        if any(not np.isfinite(value) or value < 0.0 for value in finite_nonnegative.values()):
            raise ValueError("physical optimizer weights and scales must be finite and non-negative")
        if self.initial_std <= 0.0 or self.min_std <= 0.0:
            raise ValueError("CEM standard deviations must be positive")
        if not np.isfinite(self.elite_momentum) or not 0.0 <= self.elite_momentum < 1.0:
            raise ValueError("elite_momentum must lie in [0,1)")
        if self.position_weight == 0.0 and self.velocity_weight == 0.0:
            raise ValueError("at least one tracking weight must be positive")
        if (
            not np.isfinite(self.initial_activation_margin)
            or self.initial_activation_margin <= 0.0
            or self.initial_activation_margin >= 0.5
        ):
            raise ValueError("initial_activation_margin must lie in (0,0.5)")
        if self.initial_forward_regularization <= 0.0:
            raise ValueError("initial_forward_regularization must be positive")
        if (
            not np.isfinite(self.initial_forward_solver_tolerance)
            or self.initial_forward_solver_tolerance <= 0.0
            or self.initial_forward_solver_tolerance >= 1.0
        ):
            raise ValueError("initial_forward_solver_tolerance must lie in (0,1)")
        if (
            not np.isfinite(self.shooting_finite_difference_step)
            or self.shooting_finite_difference_step <= 0.0
            or self.shooting_finite_difference_step >= 1.0
        ):
            raise ValueError("shooting_finite_difference_step must lie in (0,1)")
        if (
            not np.isfinite(self.shooting_solver_tolerance)
            or self.shooting_solver_tolerance <= 0.0
            or self.shooting_solver_tolerance >= 1.0
        ):
            raise ValueError("shooting_solver_tolerance must lie in (0,1)")
        return self


@dataclass(frozen=True)
class RolloutQCConfig:
    """Explicit pass/fail limits for one closed-loop primitive rollout."""

    max_position_rmse: float
    max_velocity_rmse: float
    max_position_abs: float
    max_velocity_abs: float
    max_saturation_fraction: float = 0.98
    replay_position_atol: float = 1.0e-10
    replay_velocity_atol: float = 1.0e-10
    replay_activation_atol: float = 1.0e-10
    replay_contact_force_atol: float = 1.0e-10
    min_contact_normal_force: float = 1.0e-6
    max_bilateral_contact_lag_frames: int = 5
    min_low_flight_frames: int = 2
    min_precontact_air_frames: int = 2
    min_landing_stabilization_frames: int = 10
    min_ready_hold_frames: int = 10
    min_phase_transitions: int = 2
    min_ready_frames: int = 5
    max_stable_root_vertical_speed: float = 0.20
    max_ready_com_vertical_speed: float = 0.15
    max_post_impact_com_vertical_speed: float = 0.20
    max_ready_hold_com_vertical_speed: float = 0.15
    min_com_vertical_excursion: float = 0.03
    max_axial_neutral_speed: float = 0.60
    min_axial_rotation_excursion: float = 0.12
    min_axial_signed_monotonic_fraction: float = 0.90
    max_axial_recenter_error: float = 0.01
    max_axial_root_yaw_excursion: float = 0.35
    max_axial_root_xy_displacement: float = 0.25
    site_contact_baseline_quantile: float = 0.01
    site_contact_enter_height: float = 0.035
    site_contact_exit_height: float = 0.045

    def validated(self) -> RolloutQCConfig:
        positive = {
            "max_position_rmse": self.max_position_rmse,
            "max_velocity_rmse": self.max_velocity_rmse,
            "max_position_abs": self.max_position_abs,
            "max_velocity_abs": self.max_velocity_abs,
            "replay_position_atol": self.replay_position_atol,
            "replay_velocity_atol": self.replay_velocity_atol,
            "replay_activation_atol": self.replay_activation_atol,
            "replay_contact_force_atol": self.replay_contact_force_atol,
            "min_contact_normal_force": self.min_contact_normal_force,
            "max_stable_root_vertical_speed": self.max_stable_root_vertical_speed,
            "max_ready_com_vertical_speed": self.max_ready_com_vertical_speed,
            "max_post_impact_com_vertical_speed": self.max_post_impact_com_vertical_speed,
            "max_ready_hold_com_vertical_speed": self.max_ready_hold_com_vertical_speed,
            "min_com_vertical_excursion": self.min_com_vertical_excursion,
            "max_axial_neutral_speed": self.max_axial_neutral_speed,
            "min_axial_rotation_excursion": self.min_axial_rotation_excursion,
            "max_axial_recenter_error": self.max_axial_recenter_error,
            "max_axial_root_yaw_excursion": self.max_axial_root_yaw_excursion,
            "max_axial_root_xy_displacement": self.max_axial_root_xy_displacement,
            "site_contact_enter_height": self.site_contact_enter_height,
            "site_contact_exit_height": self.site_contact_exit_height,
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("tracking and replay QC limits must be finite and positive")
        if (
            not np.isfinite(self.max_saturation_fraction)
            or self.max_saturation_fraction < 0.0
            or self.max_saturation_fraction > 1.0
        ):
            raise ValueError("max_saturation_fraction must lie in [0,1]")
        if (
            not np.isfinite(self.min_axial_signed_monotonic_fraction)
            or not 0.0 <= self.min_axial_signed_monotonic_fraction <= 1.0
        ):
            raise ValueError("min_axial_signed_monotonic_fraction must lie in [0,1]")
        if (
            not np.isfinite(self.site_contact_baseline_quantile)
            or not 0.0 < self.site_contact_baseline_quantile <= 0.25
        ):
            raise ValueError("site_contact_baseline_quantile must lie in (0,0.25]")
        if self.site_contact_exit_height <= self.site_contact_enter_height:
            raise ValueError("site contact exit height must exceed enter height for hysteresis")
        integer_thresholds = {
            "max_bilateral_contact_lag_frames": self.max_bilateral_contact_lag_frames,
            "min_low_flight_frames": self.min_low_flight_frames,
            "min_precontact_air_frames": self.min_precontact_air_frames,
            "min_landing_stabilization_frames": self.min_landing_stabilization_frames,
            "min_ready_hold_frames": self.min_ready_hold_frames,
            "min_phase_transitions": self.min_phase_transitions,
            "min_ready_frames": self.min_ready_frames,
        }
        if any(type(value) is not int or value < 0 for value in integer_thresholds.values()):
            raise ValueError("contact/phase QC frame thresholds must be non-negative integers")
        if self.max_bilateral_contact_lag_frames <= 0:
            raise ValueError("max_bilateral_contact_lag_frames must be positive")
        if any(value <= 0 for name, value in integer_thresholds.items() if name != "max_bilateral_contact_lag_frames"):
            raise ValueError("contact/phase minimum frame thresholds must be positive")
        return self

    def semantic_thresholds(self) -> dict[str, Any]:
        """Return the contact/phase subset bound into preflight and rollout QC."""

        return {
            "min_contact_normal_force": float(self.min_contact_normal_force),
            "max_bilateral_contact_lag_frames": int(self.max_bilateral_contact_lag_frames),
            "min_low_flight_frames": int(self.min_low_flight_frames),
            "min_precontact_air_frames": int(self.min_precontact_air_frames),
            "min_landing_stabilization_frames": int(self.min_landing_stabilization_frames),
            "min_ready_hold_frames": int(self.min_ready_hold_frames),
            "min_phase_transitions": int(self.min_phase_transitions),
            "min_ready_frames": int(self.min_ready_frames),
            "max_stable_root_vertical_speed": float(self.max_stable_root_vertical_speed),
            "max_ready_com_vertical_speed": float(self.max_ready_com_vertical_speed),
            "max_post_impact_com_vertical_speed": float(self.max_post_impact_com_vertical_speed),
            "max_ready_hold_com_vertical_speed": float(self.max_ready_hold_com_vertical_speed),
            "min_com_vertical_excursion": float(self.min_com_vertical_excursion),
            "max_axial_neutral_speed": float(self.max_axial_neutral_speed),
            "min_axial_rotation_excursion": float(self.min_axial_rotation_excursion),
            "min_axial_signed_monotonic_fraction": float(self.min_axial_signed_monotonic_fraction),
            "max_axial_recenter_error": float(self.max_axial_recenter_error),
            "max_axial_root_yaw_excursion": float(self.max_axial_root_yaw_excursion),
            "max_axial_root_xy_displacement": float(self.max_axial_root_xy_displacement),
            "site_contact_baseline_quantile": float(self.site_contact_baseline_quantile),
            "site_contact_enter_height": float(self.site_contact_enter_height),
            "site_contact_exit_height": float(self.site_contact_exit_height),
        }


@dataclass(frozen=True)
class MotionTarget:
    """A non-ChinaJump state trajectory with transition-aligned phase labels."""

    qpos: np.ndarray
    qvel: np.ndarray
    phase_id: np.ndarray
    transition_substeps: np.ndarray
    source_path: Path | None
    source_motion_path: str
    source_sha256: str
    source_frequency_hz: float
    source_start_frame: int
    source_end_frame_exclusive: int
    source_total_frames: int
    phase_schema_fingerprint: str
    task_id: str

    @property
    def transition_count(self) -> int:
        return int(self.phase_id.shape[0])

    def validated(self, model: mujoco.MjModel) -> MotionTarget:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        qvel = np.asarray(self.qvel, dtype=np.float64)
        phase = np.asarray(self.phase_id)
        substeps = np.asarray(self.transition_substeps)
        if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != int(model.nq):
            raise ValueError(f"target qpos must have shape [N,{int(model.nq)}] with N>=2")
        if qvel.shape != (qpos.shape[0], int(model.nv)):
            raise ValueError(f"target qvel must have shape [{qpos.shape[0]},{int(model.nv)}]")
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            raise ValueError("target qpos/qvel contains non-finite values")
        transition_count = qpos.shape[0] - 1
        if (
            phase.shape != (transition_count,)
            or np.issubdtype(phase.dtype, np.bool_)
            or not np.issubdtype(phase.dtype, np.integer)
            or np.any(phase < 0)
            or np.any(phase > np.iinfo(np.int32).max)
        ):
            raise ValueError("phase_id must be a non-negative integer label for every transition")
        if (
            substeps.shape != (transition_count,)
            or np.issubdtype(substeps.dtype, np.bool_)
            or not np.issubdtype(substeps.dtype, np.integer)
            or np.any(substeps <= 0)
        ):
            raise ValueError("transition_substeps must be a positive integer for every transition")
        if not str(self.task_id).strip():
            raise ValueError("target task_id must be non-empty")
        normalized = normalize_relative_motion_path(self.source_motion_path)
        _assert_not_target_skill_source(normalized, self.source_path, target_skill_id="ChinaJump")
        if not _is_sha256(self.source_sha256) or not _is_sha256(self.phase_schema_fingerprint):
            raise ValueError("target source and phase-schema fingerprints must be SHA-256 values")
        if not np.isfinite(self.source_frequency_hz) or self.source_frequency_hz <= 0.0:
            raise ValueError("target source_frequency_hz must be finite and positive")
        if (
            type(self.source_start_frame) is not int
            or type(self.source_end_frame_exclusive) is not int
            or type(self.source_total_frames) is not int
            or self.source_start_frame < 0
            or self.source_end_frame_exclusive <= self.source_start_frame
            or self.source_end_frame_exclusive > self.source_total_frames
            or self.source_end_frame_exclusive - self.source_start_frame != qpos.shape[0]
        ):
            raise ValueError("target source frame interval is empty, out of bounds, or inconsistent")
        return MotionTarget(
            qpos=qpos,
            qvel=qvel,
            phase_id=phase.astype(np.int32),
            transition_substeps=substeps.astype(np.int32),
            source_path=self.source_path,
            source_motion_path=normalized,
            source_sha256=str(self.source_sha256),
            source_frequency_hz=float(self.source_frequency_hz),
            source_start_frame=self.source_start_frame,
            source_end_frame_exclusive=self.source_end_frame_exclusive,
            source_total_frames=self.source_total_frames,
            phase_schema_fingerprint=str(self.phase_schema_fingerprint),
            task_id=str(self.task_id),
        )


@dataclass(frozen=True)
class FootFloorContactContract:
    """Stable geom-name/ID binding used for exact foot-floor evidence."""

    available: bool
    floor_geom_id: int | None
    floor_geom_name: str | None
    left_geom_ids: tuple[int, ...]
    left_geom_names: tuple[str, ...]
    right_geom_ids: tuple[int, ...]
    right_geom_names: tuple[str, ...]
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "exact_foot_floor_contact_contract_v1",
            "available": bool(self.available),
            "floor_geom_id": self.floor_geom_id,
            "floor_geom_name": self.floor_geom_name,
            "left_foot_geom_ids": list(self.left_geom_ids),
            "left_foot_geom_names": list(self.left_geom_names),
            "right_foot_geom_ids": list(self.right_geom_ids),
            "right_foot_geom_names": list(self.right_geom_names),
            "normal_force_source": "mujoco.mj_contactForce(...)[0]",
            "contact_sample_time": "post_transition",
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class AxialRotationSignalContract:
    """Stable, named MuJoCo binding for the P08 axial-rotation signal."""

    available: bool
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    qpos_addresses: tuple[int, ...]
    dof_addresses: tuple[int, ...]
    root_joint_id: int | None
    root_qpos_address: int | None
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _AXIAL_ROTATION_CONTRACT_SCHEMA_VERSION,
            "available": bool(self.available),
            "joint_names": list(self.joint_names),
            "joint_ids": list(self.joint_ids),
            "qpos_addresses": list(self.qpos_addresses),
            "dof_addresses": list(self.dof_addresses),
            "position_signal": _AXIAL_ROTATION_POSITION_SIGNAL,
            "velocity_signal": _AXIAL_ROTATION_VELOCITY_SIGNAL,
            "sample_time": "post_transition",
            "initial_sample": "pre_first_transition_qpos",
            "root_yaw_signal": "unwrapped_yaw_from_root_freejoint_qpos_quaternion",
            "root_xy_signal": "root_freejoint_qpos_xy",
            "root_joint_id": self.root_joint_id,
            "root_qpos_address": self.root_qpos_address,
            "proxy_fallback_allowed": False,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class AxialRotationEvidence:
    """Transition-aligned P08 kinematics reconstructed from one qpos sequence."""

    position: np.ndarray
    velocity: np.ndarray
    root_yaw: np.ndarray
    root_xy: np.ndarray
    initial_position: float
    initial_root_yaw: float
    initial_root_xy: np.ndarray
    contract: AxialRotationSignalContract


@dataclass(frozen=True)
class TargetContactAudit:
    """Exact state-by-state target contact evidence and semantic verdict."""

    left_foot_floor_contact: np.ndarray
    right_foot_floor_contact: np.ndarray
    left_foot_floor_normal_force: np.ndarray
    right_foot_floor_normal_force: np.ndarray
    initial_left_foot_floor_contact: bool
    initial_right_foot_floor_contact: bool
    initial_left_foot_floor_normal_force: float
    initial_right_foot_floor_normal_force: float
    site_proxy_left_foot_contact: np.ndarray
    site_proxy_right_foot_contact: np.ndarray
    initial_site_proxy_left_foot_contact: bool
    initial_site_proxy_right_foot_contact: bool
    site_proxy_left_clearance: np.ndarray
    site_proxy_right_clearance: np.ndarray
    root_vertical_position: np.ndarray
    root_vertical_velocity: np.ndarray
    com_vertical_position: np.ndarray
    com_vertical_velocity: np.ndarray
    axial_rotation: AxialRotationEvidence
    semantics: dict[str, Any]


@dataclass(frozen=True)
class RolloutArrays:
    applied_ctrl: np.ndarray
    muscle_activation: np.ndarray
    muscle_force: np.ndarray
    muscle_tendon_length: np.ndarray
    muscle_tendon_velocity: np.ndarray
    actual_qpos: np.ndarray
    actual_qvel: np.ndarray
    actual_root_vertical_position: np.ndarray
    actual_root_vertical_velocity: np.ndarray
    actual_com_vertical_position: np.ndarray
    actual_com_vertical_velocity: np.ndarray
    actual_axial_rotation: AxialRotationEvidence
    target_qpos: np.ndarray
    target_qvel: np.ndarray
    position_error: np.ndarray
    velocity_error: np.ndarray
    phase_id: np.ndarray
    transition_substeps: np.ndarray
    initialization: RolloutInitialization
    initial_integration_state: np.ndarray
    proposal_tracking_residual_norm: np.ndarray
    left_foot_floor_contact: np.ndarray
    right_foot_floor_contact: np.ndarray
    left_foot_floor_normal_force: np.ndarray
    right_foot_floor_normal_force: np.ndarray

    @property
    def transition_count(self) -> int:
        return int(self.applied_ctrl.shape[0])


@dataclass(frozen=True)
class ReplayArrays:
    qpos: np.ndarray
    qvel: np.ndarray
    muscle_activation: np.ndarray
    left_foot_floor_contact: np.ndarray
    right_foot_floor_contact: np.ndarray
    left_foot_floor_normal_force: np.ndarray
    right_foot_floor_normal_force: np.ndarray
    axial_rotation: AxialRotationEvidence


@dataclass(frozen=True)
class ProducerResult:
    output_dir: Path
    controller_dir: Path
    controller_fingerprint: str
    rollout_fingerprint: str
    trial_path: Path | None
    success: bool
    qc: dict[str, Any]


@dataclass(frozen=True)
class RolloutInitialization:
    """Exact muscle state used before the first recorded transition."""

    contract: str
    target_acceleration_method: str
    initial_activation: np.ndarray
    initial_ctrl: np.ndarray
    solver_kind: str
    solver_status: int
    solver_iterations: int
    solver_optimality: float
    linearized_acceleration_residual_norm: float
    forward_acceleration_error: np.ndarray
    shadow_transition_duration: float
    initial_left_foot_floor_contact: bool
    initial_right_foot_floor_contact: bool
    initial_left_foot_floor_normal_force: float
    initial_right_foot_floor_normal_force: float
    shadow_qpos: np.ndarray
    shadow_qvel: np.ndarray
    shadow_position_error: np.ndarray
    shadow_velocity_error: np.ndarray
    shadow_left_foot_floor_contact: np.ndarray
    shadow_right_foot_floor_contact: np.ndarray
    shadow_left_foot_floor_normal_force: np.ndarray
    shadow_right_foot_floor_normal_force: np.ndarray
    shadow_final_integration_state: np.ndarray


@dataclass(frozen=True)
class TaskFactoryRuntimeModel:
    """Model built by the exact resolved ChinaJump Hydra TaskFactory config."""

    model: mujoco.MjModel
    binding: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _ComposedChinaJumpRuntimeConfig:
    """Resolved Hydra identity; composing this object never constructs an env."""

    name: str
    hydra_overrides: tuple[str, ...]
    config: Any
    resolved_config: dict[str, Any]
    declared_production_num_envs: int


@dataclass(frozen=True)
class PolicyControlImport:
    """Validated full-354 teacher controls and their two provenance layers."""

    planner: ScriptedPhysicalControlPlanner
    controller_binding: dict[str, Any]
    rollout_binding: dict[str, Any]


class PhysicalControlPlanner(Protocol):
    """Transition-level planner that returns model-unit physical control."""

    def reset(self, seed: int) -> None: ...

    def initialize(
        self,
        data: mujoco.MjData,
        target: MotionTarget,
        *,
        contact_contract: FootFloorContactContract,
        min_contact_normal_force: float,
    ) -> RolloutInitialization: ...

    def plan(
        self,
        data: mujoco.MjData,
        target: MotionTarget,
        transition_index: int,
        previous_ctrl: np.ndarray,
    ) -> tuple[np.ndarray, float]: ...


class ComputedMuscleCEMPlanner:
    """Contact-forward transition shooting refined by bounded receding-horizon CEM."""

    def __init__(self, model: mujoco.MjModel, config: PhysicalOptimizerConfig):
        self.model = model
        self.config = config.validated()
        self.names = complete_actuator_names(model)
        self.channel_contract = resolve_muscle_channel_contract(model, self.names)
        self.ctrlrange = validate_unit_muscle_ctrlrange(self.names, model.actuator_ctrlrange)
        self.actadr = np.asarray(self.channel_contract.actuator_actadr, dtype=np.int32)
        self._candidate = mujoco.MjData(model)
        self._rng = np.random.default_rng(0)
        self._warm_plan: np.ndarray | None = None
        self._p08_axial_dof_cache: np.ndarray | None = None
        self._p08_root_orientation_dof_cache: np.ndarray | None = None

    def reset(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("optimizer seed must be a non-negative integer")
        self._rng = np.random.default_rng(seed)
        self._warm_plan = None

    def initialize(
        self,
        data: mujoco.MjData,
        target: MotionTarget,
        *,
        contact_contract: FootFloorContactContract,
        min_contact_normal_force: float,
    ) -> RolloutInitialization:
        """Infer and shadow-audit a contact-aware initial activation seed.

        Retargeted kinematics contain no muscle state.  A zero-activation
        weight-bearing pose creates an artificial impulse, while an inverse
        muscle solve does not guarantee forward mechanical balance under
        contact.  At the exact first state, this method samples the constrained
        MuJoCo forward-acceleration response of every activation coordinate,
        solves a bounded regularized linearization for the target finite-
        difference acceleration, and uses the result as both ``act`` and
        ``ctrl``.  It is explicitly a seed, not a claimed equilibrium.  The
        exact forward state and a constant-control shadow transition are
        recorded and must pass independent QC before a trial can succeed.
        """

        target = target.validated(self.model)
        _require_first_target_state(self.model, data, target)
        duration = float(target.transition_substeps[0]) * float(self.model.opt.timestep)
        desired_qacc = (target.qvel[1] - target.qvel[0]) / duration
        data.act[:] = 0.0
        data.ctrl[:] = self.ctrlrange[:, 0]
        mujoco.mj_forward(self.model, data)
        base_qacc = np.asarray(data.qacc, dtype=np.float64).copy()
        response = np.empty((int(self.model.nv), int(self.model.nu)), dtype=np.float64)
        for actuator_index, activation_address in enumerate(self.actadr):
            data.act[:] = 0.0
            data.act[int(activation_address)] = 1.0
            mujoco.mj_forward(self.model, data)
            response[:, actuator_index] = np.asarray(data.qacc, dtype=np.float64) - base_qacc

        regularization = float(self.config.initial_forward_regularization)
        matrix = np.concatenate(
            (
                response,
                np.sqrt(regularization) * np.eye(int(self.model.nu), dtype=np.float64),
            ),
            axis=0,
        )
        rhs = np.concatenate((desired_qacc - base_qacc, np.zeros(int(self.model.nu))), axis=0)
        margin = float(self.config.initial_activation_margin)
        solved = lsq_linear(
            matrix,
            rhs,
            bounds=(margin, 1.0 - margin),
            method="bvls",
            tol=float(self.config.initial_forward_solver_tolerance),
            max_iter=int(self.config.initial_forward_solver_max_iterations),
            verbose=0,
        )
        if int(solved.status) <= 0 or not np.all(np.isfinite(solved.x)):
            raise ValueError("bounded forward-dynamics initialization solver did not converge")
        initial_activation = np.asarray(solved.x, dtype=np.float64)
        data.act[:] = 0.0
        data.act[self.actadr] = initial_activation
        data.ctrl[:] = initial_activation
        mujoco.mj_forward(self.model, data)
        if not _finite_dynamic_state(data):
            raise ValueError("bounded forward-dynamics initialization produced a non-finite state")
        linearized_residual = float(np.linalg.norm(response @ initial_activation - (desired_qacc - base_qacc)))
        return _build_rollout_initialization(
            self.model,
            data,
            target,
            actadr=self.actadr,
            contract=_INFERRED_INITIAL_STATE_CONTRACT,
            target_acceleration_method="(target_qvel[1]-target_qvel[0])/first_transition_duration",
            solver_kind="scipy_lsq_linear_bvls_on_mujoco_contact_forward_acceleration_linearization",
            solver_status=int(solved.status),
            solver_iterations=int(solved.nit),
            solver_optimality=float(solved.optimality),
            linearized_acceleration_residual_norm=linearized_residual,
            contact_contract=contact_contract,
            min_contact_normal_force=min_contact_normal_force,
        )

    def plan(
        self,
        data: mujoco.MjData,
        target: MotionTarget,
        transition_index: int,
        previous_ctrl: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        t = int(transition_index)
        if t < 0 or t >= target.transition_count:
            raise IndexError("transition index is outside target")
        previous = np.asarray(previous_ctrl, dtype=np.float64)
        if previous.shape != (int(self.model.nu),):
            raise ValueError("previous physical ctrl has wrong model width")
        state = _capture_integration_state(self.model, data)
        proposal, residual = self._transition_shooting_proposal(
            state,
            target_qpos=target.qpos[t + 1],
            target_qvel=target.qvel[t + 1],
            task_id=target.task_id,
            previous_ctrl=previous,
            substeps=int(target.transition_substeps[t]),
        )
        horizon = min(self.config.horizon, target.transition_count - t)
        if self._warm_plan is None:
            mean = np.repeat(proposal[None, :], horizon, axis=0)
        else:
            warm = self._warm_plan
            mean = np.empty((horizon, int(self.model.nu)), dtype=np.float64)
            usable = min(horizon, warm.shape[0])
            mean[:usable] = warm[:usable]
            if usable < horizon:
                mean[usable:] = proposal
            mean[0] = proposal
        ctrl_span = self.ctrlrange[:, 1] - self.ctrlrange[:, 0]
        std = np.repeat((self.config.initial_std * ctrl_span)[None, :], horizon, axis=0)
        best_plan = mean.copy()
        best_cost = self._plan_cost(state, best_plan, target, t, previous)
        for _ in range(self.config.iterations):
            candidates = self._rng.normal(
                loc=mean[None, :, :],
                scale=std[None, :, :],
                size=(self.config.population, horizon, int(self.model.nu)),
            )
            candidates = np.clip(
                candidates,
                self.ctrlrange[:, 0][None, None, :],
                self.ctrlrange[:, 1][None, None, :],
            )
            candidates[0] = mean
            costs = np.asarray(
                [self._plan_cost(state, candidate, target, t, previous) for candidate in candidates],
                dtype=np.float64,
            )
            elite_indices = np.argsort(costs, kind="stable")[: self.config.elite_count]
            elites = candidates[elite_indices]
            elite_mean = np.mean(elites, axis=0)
            elite_std = np.std(elites, axis=0)
            momentum = self.config.elite_momentum
            mean = momentum * mean + (1.0 - momentum) * elite_mean
            std = np.maximum(
                momentum * std + (1.0 - momentum) * elite_std,
                self.config.min_std * ctrl_span[None, :],
            )
            iteration_best = int(elite_indices[0])
            if float(costs[iteration_best]) < best_cost:
                best_cost = float(costs[iteration_best])
                best_plan = candidates[iteration_best].copy()
        # Shift the chosen horizon; the next call replaces its first element
        # with a newly computed exact-transition shooting proposal but retains
        # future structure.
        self._warm_plan = np.concatenate((best_plan[1:], best_plan[-1:]), axis=0)
        return best_plan[0].copy(), float(residual)

    def _transition_shooting_proposal(
        self,
        state: np.ndarray,
        *,
        target_qpos: np.ndarray,
        target_qvel: np.ndarray,
        task_id: str,
        previous_ctrl: np.ndarray,
        substeps: int,
    ) -> tuple[np.ndarray, float]:
        """Solve a bounded one-transition shooting linearization in ctrl space."""

        previous = np.asarray(previous_ctrl, dtype=np.float64)
        base_position_error, base_velocity_error = self._simulate_transition_error(
            state,
            ctrl=previous,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            substeps=substeps,
        )
        position_jacobian = np.empty((int(self.model.nv), int(self.model.nu)), dtype=np.float64)
        velocity_jacobian = np.empty_like(position_jacobian)
        span = self.ctrlrange[:, 1] - self.ctrlrange[:, 0]
        nominal_step = float(self.config.shooting_finite_difference_step) * span
        for actuator_index in range(int(self.model.nu)):
            positive_room = self.ctrlrange[actuator_index, 1] - previous[actuator_index]
            if positive_room >= nominal_step[actuator_index]:
                difference = nominal_step[actuator_index]
            else:
                difference = -nominal_step[actuator_index]
            perturbed = previous.copy()
            perturbed[actuator_index] += difference
            position_error, velocity_error = self._simulate_transition_error(
                state,
                ctrl=perturbed,
                target_qpos=target_qpos,
                target_qvel=target_qvel,
                substeps=substeps,
            )
            position_jacobian[:, actuator_index] = (position_error - base_position_error) / difference
            velocity_jacobian[:, actuator_index] = (velocity_error - base_velocity_error) / difference

        position_scale = np.sqrt(float(self.config.position_weight) / float(self.model.nv))
        velocity_scale = np.sqrt(float(self.config.velocity_weight) / float(self.model.nv))
        matrix_parts = [position_scale * position_jacobian, velocity_scale * velocity_jacobian]
        rhs_parts = [-position_scale * base_position_error, -velocity_scale * base_velocity_error]
        axial_dofs = self._p08_axial_tracking_dofs(task_id)
        if axial_dofs is not None and self.config.p08_axial_position_weight > 0.0:
            axial_position_scale = np.sqrt(float(self.config.p08_axial_position_weight))
            matrix_parts.append(axial_position_scale * np.sum(position_jacobian[axial_dofs], axis=0, keepdims=True))
            rhs_parts.append(
                np.asarray(
                    [-axial_position_scale * float(np.sum(base_position_error[axial_dofs]))],
                    dtype=np.float64,
                )
            )
        if axial_dofs is not None and self.config.p08_axial_velocity_weight > 0.0:
            axial_velocity_scale = np.sqrt(float(self.config.p08_axial_velocity_weight))
            matrix_parts.append(axial_velocity_scale * np.sum(velocity_jacobian[axial_dofs], axis=0, keepdims=True))
            rhs_parts.append(
                np.asarray(
                    [-axial_velocity_scale * float(np.sum(base_velocity_error[axial_dofs]))],
                    dtype=np.float64,
                )
            )
        p08_task = str(task_id).split("_", 1)[0] == "P08"
        root_orientation_dofs = self._p08_root_orientation_tracking_dofs(task_id)
        identity = np.eye(int(self.model.nu), dtype=np.float64)
        if self.config.rate_weight > 0.0:
            matrix_parts.append(np.sqrt(float(self.config.rate_weight) / float(self.model.nu)) * identity)
            rhs_parts.append(np.zeros(int(self.model.nu), dtype=np.float64))
        if self.config.effort_weight > 0.0:
            effort_scale = np.sqrt(float(self.config.effort_weight) / float(self.model.nu))
            matrix_parts.append(effort_scale * identity)
            rhs_parts.append(-effort_scale * previous)
        solved = lsq_linear(
            np.concatenate(matrix_parts, axis=0),
            np.concatenate(rhs_parts),
            bounds=(self.ctrlrange[:, 0] - previous, self.ctrlrange[:, 1] - previous),
            method="bvls",
            tol=float(self.config.shooting_solver_tolerance),
            max_iter=int(self.config.shooting_solver_max_iterations),
            verbose=0,
        )
        if int(solved.status) <= 0 or not np.all(np.isfinite(solved.x)):
            raise ValueError("bounded transition-shooting proposal solver did not converge")
        proposal = np.clip(
            previous + np.asarray(solved.x, dtype=np.float64), self.ctrlrange[:, 0], self.ctrlrange[:, 1]
        )
        proposal_position_error, proposal_velocity_error = self._simulate_transition_error(
            state,
            ctrl=proposal,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            substeps=substeps,
        )
        residual_squared = float(self.config.position_weight) * float(
            np.mean(np.square(proposal_position_error))
        ) + float(self.config.velocity_weight) * float(np.mean(np.square(proposal_velocity_error)))
        if axial_dofs is not None:
            residual_squared += float(self.config.p08_axial_position_weight) * float(
                np.square(np.sum(proposal_position_error[axial_dofs]))
            )
            residual_squared += float(self.config.p08_axial_velocity_weight) * float(
                np.square(np.sum(proposal_velocity_error[axial_dofs]))
            )
        if root_orientation_dofs is not None:
            residual_squared += float(self.config.p08_root_orientation_weight) * float(
                np.mean(np.square(proposal_position_error[root_orientation_dofs]))
            )
        if p08_task:
            residual_squared += float(self.config.p08_position_abs_weight) * float(
                np.max(np.square(proposal_position_error))
            )
        residual = float(np.sqrt(residual_squared))
        return proposal, residual

    def _p08_axial_tracking_dofs(self, task_id: str) -> np.ndarray | None:
        """Return the six named P08 hinge DOFs when the optional objective is active."""

        if str(task_id).split("_", 1)[0] != "P08" or (
            self.config.p08_axial_position_weight == 0.0 and self.config.p08_axial_velocity_weight == 0.0
        ):
            return None
        if self._p08_axial_dof_cache is None:
            contract = resolve_axial_rotation_signal_contract(self.model)
            addresses = np.asarray(contract.dof_addresses, dtype=np.int32)
            addresses.setflags(write=False)
            self._p08_axial_dof_cache = addresses
        return self._p08_axial_dof_cache

    def _p08_root_orientation_tracking_dofs(self, task_id: str) -> np.ndarray | None:
        """Return the root free-joint rotation DOFs for optional P08 pose tracking."""

        if str(task_id).split("_", 1)[0] != "P08" or self.config.p08_root_orientation_weight == 0.0:
            return None
        if self._p08_root_orientation_dof_cache is None:
            contract = resolve_axial_rotation_signal_contract(self.model)
            if contract.root_joint_id is None:
                raise ValueError("P08 root-orientation objective requires the named root free joint")
            root_dof = int(self.model.jnt_dofadr[contract.root_joint_id])
            addresses = np.arange(root_dof + 3, root_dof + 6, dtype=np.int32)
            if np.any(addresses < 0) or np.any(addresses >= int(self.model.nv)):
                raise ValueError("P08 root-orientation DOF address lies outside the model")
            addresses.setflags(write=False)
            self._p08_root_orientation_dof_cache = addresses
        return self._p08_root_orientation_dof_cache

    def _simulate_transition_error(
        self,
        state: np.ndarray,
        *,
        ctrl: np.ndarray,
        target_qpos: np.ndarray,
        target_qvel: np.ndarray,
        substeps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        _restore_integration_state(self.model, self._candidate, state)
        self._candidate.ctrl[:] = ctrl
        for _ in range(int(substeps)):
            mujoco.mj_step(self.model, self._candidate)
        if not _finite_dynamic_state(self._candidate):
            raise ValueError("transition shooting encountered a non-finite simulated state")
        return (
            _position_difference(self.model, self._candidate.qpos, target_qpos),
            np.asarray(self._candidate.qvel - target_qvel, dtype=np.float64).copy(),
        )

    def _plan_cost(
        self,
        state: np.ndarray,
        plan: np.ndarray,
        target: MotionTarget,
        transition_index: int,
        previous_ctrl: np.ndarray,
    ) -> float:
        _restore_integration_state(self.model, self._candidate, state)
        total = 0.0
        prior = np.asarray(previous_ctrl, dtype=np.float64)
        for offset, ctrl in enumerate(np.asarray(plan, dtype=np.float64)):
            index = transition_index + offset
            self._candidate.ctrl[:] = ctrl
            for _ in range(int(target.transition_substeps[index])):
                mujoco.mj_step(self.model, self._candidate)
            if not _finite_dynamic_state(self._candidate):
                return float(np.finfo(np.float64).max / 4.0)
            position_error = _position_difference(
                self.model,
                self._candidate.qpos,
                target.qpos[index + 1],
            )
            velocity_error = self._candidate.qvel - target.qvel[index + 1]
            step_weight = self.config.terminal_weight if offset == len(plan) - 1 else 1.0
            tracking_cost = self.config.position_weight * float(
                np.mean(np.square(position_error))
            ) + self.config.velocity_weight * float(np.mean(np.square(velocity_error)))
            axial_dofs = self._p08_axial_tracking_dofs(target.task_id)
            if axial_dofs is not None:
                tracking_cost += self.config.p08_axial_position_weight * float(
                    np.square(np.sum(position_error[axial_dofs]))
                )
                tracking_cost += self.config.p08_axial_velocity_weight * float(
                    np.square(np.sum(velocity_error[axial_dofs]))
                )
            root_orientation_dofs = self._p08_root_orientation_tracking_dofs(target.task_id)
            if root_orientation_dofs is not None:
                tracking_cost += self.config.p08_root_orientation_weight * float(
                    np.mean(np.square(position_error[root_orientation_dofs]))
                )
            if target.task_id.split("_", 1)[0] == "P08":
                tracking_cost += self.config.p08_position_abs_weight * float(np.max(np.square(position_error)))
            total += step_weight * tracking_cost
            total += self.config.effort_weight * float(np.mean(np.square(ctrl)))
            total += self.config.rate_weight * float(np.mean(np.square(ctrl - prior)))
            prior = ctrl
        return float(total)


class ScriptedPhysicalControlPlanner:
    """Replay a full-action teacher's already captured physical controls.

    The input must come from the repository physical collector's
    ``teacher_ctrl_physical`` field.  Normalized ``teacher_action`` is never a
    fallback.  The producer still executes every value on the current exact CPU
    model and bases success on that new rollout's tracking/replay QC.
    """

    def __init__(self, model: mujoco.MjModel, controls: Any):
        self.model = model
        names = complete_actuator_names(model)
        channel_contract = resolve_muscle_channel_contract(model, names)
        ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
        values = np.asarray(controls, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != int(model.nu):
            raise ValueError(f"scripted physical controls must have shape [T,{int(model.nu)}]")
        if not np.all(np.isfinite(values)):
            raise ValueError("scripted physical controls contain non-finite values")
        if np.any(values < ctrlrange[:, 0]) or np.any(values > ctrlrange[:, 1]):
            raise ValueError(
                "scripted control lies outside [0,1] physical muscle ctrlrange; "
                "signed/normalized Paper_Need actions are forbidden"
            )
        self.controls = values.copy()
        self.ctrlrange = np.asarray(ctrlrange, dtype=np.float64).copy()
        self.actadr = np.asarray(channel_contract.actuator_actadr, dtype=np.int32)

    def reset(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("policy replay seed must be a non-negative integer")

    def initialize(
        self,
        data: mujoco.MjData,
        target: MotionTarget,
        *,
        contact_contract: FootFloorContactContract,
        min_contact_normal_force: float,
    ) -> RolloutInitialization:
        """Preserve the historical zero-state contract for explicit control replay.

        A policy-control import does not currently carry a source integration
        state.  It therefore cannot claim an inferred muscle state.  The zero
        state remains explicit and is captured exactly for replay; production
        success still depends on the resulting closed-loop QC.
        """

        target = target.validated(self.model)
        _require_first_target_state(self.model, data, target)
        data.act[:] = 0.0
        data.ctrl[:] = self.ctrlrange[:, 0]
        mujoco.mj_forward(self.model, data)
        return _build_rollout_initialization(
            self.model,
            data,
            target,
            actadr=self.actadr,
            contract=_ZERO_INITIAL_STATE_CONTRACT,
            target_acceleration_method="not_applicable_explicit_control_replay",
            solver_kind="none_explicit_zero_state",
            solver_status=0,
            solver_iterations=0,
            solver_optimality=0.0,
            linearized_acceleration_residual_norm=0.0,
            contact_contract=contact_contract,
            min_contact_normal_force=min_contact_normal_force,
        )

    def plan(
        self,
        data: mujoco.MjData,
        target: MotionTarget,
        transition_index: int,
        previous_ctrl: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        del data, target, previous_ctrl
        index = int(transition_index)
        if index < 0 or index >= self.controls.shape[0]:
            raise IndexError("scripted policy controls do not cover the requested transition")
        return self.controls[index].copy(), 0.0


class CanonicalTonicControlPlanner:
    """Apply one train-only canonical physical muscle control for every transition."""

    def __init__(self, model: mujoco.MjModel, control: Any):
        self.model = model
        names = complete_actuator_names(model)
        channel = resolve_muscle_channel_contract(model, names)
        self.ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
        self.actadr = np.asarray(channel.actuator_actadr, dtype=np.int32)
        self.control = np.asarray(control, dtype=np.float64).copy()
        if self.control.shape != (int(model.nu),) or not np.all(np.isfinite(self.control)):
            raise ValueError("canonical tonic control has wrong model width or non-finite values")
        if np.any(self.control < self.ctrlrange[:, 0]) or np.any(self.control > self.ctrlrange[:, 1]):
            raise ValueError("canonical tonic control lies outside physical ctrlrange")

    def reset(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("canonical tonic seed must be a non-negative integer")

    def initialize(self, data, target, *, contact_contract, min_contact_normal_force):
        target = target.validated(self.model)
        _require_first_target_state(self.model, data, target)
        data.act[:] = 0.0
        data.act[self.actadr] = self.control
        data.ctrl[:] = self.control
        mujoco.mj_forward(self.model, data)
        return _build_rollout_initialization(
            self.model,
            data,
            target,
            actadr=self.actadr,
            contract=_CANONICAL_TONIC_INITIAL_STATE_CONTRACT,
            target_acceleration_method="train_only_canonical_coordinate_mean_not_target_conditioned",
            solver_kind="none_content_addressed_canonical_tonic_control",
            solver_status=0,
            solver_iterations=0,
            solver_optimality=0.0,
            linearized_acceleration_residual_norm=0.0,
            contact_contract=contact_contract,
            min_contact_normal_force=min_contact_normal_force,
        )

    def plan(self, data, target, transition_index, previous_ctrl):
        del data, target, previous_ctrl
        if int(transition_index) < 0:
            raise IndexError("transition index is outside target")
        return self.control.copy(), 0.0


def complete_actuator_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return the exact complete actuator order, rejecting unnamed channels."""

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("physical primitive producer requires mujoco.MjModel")
    names: list[str] = []
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            raise ValueError(f"compiled model actuator {actuator_id} has no stable name")
        names.append(str(name))
    if not names or len(names) != len(set(names)):
        raise ValueError("compiled model actuator names must be non-empty and unique")
    return tuple(names)


def resolve_foot_floor_contact_contract(
    model: mujoco.MjModel,
    *,
    allow_unavailable: bool = False,
) -> FootFloorContactContract:
    """Resolve the exact production floor and foot geoms by stable names.

    Production never infers laterality from body position or a loose substring
    at runtime.  The complete expected MyoFullBody talus/foot/bofoot name set
    must exist and must match the IDs returned by MuJoCo.  P00 unit fixtures may
    explicitly request an unavailable binding; no production task may do so.
    """

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("foot-floor contact resolution requires mujoco.MjModel")
    missing: list[str] = []

    def resolve(name: str) -> int:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0:
            missing.append(name)
            return -1
        resolved = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if resolved != name:
            raise ValueError(f"MuJoCo geom name/ID round trip failed for {name!r}")
        return geom_id

    floor_id = resolve(_FLOOR_GEOM_NAME)
    left_ids = tuple(resolve(name) for name in _FOOT_GEOM_NAMES["left"])
    right_ids = tuple(resolve(name) for name in _FOOT_GEOM_NAMES["right"])
    if missing:
        reason = f"missing exact floor/foot geoms: {sorted(missing)}"
        if not allow_unavailable:
            raise ValueError(reason)
        return FootFloorContactContract(
            available=False,
            floor_geom_id=None,
            floor_geom_name=None,
            left_geom_ids=(),
            left_geom_names=(),
            right_geom_ids=(),
            right_geom_names=(),
            unavailable_reason=reason,
        )
    if len({*left_ids, *right_ids, floor_id}) != 1 + len(left_ids) + len(right_ids):
        raise ValueError("floor and left/right foot geom IDs must be unique and disjoint")

    expected = set(_FOOT_GEOM_NAMES["left"]) | set(_FOOT_GEOM_NAMES["right"])
    observed: set[str] = set()
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        if (name.startswith("l_") or name.startswith("r_")) and (
            "foot" in name.casefold() or "talus" in name.casefold()
        ):
            observed.add(str(name))
    if observed != expected:
        raise ValueError(
            "model foot/talus/bofoot geom inventory differs from the stable production contract: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    return FootFloorContactContract(
        available=True,
        floor_geom_id=floor_id,
        floor_geom_name=_FLOOR_GEOM_NAME,
        left_geom_ids=left_ids,
        left_geom_names=_FOOT_GEOM_NAMES["left"],
        right_geom_ids=right_ids,
        right_geom_names=_FOOT_GEOM_NAMES["right"],
    )


def _foot_floor_contact_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    contract: FootFloorContactContract,
    min_normal_force: float,
) -> tuple[bool, bool, float, float]:
    if not np.isfinite(min_normal_force) or min_normal_force <= 0.0:
        raise ValueError("minimum exact contact normal force must be finite and positive")
    if not contract.available:
        return False, False, 0.0, 0.0
    if contract.floor_geom_id is None:
        raise ValueError("available foot-floor contact contract has no floor geom ID")
    left_ids = set(contract.left_geom_ids)
    right_ids = set(contract.right_geom_ids)
    left_force = 0.0
    right_force = 0.0
    contact_force = np.zeros((6,), dtype=np.float64)
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 == contract.floor_geom_id:
            other = geom2
        elif geom2 == contract.floor_geom_id:
            other = geom1
        else:
            continue
        if other not in left_ids and other not in right_ids:
            continue
        contact_force.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        normal = float(contact_force[0])
        if not np.isfinite(normal):
            raise ValueError("MuJoCo returned a non-finite contact normal force")
        positive_normal = max(normal, 0.0)
        if other in left_ids:
            left_force += positive_normal
        if other in right_ids:
            right_force += positive_normal
    return (
        bool(left_force >= min_normal_force),
        bool(right_force >= min_normal_force),
        float(left_force),
        float(right_force),
    )


def _require_first_target_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: MotionTarget,
) -> None:
    if not np.array_equal(np.asarray(data.qpos), target.qpos[0]) or not np.array_equal(
        np.asarray(data.qvel), target.qvel[0]
    ):
        raise ValueError("planner initialization data must equal the first target state")
    if np.asarray(data.act).shape != (int(model.na),) or np.asarray(data.ctrl).shape != (int(model.nu),):
        raise ValueError("planner initialization simulator state has inconsistent muscle widths")


def _build_rollout_initialization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: MotionTarget,
    *,
    actadr: np.ndarray,
    contract: str,
    target_acceleration_method: str,
    solver_kind: str,
    solver_status: int,
    solver_iterations: int,
    solver_optimality: float,
    linearized_acceleration_residual_norm: float,
    contact_contract: FootFloorContactContract,
    min_contact_normal_force: float,
) -> RolloutInitialization:
    """Capture exact forward and no-hidden-step shadow evidence for initialization."""

    duration = float(target.transition_substeps[0]) * float(model.opt.timestep)
    desired_qacc = (target.qvel[1] - target.qvel[0]) / duration
    forward_error = np.asarray(data.qacc, dtype=np.float64).copy() - desired_qacc
    initial_left, initial_right, initial_left_force, initial_right_force = _foot_floor_contact_sample(
        model,
        data,
        contract=contact_contract,
        min_normal_force=min_contact_normal_force,
    )
    initial_state = _capture_integration_state(model, data)
    shadow = mujoco.MjData(model)
    _restore_integration_state(model, shadow, initial_state)
    shadow_qpos_rows: list[np.ndarray] = []
    shadow_qvel_rows: list[np.ndarray] = []
    shadow_left_rows: list[bool] = []
    shadow_right_rows: list[bool] = []
    shadow_left_force_rows: list[float] = []
    shadow_right_force_rows: list[float] = []
    for _ in range(int(target.transition_substeps[0])):
        mujoco.mj_step(model, shadow)
        shadow_left, shadow_right, shadow_left_force, shadow_right_force = _foot_floor_contact_sample(
            model,
            shadow,
            contract=contact_contract,
            min_normal_force=min_contact_normal_force,
        )
        shadow_qpos_rows.append(np.asarray(shadow.qpos, dtype=np.float64).copy())
        shadow_qvel_rows.append(np.asarray(shadow.qvel, dtype=np.float64).copy())
        shadow_left_rows.append(shadow_left)
        shadow_right_rows.append(shadow_right)
        shadow_left_force_rows.append(shadow_left_force)
        shadow_right_force_rows.append(shadow_right_force)
    initialization = RolloutInitialization(
        contract=str(contract),
        target_acceleration_method=str(target_acceleration_method),
        initial_activation=np.asarray(data.act[actadr], dtype=np.float64).copy(),
        initial_ctrl=np.asarray(data.ctrl, dtype=np.float64).copy(),
        solver_kind=str(solver_kind),
        solver_status=int(solver_status),
        solver_iterations=int(solver_iterations),
        solver_optimality=float(solver_optimality),
        linearized_acceleration_residual_norm=float(linearized_acceleration_residual_norm),
        forward_acceleration_error=forward_error,
        shadow_transition_duration=float(duration),
        initial_left_foot_floor_contact=bool(initial_left),
        initial_right_foot_floor_contact=bool(initial_right),
        initial_left_foot_floor_normal_force=float(initial_left_force),
        initial_right_foot_floor_normal_force=float(initial_right_force),
        shadow_qpos=_rows(shadow_qpos_rows, width=int(model.nq)),
        shadow_qvel=_rows(shadow_qvel_rows, width=int(model.nv)),
        shadow_position_error=_position_difference(model, shadow.qpos, target.qpos[1]),
        shadow_velocity_error=np.asarray(shadow.qvel - target.qvel[1], dtype=np.float64).copy(),
        shadow_left_foot_floor_contact=np.asarray(shadow_left_rows, dtype=np.bool_),
        shadow_right_foot_floor_contact=np.asarray(shadow_right_rows, dtype=np.bool_),
        shadow_left_foot_floor_normal_force=np.asarray(shadow_left_force_rows, dtype=np.float64),
        shadow_right_foot_floor_normal_force=np.asarray(shadow_right_force_rows, dtype=np.float64),
        shadow_final_integration_state=_capture_integration_state(model, shadow),
    )
    numeric = (
        initialization.initial_activation,
        initialization.initial_ctrl,
        initialization.forward_acceleration_error,
        initialization.shadow_qpos,
        initialization.shadow_qvel,
        initialization.shadow_position_error,
        initialization.shadow_velocity_error,
        initialization.shadow_left_foot_floor_normal_force,
        initialization.shadow_right_foot_floor_normal_force,
        initialization.shadow_final_integration_state,
    )
    scalar = (
        initialization.solver_optimality,
        initialization.linearized_acceleration_residual_norm,
        initialization.shadow_transition_duration,
        initialization.initial_left_foot_floor_normal_force,
        initialization.initial_right_foot_floor_normal_force,
    )
    shadow_count = int(target.transition_substeps[0])
    if (
        initialization.shadow_qpos.shape != (shadow_count, int(model.nq))
        or initialization.shadow_qvel.shape != (shadow_count, int(model.nv))
        or any(
            array.shape != (shadow_count,)
            for array in (
                initialization.shadow_left_foot_floor_contact,
                initialization.shadow_right_foot_floor_contact,
                initialization.shadow_left_foot_floor_normal_force,
                initialization.shadow_right_foot_floor_normal_force,
            )
        )
    ):
        raise ValueError("rollout initialization shadow evidence has inconsistent substep shapes")
    if not all(np.all(np.isfinite(value)) for value in numeric) or not all(np.isfinite(value) for value in scalar):
        raise ValueError("rollout initialization evidence contains non-finite values")
    return initialization


def resolve_axial_rotation_signal_contract(
    model: mujoco.MjModel,
    *,
    allow_unavailable: bool = False,
) -> AxialRotationSignalContract:
    """Resolve P08 only through the six declared hinge names, never a proxy."""

    try:
        joint_ids: list[int] = []
        qpos_addresses: list[int] = []
        dof_addresses: list[int] = []
        missing: list[str] = []
        for name in _AXIAL_ROTATION_JOINT_NAMES:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                missing.append(name)
                continue
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"P08 axial joint {name!r} is not a hinge")
            joint_ids.append(joint_id)
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        if missing:
            raise ValueError(f"P08 model is missing named axial hinge joints: {missing}")
        if len(set(joint_ids)) != len(_AXIAL_ROTATION_JOINT_NAMES):
            raise ValueError("P08 named axial joints do not resolve to unique joint IDs")
        if len(set(qpos_addresses)) != len(_AXIAL_ROTATION_JOINT_NAMES):
            raise ValueError("P08 named axial joints do not resolve to unique qpos addresses")
        if len(set(dof_addresses)) != len(_AXIAL_ROTATION_JOINT_NAMES):
            raise ValueError("P08 named axial joints do not resolve to unique dof addresses")
        if any(address < 0 or address >= int(model.nq) for address in qpos_addresses):
            raise ValueError("P08 axial qpos address lies outside the model")
        if any(address < 0 or address >= int(model.nv) for address in dof_addresses):
            raise ValueError("P08 axial dof address lies outside the model")

        root_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root"))
        if root_joint_id < 0 or int(model.jnt_type[root_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("P08 signal contract requires a free joint named 'root'")
        root_qpos_address = int(model.jnt_qposadr[root_joint_id])
        if root_qpos_address < 0 or root_qpos_address + 6 >= int(model.nq):
            raise ValueError("P08 root free-joint qpos address lies outside the model")
        return AxialRotationSignalContract(
            available=True,
            joint_names=_AXIAL_ROTATION_JOINT_NAMES,
            joint_ids=tuple(joint_ids),
            qpos_addresses=tuple(qpos_addresses),
            dof_addresses=tuple(dof_addresses),
            root_joint_id=root_joint_id,
            root_qpos_address=root_qpos_address,
        )
    except ValueError as error:
        if not allow_unavailable:
            raise
        return AxialRotationSignalContract(
            available=False,
            joint_names=_AXIAL_ROTATION_JOINT_NAMES,
            joint_ids=(),
            qpos_addresses=(),
            dof_addresses=(),
            root_joint_id=None,
            root_qpos_address=None,
            unavailable_reason=str(error),
        )


def _root_yaw_from_qpos(qpos: np.ndarray, root_qpos_address: int) -> np.ndarray:
    quaternions = np.asarray(qpos[:, root_qpos_address + 3 : root_qpos_address + 7], dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.all(np.isfinite(quaternions)) or np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("P08 root qpos contains a non-finite or zero-norm quaternion")
    normalized = quaternions / norms[:, None]
    w, x, y, z = normalized.T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.unwrap(yaw)


def reconstruct_axial_rotation_evidence(
    model: mujoco.MjModel,
    *,
    initial_qpos: Any,
    post_transition_qpos: Any,
    transition_substeps: Any,
    allow_unavailable: bool = False,
) -> AxialRotationEvidence:
    """Reconstruct the identical named-qpos P08 signal for target/actual/replay."""

    initial = np.asarray(initial_qpos, dtype=np.float64)
    qpos = np.asarray(post_transition_qpos, dtype=np.float64)
    substeps = np.asarray(transition_substeps)
    count = int(qpos.shape[0]) if qpos.ndim == 2 else -1
    if (
        initial.shape != (int(model.nq),)
        or qpos.shape != (count, int(model.nq))
        or count < 0
        or substeps.shape != (count,)
        or np.issubdtype(substeps.dtype, np.bool_)
        or not np.issubdtype(substeps.dtype, np.integer)
        or np.any(substeps <= 0)
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(qpos))
    ):
        raise ValueError("P08 axial-signal reconstruction inputs are malformed")
    contract = resolve_axial_rotation_signal_contract(model, allow_unavailable=allow_unavailable)
    if not contract.available:
        zeros = np.zeros((count,), dtype=np.float64)
        return AxialRotationEvidence(
            position=zeros.copy(),
            velocity=zeros.copy(),
            root_yaw=zeros.copy(),
            root_xy=np.zeros((count, 2), dtype=np.float64),
            initial_position=0.0,
            initial_root_yaw=0.0,
            initial_root_xy=np.zeros((2,), dtype=np.float64),
            contract=contract,
        )

    states = np.concatenate((initial[None, :], qpos), axis=0)
    addresses = np.asarray(contract.qpos_addresses, dtype=np.int32)
    position_states = np.sum(states[:, addresses], axis=1)
    duration = substeps.astype(np.float64) * float(model.opt.timestep)
    velocity = np.diff(position_states) / duration
    assert contract.root_qpos_address is not None
    yaw_states = _root_yaw_from_qpos(states, contract.root_qpos_address)
    root_xy_states = states[:, contract.root_qpos_address : contract.root_qpos_address + 2]
    evidence = AxialRotationEvidence(
        position=position_states[1:].copy(),
        velocity=velocity,
        root_yaw=yaw_states[1:].copy(),
        root_xy=root_xy_states[1:].copy(),
        initial_position=float(position_states[0]),
        initial_root_yaw=float(yaw_states[0]),
        initial_root_xy=root_xy_states[0].copy(),
        contract=contract,
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (evidence.position, evidence.velocity, evidence.root_yaw, evidence.root_xy)
    ):
        raise ValueError("P08 reconstructed axial evidence contains non-finite values")
    return evidence


def _axial_semantic_arguments(evidence: AxialRotationEvidence) -> dict[str, Any]:
    """Bind one reconstructed signal to the semantic evaluator without drift."""

    return {
        "axial_position": evidence.position,
        "axial_velocity": evidence.velocity,
        "axial_root_yaw": evidence.root_yaw,
        "axial_root_xy": evidence.root_xy,
        "axial_initial_position": evidence.initial_position,
        "axial_initial_root_yaw": evidence.initial_root_yaw,
        "axial_initial_root_xy": evidence.initial_root_xy,
        "axial_signal_contract": evidence.contract,
    }


def _root_kinematic_binding(model: mujoco.MjModel) -> tuple[int, int, int]:
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root"))
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("production semantic QC requires a free joint named 'root'")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    body_id = int(model.jnt_bodyid[joint_id])
    if qpos_adr < 0 or qpos_adr + 2 >= int(model.nq) or dof_adr < 0 or dof_adr + 2 >= int(model.nv):
        raise ValueError("root free-joint qpos/qvel addresses lie outside the model")
    return qpos_adr + 2, dof_adr + 2, body_id


def _reconstruct_actual_vertical_kinematics(
    model: mujoco.MjModel,
    *,
    initial_integration_state: Any,
    actual_qpos: Any,
    actual_qvel: Any,
    transition_substeps: Any,
    allow_unavailable: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute root and COM signals from the exact recorded state sequence."""

    qpos = np.asarray(actual_qpos, dtype=np.float64)
    qvel = np.asarray(actual_qvel, dtype=np.float64)
    substeps = np.asarray(transition_substeps)
    count = int(qpos.shape[0]) if qpos.ndim == 2 else -1
    if (
        count < 0
        or qpos.shape != (count, int(model.nq))
        or qvel.shape != (count, int(model.nv))
        or substeps.shape != (count,)
        or np.issubdtype(substeps.dtype, np.bool_)
        or not np.issubdtype(substeps.dtype, np.integer)
        or np.any(substeps <= 0)
    ):
        raise ValueError("actual vertical-signal reconstruction inputs are malformed")
    try:
        root_qpos_z, root_qvel_z, root_body = _root_kinematic_binding(model)
    except ValueError:
        if not allow_unavailable:
            raise
        zeros = np.zeros((count,), dtype=np.float64)
        return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

    data = mujoco.MjData(model)
    _restore_integration_state(model, data, initial_integration_state)
    com_position_states = np.empty((count + 1,), dtype=np.float64)
    com_position_states[0] = float(data.subtree_com[root_body, 2])
    root_position = np.empty((count,), dtype=np.float64)
    root_velocity = np.empty((count,), dtype=np.float64)
    for index in range(count):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        root_position[index] = float(data.qpos[root_qpos_z])
        root_velocity[index] = float(data.qvel[root_qvel_z])
        com_position_states[index + 1] = float(data.subtree_com[root_body, 2])
    duration = substeps.astype(np.float64) * float(model.opt.timestep)
    com_velocity = np.diff(com_position_states) / duration
    return root_position, root_velocity, com_position_states[1:].copy(), com_velocity


def _vertical_signal_for_task(
    task_id: str,
    *,
    root_position: np.ndarray,
    root_velocity: np.ndarray,
    com_position: np.ndarray,
    com_velocity: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Select the one task-authoritative vertical signal without fallback."""

    family = str(task_id).split("_", 1)[0]
    if family == "P01":
        return _ROOT_VERTICAL_SIGNAL, root_position, root_velocity
    if family in _DYNAMIC_COM_TASK_FAMILIES:
        return _COM_VERTICAL_SIGNAL, com_position, com_velocity
    if family == "P08":
        return _P08_AUXILIARY_VERTICAL_SIGNAL, com_position, com_velocity
    if str(task_id) == "P00_synthetic_fixture":
        return _TOY_VERTICAL_SIGNAL, root_position, root_velocity
    return "unsupported_task_vertical_signal", com_position, com_velocity


def _phase_runs(phase_id: np.ndarray) -> list[dict[str, int]]:
    if phase_id.size == 0:
        return []
    runs: list[dict[str, int]] = []
    start = 0
    for index in range(1, int(phase_id.size) + 1):
        if index == int(phase_id.size) or int(phase_id[index]) != int(phase_id[start]):
            runs.append({"phase_id": int(phase_id[start]), "start": start, "end": index, "length": index - start})
            start = index
    return runs


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


def _max_abs_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.max(np.abs(values)))


def _negative_to_positive_crossings(values: np.ndarray) -> int:
    if values.size < 2:
        return 0
    return int(np.count_nonzero((values[:-1] < 0.0) & (values[1:] >= 0.0)))


def evaluate_task_contact_semantics(
    *,
    task_id: str,
    phase_id: Any,
    left_contact: Any,
    right_contact: Any,
    left_normal_force: Any,
    right_normal_force: Any,
    vertical_position: Any,
    vertical_velocity: Any,
    config: RolloutQCConfig,
    evidence_kind: str,
    vertical_signal: str | None = None,
    axial_position: Any | None = None,
    axial_velocity: Any | None = None,
    axial_root_yaw: Any | None = None,
    axial_root_xy: Any | None = None,
    axial_initial_position: float | None = None,
    axial_initial_root_yaw: float | None = None,
    axial_initial_root_xy: Any | None = None,
    axial_signal_contract: Mapping[str, Any] | AxialRotationSignalContract | None = None,
) -> dict[str, Any]:
    """Apply the fail-closed task-specific contact/phase state machine."""

    config = config.validated()
    phases = np.asarray(phase_id)
    left = np.asarray(left_contact, dtype=np.bool_)
    right = np.asarray(right_contact, dtype=np.bool_)
    left_force = np.asarray(left_normal_force, dtype=np.float64)
    right_force = np.asarray(right_normal_force, dtype=np.float64)
    position = np.asarray(vertical_position, dtype=np.float64)
    velocity = np.asarray(vertical_velocity, dtype=np.float64)
    axial = np.asarray([] if axial_position is None else axial_position, dtype=np.float64)
    axial_speed = np.asarray([] if axial_velocity is None else axial_velocity, dtype=np.float64)
    root_yaw = np.asarray([] if axial_root_yaw is None else axial_root_yaw, dtype=np.float64)
    root_xy = np.asarray([] if axial_root_xy is None else axial_root_xy, dtype=np.float64)
    initial_root_xy = np.asarray([] if axial_initial_root_xy is None else axial_initial_root_xy, dtype=np.float64)
    if isinstance(axial_signal_contract, AxialRotationSignalContract):
        axial_contract = axial_signal_contract.as_dict()
    elif isinstance(axial_signal_contract, Mapping):
        axial_contract = dict(axial_signal_contract)
    else:
        axial_contract = {}
    count = int(phases.size) if phases.ndim == 1 else -1
    aligned = count >= 0 and all(
        value.shape == (count,) for value in (left, right, left_force, right_force, position, velocity)
    )
    finite = bool(aligned and np.all(np.isfinite(left_force)) and np.all(np.isfinite(right_force)))
    finite = bool(finite and np.all(np.isfinite(position)) and np.all(np.isfinite(velocity)))
    force_consistent = bool(
        aligned
        and np.array_equal(left, left_force >= config.min_contact_normal_force)
        and np.array_equal(right, right_force >= config.min_contact_normal_force)
    )
    family = str(task_id).split("_", 1)[0]
    axial_required = family == "P08"
    axial_aligned = bool(
        not axial_required
        or (
            axial.shape == (count,)
            and axial_speed.shape == (count,)
            and root_yaw.shape == (count,)
            and root_xy.shape == (count, 2)
            and initial_root_xy.shape == (2,)
            and axial_initial_position is not None
            and axial_initial_root_yaw is not None
        )
    )
    axial_finite = bool(
        not axial_required
        or (
            axial_aligned
            and np.all(np.isfinite(axial))
            and np.all(np.isfinite(axial_speed))
            and np.all(np.isfinite(root_yaw))
            and np.all(np.isfinite(root_xy))
            and np.all(np.isfinite(initial_root_xy))
            and np.isfinite(axial_initial_position)
            and np.isfinite(axial_initial_root_yaw)
        )
    )
    axial_contract_valid = bool(
        not axial_required
        or (
            axial_contract.get("schema_version") == _AXIAL_ROTATION_CONTRACT_SCHEMA_VERSION
            and axial_contract.get("available") is True
            and tuple(axial_contract.get("joint_names", ())) == _AXIAL_ROTATION_JOINT_NAMES
            and axial_contract.get("position_signal") == _AXIAL_ROTATION_POSITION_SIGNAL
            and axial_contract.get("velocity_signal") == _AXIAL_ROTATION_VELOCITY_SIGNAL
            and axial_contract.get("sample_time") == "post_transition"
            and axial_contract.get("proxy_fallback_allowed") is False
        )
    )
    if family == "P01":
        expected_vertical_signal = _ROOT_VERTICAL_SIGNAL
    elif family in _DYNAMIC_COM_TASK_FAMILIES:
        expected_vertical_signal = _COM_VERTICAL_SIGNAL
    elif family == "P08":
        expected_vertical_signal = _P08_AUXILIARY_VERTICAL_SIGNAL
    elif str(task_id) == "P00_synthetic_fixture":
        expected_vertical_signal = _TOY_VERTICAL_SIGNAL
    else:
        expected_vertical_signal = "unsupported_task_vertical_signal"
    selected_vertical_signal = expected_vertical_signal if vertical_signal is None else str(vertical_signal)
    runs = _phase_runs(phases.astype(np.int64, copy=False)) if phases.ndim == 1 else []
    evidence = {
        "left_foot_floor_contact": left.tolist() if aligned else [],
        "right_foot_floor_contact": right.tolist() if aligned else [],
        "left_foot_floor_normal_force": left_force.tolist() if aligned else [],
        "right_foot_floor_normal_force": right_force.tolist() if aligned else [],
        "vertical_position": position.tolist() if aligned else [],
        "vertical_velocity": velocity.tolist() if aligned else [],
        "axial_position": axial.tolist() if axial_aligned and axial_required else [],
        "axial_velocity": axial_speed.tolist() if axial_aligned and axial_required else [],
        "axial_root_yaw": root_yaw.tolist() if axial_aligned and axial_required else [],
        "axial_root_xy": root_xy.tolist() if axial_aligned and axial_required else [],
        "axial_initial_position": float(axial_initial_position) if axial_finite and axial_required else None,
        "axial_initial_root_yaw": float(axial_initial_root_yaw) if axial_finite and axial_required else None,
        "axial_initial_root_xy": initial_root_xy.tolist() if axial_finite and axial_required else [],
    }
    if str(task_id) == "P00_synthetic_fixture":
        return {
            "passed": True,
            "task_id": str(task_id),
            "task_family": "P00",
            "evidence_kind": str(evidence_kind),
            "vertical_signal": selected_vertical_signal,
            "axial_signal_contract": axial_contract if axial_required else None,
            "semantic_gate": "skipped_explicit_toy_fixture",
            "production_eligible": False,
            "supported": True,
            "gates": {"explicit_p00_toy_fixture": True},
            "metrics": {"transition_count": max(count, 0)},
            "phase_runs": runs,
            "evidence": evidence,
        }

    expected_order = _TASK_PHASE_ORDER.get(family)
    gates: dict[str, bool] = {
        "aligned_contact_phase_arrays": bool(aligned),
        "finite_contact_and_vertical_signals": bool(finite),
        "contact_bool_matches_normal_force_threshold": bool(force_consistent),
        "vertical_signal_contract": selected_vertical_signal == expected_vertical_signal,
        "aligned_axial_signal_arrays": axial_aligned,
        "finite_axial_signals": axial_finite,
        "named_axial_signal_contract": axial_contract_valid,
        "supported_task_semantics": expected_order is not None,
    }
    if expected_order is None or not aligned or not axial_aligned or not axial_finite or count <= 0:
        return {
            "passed": False,
            "task_id": str(task_id),
            "task_family": family,
            "evidence_kind": str(evidence_kind),
            "vertical_signal": selected_vertical_signal,
            "axial_signal_contract": axial_contract if axial_required else None,
            "semantic_gate": "required_fail_closed",
            "production_eligible": False,
            "supported": expected_order is not None,
            "gates": gates,
            "metrics": {"transition_count": max(count, 0)},
            "phase_runs": runs,
            "evidence": evidence,
        }

    run_order = tuple(run["phase_id"] for run in runs)
    gates["phase_order_exact"] = run_order == expected_order
    gates["each_phase_min_transitions"] = bool(
        run_order == expected_order and all(run["length"] >= config.min_phase_transitions for run in runs)
    )
    both = left & right
    air = ~left & ~right
    support_count = left.astype(np.int8) + right.astype(np.int8)
    metrics: dict[str, Any] = {
        "transition_count": count,
        "bilateral_contact_count": int(np.count_nonzero(both)),
        "airborne_count": int(np.count_nonzero(air)),
        "left_first_contact_index": _first_true(left),
        "right_first_contact_index": _first_true(right),
        "vertical_speed_abs_max": _max_abs_or_none(velocity),
    }

    def mask(phase: int) -> np.ndarray:
        return phases == int(phase)

    if family == "P01":
        gates["bilateral_contact_continuous"] = bool(np.all(both))
        gates["stable_root_vertical_speed"] = bool(np.all(np.abs(velocity) <= config.max_stable_root_vertical_speed))
    elif family == "P05":
        ready = mask(0)
        descent = mask(1)
        reversal = mask(2)
        propulsion = mask(3)
        reversal_indices = np.flatnonzero(reversal)
        low_index = int(np.argmin(position))
        unique_low = int(np.count_nonzero(position == position[low_index])) == 1
        gates["bilateral_contact_entire_primitive"] = bool(np.all(both))
        gates["ready_min_frames"] = int(np.count_nonzero(ready)) >= config.min_ready_frames
        gates["ready_com_vertical_speed"] = bool(np.all(np.abs(velocity[ready]) <= config.max_ready_com_vertical_speed))
        gates["descent_com_excursion"] = bool(
            np.count_nonzero(descent) >= 2
            and position[descent][0] - position[descent][-1] >= config.min_com_vertical_excursion
        )
        gates["reversal_contains_unique_low"] = bool(
            reversal_indices.size > 0
            and unique_low
            and low_index in set(reversal_indices.tolist())
            and _negative_to_positive_crossings(velocity[reversal]) == 1
        )
        gates["propulsive_extension_rises"] = bool(
            np.count_nonzero(propulsion) >= 2
            and position[propulsion][-1] - position[propulsion][0] >= config.min_com_vertical_excursion
        )
        metrics["vertical_low_index"] = low_index
        metrics["vertical_low_is_unique"] = unique_low
        metrics["reversal_negative_to_positive_crossings"] = _negative_to_positive_crossings(velocity[reversal])
    elif family == "P06":
        preload_propulsion = mask(0) | mask(1)
        toe_off = mask(2)
        low_flight = mask(3)
        first_not_both = _first_true(~both)
        first_air = _first_true(air)
        monotonic = first_not_both is not None and first_air is not None
        if monotonic:
            monotonic = bool(
                np.all(support_count[:first_not_both] == 2)
                and np.all(support_count[first_not_both:first_air] == 1)
                and np.all(support_count[first_air:] == 0)
            )
        single_frames = None if first_not_both is None or first_air is None else first_air - first_not_both
        gates["preload_and_propulsion_bilateral"] = bool(np.all(both[preload_propulsion]))
        gates["both_single_air_monotonic_no_recontact"] = bool(monotonic)
        gates["single_support_within_lag_limit"] = bool(
            single_frames is not None and single_frames <= config.max_bilateral_contact_lag_frames
        )
        gates["toe_off_phase_contains_contact_loss"] = bool(
            first_not_both is not None and first_air is not None and toe_off[first_not_both] and toe_off[first_air]
        )
        gates["low_flight_airborne_min_frames"] = bool(
            np.count_nonzero(low_flight) >= config.min_low_flight_frames and np.all(air[low_flight])
        )
        metrics["first_non_bilateral_index"] = first_not_both
        metrics["first_airborne_index"] = first_air
        metrics["single_support_frames"] = single_frames
    elif family == "P07":
        precontact = mask(0)
        initial_contact = mask(1)
        stabilization = mask(3)
        left_first = _first_true(left)
        right_first = _first_true(right)
        first_any = None if left_first is None or right_first is None else min(left_first, right_first)
        first_both = None if left_first is None or right_first is None else max(left_first, right_first)
        lag = None if left_first is None or right_first is None else abs(left_first - right_first)
        gates["precontact_airborne_min_frames"] = bool(
            np.count_nonzero(precontact) >= config.min_precontact_air_frames and np.all(air[precontact])
        )
        gates["initial_contact_event_in_phase"] = bool(first_any is not None and initial_contact[first_any])
        gates["bilateral_landing_lag"] = bool(lag is not None and lag <= config.max_bilateral_contact_lag_frames)
        gates["no_contact_loss_after_bilateral_landing"] = bool(first_both is not None and np.all(both[first_both:]))
        gates["landing_stabilization_bilateral_min_frames"] = bool(
            np.count_nonzero(stabilization) >= config.min_landing_stabilization_frames and np.all(both[stabilization])
        )
        gates["landing_stabilization_com_vertical_speed"] = bool(
            np.all(np.abs(velocity[stabilization]) <= config.max_post_impact_com_vertical_speed)
        )
        metrics["first_any_contact_index"] = first_any
        metrics["first_bilateral_contact_index"] = first_both
        metrics["bilateral_contact_lag_frames"] = lag
    elif family == "P08":
        ready = mask(0)
        acceleration = mask(1)
        maximum = mask(2)
        recenter = mask(3)
        pre_recenter = ready | acceleration | maximum
        support_code = left.astype(np.int8) + 2 * right.astype(np.int8)
        support_change_indices = np.flatnonzero(np.diff(support_code) != 0) + 1
        absolute_deviation = np.abs(axial - float(axial_initial_position))
        peak_index = int(np.argmax(absolute_deviation))
        excursion = float(absolute_deviation[peak_index])
        unique_peak = int(np.count_nonzero(absolute_deviation == excursion)) == 1
        direction = float(np.sign(axial[peak_index] - float(axial_initial_position)))
        acceleration_signed = direction * axial_speed[acceleration]
        recenter_signed = -direction * axial_speed[recenter]
        acceleration_fraction = 0.0 if acceleration_signed.size == 0 else float(np.mean(acceleration_signed >= 0.0))
        recenter_fraction = 0.0 if recenter_signed.size == 0 else float(np.mean(recenter_signed >= 0.0))
        recenter_error = abs(float(axial[-1]) - float(axial_initial_position))
        yaw_states = np.concatenate(([float(axial_initial_root_yaw)], root_yaw))
        root_yaw_excursion = float(np.ptp(yaw_states))
        root_xy_displacement = np.linalg.norm(root_xy - initial_root_xy[None, :], axis=1)
        root_xy_displacement_max = float(np.max(root_xy_displacement))
        pre_support = support_code[pre_recenter]

        gates["exact_any_foot_contact_entire_primitive"] = bool(np.all(~air))
        gates["phases_0_to_2_fixed_support"] = bool(
            pre_support.size > 0 and np.all(pre_support > 0) and np.unique(pre_support).size == 1
        )
        gates["neutral_support_min_frames"] = bool(np.count_nonzero(ready) >= config.min_ready_frames)
        gates["neutral_axial_speed"] = bool(np.all(np.abs(axial_speed[ready]) <= config.max_axial_neutral_speed))
        gates["axial_excursion"] = bool(excursion >= config.min_axial_rotation_excursion)
        gates["unique_axial_extremum_in_phase_2"] = bool(unique_peak and maximum[peak_index])
        gates["rotation_acceleration_signed_monotonic_fraction"] = bool(
            acceleration_fraction >= config.min_axial_signed_monotonic_fraction
        )
        gates["deceleration_recenter_signed_monotonic_fraction"] = bool(
            recenter_fraction >= config.min_axial_signed_monotonic_fraction
        )
        gates["terminal_axial_recenter_error"] = bool(recenter_error <= config.max_axial_recenter_error)
        gates["root_yaw_excursion"] = bool(root_yaw_excursion <= config.max_axial_root_yaw_excursion)
        gates["root_xy_displacement"] = bool(root_xy_displacement_max <= config.max_axial_root_xy_displacement)
        metrics["axial_peak_index"] = peak_index
        metrics["axial_peak_is_unique"] = unique_peak
        metrics["axial_rotation_direction"] = direction
        metrics["axial_rotation_excursion"] = excursion
        metrics["phase_1_signed_monotonic_fraction"] = acceleration_fraction
        metrics["phase_3_signed_monotonic_fraction"] = recenter_fraction
        metrics["terminal_axial_recenter_error"] = recenter_error
        metrics["root_yaw_excursion"] = root_yaw_excursion
        metrics["root_xy_displacement_max"] = root_xy_displacement_max
        metrics["support_change_indices"] = support_change_indices.tolist()
    elif family == "P11":
        ready = mask(0)
        countermovement = mask(1)
        propulsion = mask(2)
        flight = mask(3)
        landing = mask(4)
        propulsion_indices = np.flatnonzero(propulsion)
        landing_indices = np.flatnonzero(landing)
        low_index = int(np.argmin(position[: int(propulsion_indices[-1]) + 1])) if propulsion_indices.size else -1
        unique_low = bool(
            low_index >= 0 and np.count_nonzero(position[: int(propulsion_indices[-1]) + 1] == position[low_index]) == 1
        )
        gates["ready_bilateral_min_frames"] = bool(
            np.count_nonzero(ready) >= config.min_ready_frames and np.all(both[ready])
        )
        gates["ready_vertical_speed"] = bool(np.all(np.abs(velocity[ready]) <= config.max_ready_com_vertical_speed))
        gates["countermovement_bilateral_unique_reversal"] = bool(
            np.all(both[countermovement])
            and unique_low
            and low_index in set(np.flatnonzero(countermovement).tolist())
            and _negative_to_positive_crossings(velocity[countermovement]) == 1
        )
        if propulsion_indices.size:
            prop_support = support_count[propulsion]
            prop_both_end = _first_true(prop_support < 2)
            prop_air = _first_true(prop_support == 0)
            prop_monotonic = prop_both_end is not None and prop_both_end > 0 and prop_air is not None
            if prop_monotonic:
                prop_monotonic = bool(
                    np.all(prop_support[:prop_both_end] == 2)
                    and np.all(prop_support[prop_both_end:prop_air] == 1)
                    and np.all(prop_support[prop_air:] == 0)
                )
            prop_single = None if prop_both_end is None or prop_air is None else prop_air - prop_both_end
        else:
            prop_monotonic = False
            prop_single = None
        gates["propulsion_both_single_air_no_recontact"] = bool(prop_monotonic)
        gates["propulsion_single_support_within_lag_limit"] = bool(
            prop_single is not None and prop_single <= config.max_bilateral_contact_lag_frames
        )
        gates["flight_airborne_min_frames"] = bool(
            np.count_nonzero(flight) >= config.min_low_flight_frames and np.all(air[flight])
        )
        left_landing = _first_true(left[landing])
        right_landing = _first_true(right[landing])
        landing_lag = None if left_landing is None or right_landing is None else abs(left_landing - right_landing)
        landing_both_local = None if left_landing is None or right_landing is None else max(left_landing, right_landing)
        landing_both_sustained = bool(
            landing_both_local is not None and np.all(both[landing_indices[landing_both_local] :])
        )
        terminal_start = max(0, count - config.min_landing_stabilization_frames)
        gates["controlled_landing_lag"] = bool(
            landing_lag is not None and landing_lag <= config.max_bilateral_contact_lag_frames
        )
        gates["controlled_landing_no_contact_loss"] = landing_both_sustained
        gates["controlled_landing_terminal_hold"] = bool(
            landing_indices.size >= config.min_landing_stabilization_frames
            and np.all(both[terminal_start:])
            and np.all(np.abs(velocity[terminal_start:]) <= config.max_post_impact_com_vertical_speed)
        )
        metrics["vertical_low_index"] = low_index
        metrics["vertical_low_is_unique"] = unique_low
        metrics["propulsion_single_support_frames"] = prop_single
        metrics["landing_contact_lag_frames"] = landing_lag
    elif family == "P12":
        landing_stabilization = mask(0)
        posture_restore = mask(1)
        ready_hold = mask(2)
        landing_heights = position[landing_stabilization]
        restore_heights = position[posture_restore]
        ready_heights = position[ready_hold]
        height_evidence_valid = bool(
            landing_heights.size > 0
            and restore_heights.size > 0
            and ready_heights.size > 0
            and np.all(np.isfinite(landing_heights))
            and np.all(np.isfinite(restore_heights))
            and np.all(np.isfinite(ready_heights))
        )
        if height_evidence_valid:
            landing_terminal_height = float(landing_heights[-1])
            restore_terminal_height = float(restore_heights[-1])
            ready_min_height = float(np.min(ready_heights))
            restore_rise = restore_terminal_height - landing_terminal_height
            ready_height_margin = ready_min_height - landing_terminal_height
        else:
            landing_terminal_height = None
            restore_terminal_height = None
            ready_min_height = None
            restore_rise = None
            ready_height_margin = None
        gates["bilateral_contact_entire_primitive"] = bool(np.all(both))
        gates["post_impact_initial_vertical_speed"] = bool(
            abs(float(velocity[0])) <= config.max_post_impact_com_vertical_speed
        )
        gates["posture_restore_com_rise"] = bool(
            restore_rise is not None and restore_rise >= config.min_com_vertical_excursion
        )
        gates["ready_hold_min_frames"] = bool(np.count_nonzero(ready_hold) >= config.min_ready_hold_frames)
        gates["ready_hold_bilateral"] = bool(np.all(both[ready_hold]))
        gates["ready_hold_vertical_speed"] = bool(
            np.all(np.abs(velocity[ready_hold]) <= config.max_ready_hold_com_vertical_speed)
        )
        gates["ready_hold_recovered_height"] = bool(
            ready_height_margin is not None and ready_height_margin >= config.min_com_vertical_excursion
        )
        metrics["landing_stabilization_terminal_com_height"] = landing_terminal_height
        metrics["posture_restore_terminal_com_height"] = restore_terminal_height
        metrics["posture_restore_com_rise"] = restore_rise
        metrics["ready_hold_min_com_height"] = ready_min_height
        metrics["ready_hold_min_height_above_landing_baseline"] = ready_height_margin

    return {
        "passed": bool(all(gates.values())),
        "task_id": str(task_id),
        "task_family": family,
        "evidence_kind": str(evidence_kind),
        "vertical_signal": selected_vertical_signal,
        "axial_signal_contract": axial_contract if axial_required else None,
        "semantic_gate": "required_fail_closed",
        "production_eligible": bool(all(gates.values())),
        "supported": True,
        "thresholds": config.semantic_thresholds(),
        "gates": gates,
        "metrics": metrics,
        "phase_runs": runs,
        "evidence": evidence,
    }


def _site_contact_hysteresis(clearance: np.ndarray, config: RolloutQCConfig) -> np.ndarray:
    result = np.zeros(clearance.shape, dtype=np.bool_)
    active = False
    for index, value in enumerate(clearance):
        if active:
            active = bool(value <= config.site_contact_exit_height)
        else:
            active = bool(value <= config.site_contact_enter_height)
        result[index] = active
    return result


def audit_target_contact_semantics(
    model: mujoco.MjModel,
    target: MotionTarget,
    *,
    config: RolloutQCConfig,
    contact_contract: FootFloorContactContract | None = None,
) -> TargetContactAudit:
    """Replay target states through ``mj_forward`` and audit exact + proxy contact.

    Exact target contact is recorded even when a dynamic reference has no
    physically resolved supporting force.  P05/P06/P07/P11/P12 may pass target
    preflight on the strict ankle/toe-site hysteresis proxy, but the report then
    marks ``target_exact_contact_incomplete=true``.  Actual rollout success
    never uses this fallback.  P01 stable stance always requires exact contact.
    """

    target = target.validated(model)
    config = config.validated()
    toy = target.task_id == "P00_synthetic_fixture"
    family = target.task_id.split("_", 1)[0]
    contract = contact_contract or resolve_foot_floor_contact_contract(
        model,
        allow_unavailable=toy,
    )
    transition_count = target.transition_count
    exact_left_states = np.zeros((transition_count + 1,), dtype=np.bool_)
    exact_right_states = np.zeros((transition_count + 1,), dtype=np.bool_)
    left_force_states = np.zeros((transition_count + 1,), dtype=np.float64)
    right_force_states = np.zeros((transition_count + 1,), dtype=np.float64)
    root_position_states = np.zeros((transition_count + 1,), dtype=np.float64)
    com_position_states = np.zeros((transition_count + 1,), dtype=np.float64)

    if toy:
        try:
            root_qpos_z, root_qvel_z, root_body = _root_kinematic_binding(model)
        except ValueError:
            root_qpos_z = root_qvel_z = root_body = None
    else:
        root_qpos_z, root_qvel_z, root_body = _root_kinematic_binding(model)

    site_names = {
        "left_ankle": "left_ankle_mimic",
        "left_toes": "left_toes_mimic",
        "right_ankle": "right_ankle_mimic",
        "right_toes": "right_toes_mimic",
    }
    site_ids: dict[str, int] = {}
    missing_sites: list[str] = []
    for label, name in site_names.items():
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
        if site_id < 0:
            missing_sites.append(name)
        else:
            site_ids[label] = site_id
    if missing_sites and not toy:
        raise ValueError(f"production target proxy is missing stable ankle/toe sites: {sorted(missing_sites)}")
    site_z = {label: np.zeros((transition_count + 1,), dtype=np.float64) for label in site_ids}
    data = mujoco.MjData(model)
    for state_index in range(transition_count + 1):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = target.qpos[state_index]
        data.qvel[:] = target.qvel[state_index]
        mujoco.mj_forward(model, data)
        left, right, left_force, right_force = _foot_floor_contact_sample(
            model,
            data,
            contract=contract,
            min_normal_force=config.min_contact_normal_force,
        )
        exact_left_states[state_index] = left
        exact_right_states[state_index] = right
        left_force_states[state_index] = left_force
        right_force_states[state_index] = right_force
        if root_qpos_z is not None and root_body is not None:
            root_position_states[state_index] = float(data.qpos[root_qpos_z])
            com_position_states[state_index] = float(data.subtree_com[root_body, 2])
        for label, site_id in site_ids.items():
            site_z[label][state_index] = float(data.site_xpos[site_id, 2])

    duration = target.transition_substeps.astype(np.float64) * float(model.opt.timestep)
    if root_qvel_z is None:
        root_velocity = np.zeros((transition_count,), dtype=np.float64)
    else:
        root_velocity = np.asarray(target.qvel[1:, root_qvel_z], dtype=np.float64).copy()
    com_velocity = np.diff(com_position_states) / duration
    if len(site_ids) == len(site_names):
        baselines = {
            label: float(np.quantile(values, config.site_contact_baseline_quantile)) for label, values in site_z.items()
        }
        left_clearance_states = np.minimum(
            site_z["left_ankle"] - baselines["left_ankle"],
            site_z["left_toes"] - baselines["left_toes"],
        )
        right_clearance_states = np.minimum(
            site_z["right_ankle"] - baselines["right_ankle"],
            site_z["right_toes"] - baselines["right_toes"],
        )
        proxy_left_states = _site_contact_hysteresis(left_clearance_states, config)
        proxy_right_states = _site_contact_hysteresis(right_clearance_states, config)
    else:
        baselines = {}
        left_clearance_states = np.full((transition_count + 1,), np.inf, dtype=np.float64)
        right_clearance_states = np.full((transition_count + 1,), np.inf, dtype=np.float64)
        proxy_left_states = np.zeros((transition_count + 1,), dtype=np.bool_)
        proxy_right_states = np.zeros((transition_count + 1,), dtype=np.bool_)

    vertical_signal, semantic_position, semantic_velocity = _vertical_signal_for_task(
        target.task_id,
        root_position=root_position_states[1:],
        root_velocity=root_velocity,
        com_position=com_position_states[1:],
        com_velocity=com_velocity,
    )
    axial_rotation = reconstruct_axial_rotation_evidence(
        model,
        initial_qpos=target.qpos[0],
        post_transition_qpos=target.qpos[1:],
        transition_substeps=target.transition_substeps,
        allow_unavailable=family != "P08",
    )
    exact_semantics = evaluate_task_contact_semantics(
        task_id=target.task_id,
        phase_id=target.phase_id,
        left_contact=exact_left_states[1:],
        right_contact=exact_right_states[1:],
        left_normal_force=left_force_states[1:],
        right_normal_force=right_force_states[1:],
        vertical_position=semantic_position,
        vertical_velocity=semantic_velocity,
        config=config,
        evidence_kind="target_mj_forward_exact_contact",
        vertical_signal=vertical_signal,
        **_axial_semantic_arguments(axial_rotation),
    )
    proxy_left = proxy_left_states[1:]
    proxy_right = proxy_right_states[1:]
    proxy_semantics = evaluate_task_contact_semantics(
        task_id=target.task_id,
        phase_id=target.phase_id,
        left_contact=proxy_left,
        right_contact=proxy_right,
        left_normal_force=proxy_left.astype(np.float64) * config.min_contact_normal_force,
        right_normal_force=proxy_right.astype(np.float64) * config.min_contact_normal_force,
        vertical_position=semantic_position,
        vertical_velocity=semantic_velocity,
        config=config,
        evidence_kind="target_site_xpos_hysteresis_proxy",
        vertical_signal=vertical_signal,
        **_axial_semantic_arguments(axial_rotation),
    )
    proxy_fallback_allowed = family in {"P05", "P06", "P07", "P11", "P12"}
    exact_passed = bool(exact_semantics["passed"])
    proxy_passed = bool(proxy_semantics["passed"])
    passed = bool(exact_passed or (proxy_fallback_allowed and proxy_passed))
    semantics = {
        "passed": passed,
        "task_id": target.task_id,
        "task_family": family,
        "vertical_signal": vertical_signal,
        "gate_basis": (
            "exact_mj_forward_contact"
            if exact_passed
            else "site_xpos_hysteresis_proxy"
            if proxy_fallback_allowed and proxy_passed
            else "none_failed_closed"
        ),
        "proxy_fallback_allowed": proxy_fallback_allowed,
        "target_exact_contact_incomplete": bool(not exact_passed and proxy_passed),
        "initial_state_contact": {
            "exact_left": bool(exact_left_states[0]),
            "exact_right": bool(exact_right_states[0]),
            "exact_left_normal_force": float(left_force_states[0]),
            "exact_right_normal_force": float(right_force_states[0]),
            "site_proxy_left": bool(proxy_left_states[0]),
            "site_proxy_right": bool(proxy_right_states[0]),
        },
        "exact": exact_semantics,
        "site_proxy": {
            **proxy_semantics,
            "site_names": site_names,
            "site_baselines_z": baselines,
            "enter_height_above_baseline": config.site_contact_enter_height,
            "exit_height_above_baseline": config.site_contact_exit_height,
        },
        "contact_contract": contract.as_dict(),
    }
    return TargetContactAudit(
        left_foot_floor_contact=exact_left_states[1:].copy(),
        right_foot_floor_contact=exact_right_states[1:].copy(),
        left_foot_floor_normal_force=left_force_states[1:].copy(),
        right_foot_floor_normal_force=right_force_states[1:].copy(),
        initial_left_foot_floor_contact=bool(exact_left_states[0]),
        initial_right_foot_floor_contact=bool(exact_right_states[0]),
        initial_left_foot_floor_normal_force=float(left_force_states[0]),
        initial_right_foot_floor_normal_force=float(right_force_states[0]),
        site_proxy_left_foot_contact=proxy_left.copy(),
        site_proxy_right_foot_contact=proxy_right.copy(),
        initial_site_proxy_left_foot_contact=bool(proxy_left_states[0]),
        initial_site_proxy_right_foot_contact=bool(proxy_right_states[0]),
        site_proxy_left_clearance=left_clearance_states[1:].copy(),
        site_proxy_right_clearance=right_clearance_states[1:].copy(),
        root_vertical_position=root_position_states[1:].copy(),
        root_vertical_velocity=root_velocity,
        com_vertical_position=com_position_states[1:].copy(),
        com_vertical_velocity=com_velocity,
        axial_rotation=axial_rotation,
        semantics=semantics,
    )


def dense_actuator_moment(data: mujoco.MjData, *, nu: int, nv: int) -> np.ndarray:
    """Decode MuJoCo's sparse actuator moment rows into ``[nu,nv]``.

    ``data.actuator_moment`` is a sparse value buffer.  Reshaping that buffer is
    wrong even when its allocated length happens to equal ``nu * nv``; row
    addresses, non-zero counts, and column indices define the actual matrix.
    """

    if type(nu) is not int or type(nv) is not int or nu <= 0 or nv <= 0:
        raise ValueError("dense actuator moment dimensions must be positive integers")
    values = np.asarray(data.actuator_moment, dtype=np.float64)
    rowadr = np.asarray(data.moment_rowadr, dtype=np.int64)
    rownnz = np.asarray(data.moment_rownnz, dtype=np.int64)
    colind = np.asarray(data.moment_colind, dtype=np.int64)
    if rowadr.shape != (nu,) or rownnz.shape != (nu,):
        raise ValueError("MuJoCo actuator moment sparse row metadata differs from model.nu")
    dense = np.zeros((nu, nv), dtype=np.float64)
    for row in range(nu):
        start = int(rowadr[row])
        count = int(rownnz[row])
        end = start + count
        if start < 0 or count < 0 or end > values.shape[0] or end > colind.shape[0]:
            raise ValueError("MuJoCo actuator moment sparse row lies outside its value/index buffers")
        columns = colind[start:end]
        if np.any(columns < 0) or np.any(columns >= nv) or len(set(columns.tolist())) != count:
            raise ValueError("MuJoCo actuator moment sparse row contains invalid/duplicate columns")
        dense[row, columns] = values[start:end]
    return dense


def build_myofullbody_354_runtime_model() -> mujoco.MjModel:
    """Construct a diagnostic no-finger asset model.

    This helper is retained for unit tests and asset diagnostics only.  It is
    *not* the production ChinaJump runtime: TaskFactory configuration mutates
    the complete model and therefore produces a different model hash.  The CLI
    uses :func:`resolve_chinajump_runtime_model`, which either constructs the
    TaskFactory model or strictly reloads its verified MJB.
    """

    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    # Match the production environment constructor exactly.  Even a visual-only
    # mutation such as ``no_skybox=True`` would change the complete MJB/model
    # hash and therefore is not appropriate for an exact runtime artifact.
    environment = MyoFullBody(disable_fingers=True)
    if getattr(environment, "_disable_fingers", None) is not True:
        raise ValueError("runtime environment did not retain disable_fingers=True")
    model = environment._model
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("MyoFullBody runtime did not expose a CPU mujoco.MjModel")
    names = complete_actuator_names(model)
    if int(model.nu) != 354 or len(names) != 354:
        raise ValueError(
            "primitive producer requires exact MyoFullBody(disable_fingers=True) width 354; "
            f"got model.nu={int(model.nu)}"
        )
    resolve_muscle_channel_contract(model, names)
    validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    return model


def _compose_chinajump_runtime_config(
    *,
    config_name: str,
    hydra_overrides: Sequence[str],
) -> _ComposedChinaJumpRuntimeConfig:
    """Compose and resolve the runtime identity without constructing an env."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    name = str(config_name).strip().removesuffix(".yaml")
    if not name or name.startswith("/") or ".." in PurePosixPath(name).parts:
        raise ValueError("config_name must be a non-empty fullbody-relative Hydra config")
    overrides = tuple(str(value) for value in hydra_overrides)
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(config)
    resolved_config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved_config, dict):
        raise ValueError("resolved Hydra experiment config is not a mapping")
    experiment = config.experiment
    if str(experiment.get("training_action", "")).casefold() != "chinajump":
        raise ValueError("production primitive runtime config must declare experiment.training_action=ChinaJump")
    if bool(experiment.env_params.get("disable_fingers", False)) is not True:
        raise ValueError("production primitive runtime config must declare disable_fingers=true")
    declared_num_envs = int(experiment.env_params.get("num_envs", experiment.get("num_envs", 0)))
    if declared_num_envs <= 0:
        raise ValueError("resolved production config has no positive environment num_envs")
    return _ComposedChinaJumpRuntimeConfig(
        name=name,
        hydra_overrides=overrides,
        config=config,
        resolved_config=resolved_config,
        declared_production_num_envs=declared_num_envs,
    )


def _resolved_runtime_model_params(
    composed: _ComposedChinaJumpRuntimeConfig,
    *,
    construction_num_envs: int,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if type(construction_num_envs) is not int or construction_num_envs <= 0:
        raise ValueError("construction_num_envs must be a positive integer")
    experiment = composed.config.experiment
    env_params = OmegaConf.to_container(experiment.env_params, resolve=True)
    task_params = OmegaConf.to_container(experiment.task_factory.params, resolve=True)
    if not isinstance(env_params, dict) or not isinstance(task_params, dict):
        raise ValueError("resolved TaskFactory model parameters must be mappings")
    env_params["num_envs"] = int(construction_num_envs)
    return {
        "env_params": env_params,
        "task_factory": {
            "name": str(experiment.task_factory.name),
            "params": task_params,
        },
    }


def build_chinajump_taskfactory_runtime_model(
    *,
    config_name: str,
    hydra_overrides: Sequence[str] = (),
    construction_num_envs: int = 1,
    verify_num_env_model_invariance: bool = True,
) -> TaskFactoryRuntimeModel:
    """Compose a ChinaJump config and construct its exact TaskFactory model.

    ``num_envs=1`` keeps this CPU-side producer small.  It is allowed only after
    constructing the declared production-width environment as well and proving
    that both complete ``MjModel.__getstate__`` hashes are identical.
    """

    from loco_mujoco.task_factories import TaskFactory

    if type(construction_num_envs) is not int or construction_num_envs <= 0:
        raise ValueError("construction_num_envs must be a positive integer")
    composed = _compose_chinajump_runtime_config(
        config_name=config_name,
        hydra_overrides=hydra_overrides,
    )
    experiment = composed.config.experiment
    original_num_envs = composed.declared_production_num_envs

    def construct(num_envs: int) -> mujoco.MjModel:
        model_params = _resolved_runtime_model_params(composed, construction_num_envs=num_envs)
        env_params = dict(model_params["env_params"])
        task_factory = model_params["task_factory"]
        if not isinstance(task_factory, Mapping) or not isinstance(task_factory.get("params"), Mapping):
            raise ValueError("resolved TaskFactory identity is malformed")
        factory = TaskFactory.get_factory_cls(str(experiment.task_factory.name))
        environment = factory.make(**env_params, **dict(task_factory["params"]))
        model = getattr(environment, "_model", None)
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("ChinaJump TaskFactory did not expose a CPU mujoco.MjModel")
        return model

    model = construct(construction_num_envs)
    names = complete_actuator_names(model)
    if int(model.nu) != 354 or len(names) != 354:
        raise ValueError(f"ChinaJump TaskFactory model must expose exactly 354 actuators, got {int(model.nu)}")
    resolve_muscle_channel_contract(model, names)
    validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    model_hash = _model_hash(model)
    production_model_hash = model_hash
    invariant = construction_num_envs == original_num_envs
    if verify_num_env_model_invariance and not invariant:
        production_model_hash = _model_hash(construct(original_num_envs))
        invariant = production_model_hash == model_hash
        if not invariant:
            raise ValueError(
                "ChinaJump TaskFactory complete model hash changes with num_envs; "
                "construct the producer with the production num_envs instead"
            )
    elif not verify_num_env_model_invariance and construction_num_envs != original_num_envs:
        raise ValueError("production runtime may not skip num_env model-hash invariance verification")

    model_parameter_payload = _resolved_runtime_model_params(
        composed,
        construction_num_envs=construction_num_envs,
    )
    binding = {
        "schema_version": "chinajump_taskfactory_runtime_model_binding_v1",
        "production_eligible": True,
        "config_name": composed.name,
        "hydra_overrides": list(composed.hydra_overrides),
        "resolved_config_sha256": canonical_json_sha256(composed.resolved_config),
        "resolved_model_params_sha256": canonical_json_sha256(model_parameter_payload),
        "resolved_model_params": model_parameter_payload,
        "declared_production_num_envs": original_num_envs,
        "construction_num_envs": construction_num_envs,
        "num_env_model_hash_invariant": bool(invariant),
        "construction_model_hash": model_hash,
        "declared_num_env_model_hash": production_model_hash,
    }
    provenance = {
        "schema_version": "primitive_runtime_model_provenance_v1",
        "source_kind": "taskfactory_constructed",
        "verified_runtime_artifact": None,
        "model_hash": model_hash,
        "config_name": composed.name,
        "hydra_overrides": list(composed.hydra_overrides),
    }
    return TaskFactoryRuntimeModel(model=model, binding=binding, provenance=provenance)


def load_verified_runtime_artifact(
    path: str | Path,
    *,
    config_name: str,
    hydra_overrides: Sequence[str] = (),
) -> TaskFactoryRuntimeModel:
    """Load a previously verified runtime MJB without constructing TaskFactory.

    This is a CPU-only reuse path, not a weaker runtime path.  The controller
    artifact, MJB, complete model ABI, immutable runtime binding, and the
    *current* Hydra composition must all reproduce the stored identities.
    """

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_dir() or not _is_sha256(artifact.name):
        raise ValueError("verified runtime artifact must be a fingerprint-named controller directory")
    manifest = _load_optimizer_artifact(artifact, expected_fingerprint=artifact.name)
    if manifest.get("schema_version") != OPTIMIZER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("verified runtime artifact has an unsupported optimizer manifest schema")
    if manifest.get("production_eligible") is not True:
        raise ValueError("verified runtime artifact is not production eligible")
    model_path = artifact / "runtime_model.mjb"
    try:
        model = mujoco.MjModel.from_binary_path(str(model_path))
    except Exception as error:
        raise ValueError("verified runtime artifact MJB cannot be loaded") from error
    model_hash = _model_hash(model)
    if manifest.get("model_hash") != model_hash:
        raise ValueError("verified runtime artifact complete model hash mismatch")
    if (
        int(model.nu) != 354
        or int(model.na) != 354
        or manifest.get("model_nu") != 354
        or manifest.get("model_na") != 354
        or manifest.get("model_nq") != int(model.nq)
        or manifest.get("model_nv") != int(model.nv)
    ):
        raise ValueError("verified runtime artifact must preserve the complete nq/nv/nu=na=354 model ABI")

    names = complete_actuator_names(model)
    if tuple(manifest.get("actuator_names", ())) != names:
        raise ValueError("verified runtime artifact actuator order differs from the loaded MJB")
    if manifest.get("actuator_schema_hash") != actuator_schema_hash(names):
        raise ValueError("verified runtime artifact actuator schema hash mismatch")
    channel_contract = resolve_muscle_channel_contract(model, names)
    if len(channel_contract.actuator_actadr) != 354:
        raise ValueError("verified runtime artifact does not expose 354 physical activation channels")
    ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    recorded_ctrlrange = np.asarray(manifest.get("ctrlrange"), dtype=np.float64)
    if recorded_ctrlrange.shape != (354, 2) or not np.array_equal(recorded_ctrlrange, ctrlrange):
        raise ValueError("verified runtime artifact ctrlrange differs from the loaded MJB")
    expected_ordered_ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={"actuator_names": list(names), "ctrlrange": ctrlrange.tolist()},
    )
    if manifest.get("ctrlrange_schema_hash") != expected_ordered_ctrlrange_hash:
        raise ValueError("verified runtime artifact ordered ctrlrange schema hash mismatch")
    if manifest.get("transform_ctrlrange_schema_hash") != ctrlrange_schema_hash(names, ctrlrange):
        raise ValueError("verified runtime artifact transform ctrlrange schema hash mismatch")

    raw_binding = manifest.get("runtime_model_binding")
    if not isinstance(raw_binding, Mapping):
        raise ValueError("verified runtime artifact has no production runtime binding")
    binding = _validate_runtime_model_binding(raw_binding, model_hash=model_hash)
    if binding is None:
        raise ValueError("verified runtime artifact has no validated runtime binding")
    supplied_name = str(config_name).strip().removesuffix(".yaml")
    supplied_overrides = tuple(str(value) for value in hydra_overrides)
    if supplied_name != binding.get("config_name"):
        raise ValueError("CLI config_name differs from the verified runtime binding")
    bound_overrides = binding.get("hydra_overrides")
    if not isinstance(bound_overrides, list) or supplied_overrides != tuple(str(value) for value in bound_overrides):
        raise ValueError("CLI hydra_overrides differ from the verified runtime binding")

    composed = _compose_chinajump_runtime_config(
        config_name=supplied_name,
        hydra_overrides=supplied_overrides,
    )
    current_config_hash = canonical_json_sha256(composed.resolved_config)
    if current_config_hash != binding.get("resolved_config_sha256"):
        raise ValueError("current resolved Hydra config differs from the verified runtime binding")
    construction_num_envs = binding.get("construction_num_envs")
    if type(construction_num_envs) is not int or construction_num_envs <= 0:
        raise ValueError("verified runtime binding construction_num_envs is invalid")
    if binding.get("declared_production_num_envs") != composed.declared_production_num_envs:
        raise ValueError("current declared production num_envs differs from the verified runtime binding")
    current_model_params = _resolved_runtime_model_params(
        composed,
        construction_num_envs=construction_num_envs,
    )
    current_model_params_hash = canonical_json_sha256(current_model_params)
    if current_model_params != binding.get("resolved_model_params") or current_model_params_hash != binding.get(
        "resolved_model_params_sha256"
    ):
        raise ValueError("current resolved model params differ from the verified runtime binding")

    provenance = {
        "schema_version": "primitive_runtime_model_provenance_v1",
        "source_kind": "verified_runtime_artifact_reuse",
        "verified_runtime_artifact": {
            "path": str(artifact),
            "optimizer_fingerprint": artifact.name,
            "optimizer_manifest_sha256": file_sha256(artifact / "optimizer_manifest.json"),
            "runtime_model_mjb_sha256": file_sha256(model_path),
        },
        "model_hash": model_hash,
        "config_name": composed.name,
        "hydra_overrides": list(composed.hydra_overrides),
        "current_resolved_config_sha256": current_config_hash,
        "current_resolved_model_params_sha256": current_model_params_hash,
    }
    return TaskFactoryRuntimeModel(model=model, binding=binding, provenance=provenance)


def resolve_chinajump_runtime_model(
    *,
    config_name: str,
    hydra_overrides: Sequence[str] = (),
    verified_runtime_artifact: str | Path | None = None,
) -> TaskFactoryRuntimeModel:
    """Build the default runtime or load an exact, previously verified MJB."""

    if verified_runtime_artifact is None:
        return build_chinajump_taskfactory_runtime_model(
            config_name=config_name,
            hydra_overrides=hydra_overrides,
        )
    return load_verified_runtime_artifact(
        verified_runtime_artifact,
        config_name=config_name,
        hydra_overrides=hydra_overrides,
    )


def load_transition_phase_plan(
    path: str | Path,
    *,
    phase_schema: PrimitivePhaseSchema,
    source_motion_path: str,
    transition_count: int,
    start_frame: int,
    end_frame_exclusive: int,
    source_total_frames: int,
) -> np.ndarray:
    """Load an explicit, gap-free transition phase plan.

    The producer never guesses phases from frame progress.  Segments use
    half-open transition indices and must cover the complete trajectory.
    """

    plan_path = Path(path).expanduser().resolve()
    payload = load_json_strict(plan_path)
    if not isinstance(payload, Mapping):
        raise ValueError("primitive phase plan must contain a JSON object")
    required = {
        "schema_version",
        "task_id",
        "source_motion_path",
        "start_frame",
        "end_frame_exclusive",
        "source_total_frames",
        "transition_count",
        "segments",
    }
    _require_exact_fields(payload, required, "primitive phase plan")
    if payload["schema_version"] != PHASE_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported primitive phase plan schema_version")
    if payload["task_id"] != phase_schema.task_id:
        raise ValueError("primitive phase plan task_id differs from phase schema")
    normalized_source = normalize_relative_motion_path(str(source_motion_path))
    if normalize_relative_motion_path(str(payload["source_motion_path"])) != normalized_source:
        raise ValueError("primitive phase plan source_motion_path differs from requested target")
    interval = (payload["start_frame"], payload["end_frame_exclusive"], payload["source_total_frames"])
    expected_interval = (int(start_frame), int(end_frame_exclusive), int(source_total_frames))
    if any(type(value) is not int for value in interval) or interval != expected_interval:
        raise ValueError("primitive phase plan source frame interval differs from requested crop")
    if type(payload["transition_count"]) is not int or payload["transition_count"] != int(transition_count):
        raise ValueError("primitive phase plan transition_count differs from target trajectory")
    raw_segments = payload["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("primitive phase plan segments must be a non-empty JSON array")
    labels = np.full((int(transition_count),), -1, dtype=np.int32)
    cursor = 0
    observed: list[int] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"primitive phase plan segments[{index}] must be an object")
        _require_exact_fields(
            segment,
            {"phase_id", "start_transition", "end_transition"},
            f"primitive phase plan segments[{index}]",
        )
        phase_id = segment["phase_id"]
        start = segment["start_transition"]
        end = segment["end_transition"]
        if any(type(value) is not int for value in (phase_id, start, end)):
            raise ValueError("primitive phase plan segment values must be integers")
        if start != cursor or end <= start or end > int(transition_count):
            raise ValueError("primitive phase plan segments must be ordered, contiguous, and non-empty")
        if phase_id not in phase_schema.required_phase_ids:
            raise ValueError("primitive phase plan contains an id absent from the task phase schema")
        labels[start:end] = int(phase_id)
        cursor = end
        observed.append(int(phase_id))
    if cursor != int(transition_count) or np.any(labels < 0):
        raise ValueError("primitive phase plan must cover every target transition")
    if set(observed) != set(phase_schema.required_phase_ids):
        raise ValueError("primitive phase plan must exercise every phase required by the task schema")
    return labels


def load_retargeted_motion_target(
    source_npz: str | Path,
    *,
    model: mujoco.MjModel,
    source_motion_path: str,
    phase_schema_path: str | Path,
    phase_plan_path: str | Path,
    target_skill_id: str = "ChinaJump",
    excluded_target_motion_paths: Sequence[str] = (),
    start_frame: int = 0,
    end_frame_exclusive: int | None = None,
) -> MotionTarget:
    """Load qpos/qvel target data without treating it as control evidence."""

    source = Path(source_npz).expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() != ".npz":
        raise FileNotFoundError(f"retargeted source NPZ does not exist: {source}")
    normalized = normalize_relative_motion_path(source_motion_path)
    _assert_not_target_skill_source(normalized, source, target_skill_id=target_skill_id)
    excluded = {normalize_relative_motion_path(path) for path in excluded_target_motion_paths}
    if normalized in excluded or stable_motion_uid(normalized) in {stable_motion_uid(path) for path in excluded}:
        raise ValueError("primitive target is one of the explicitly excluded target-skill motions")
    with np.load(source, allow_pickle=False) as payload:
        required = {"qpos", "qvel", "frequency"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"retargeted target is missing required arrays: {sorted(missing)}")
        complete_qpos = np.asarray(payload["qpos"], dtype=np.float64)
        complete_qvel = np.asarray(payload["qvel"], dtype=np.float64)
        frequency = float(np.asarray(payload["frequency"]).item())
    if complete_qpos.ndim != 2 or complete_qvel.ndim != 2 or complete_qpos.shape[0] != complete_qvel.shape[0]:
        raise ValueError("retargeted target qpos/qvel must be aligned rank-2 arrays")
    total_frames = int(complete_qpos.shape[0])
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer")
    resolved_end = total_frames if end_frame_exclusive is None else end_frame_exclusive
    if (
        type(resolved_end) is not int
        or resolved_end > total_frames
        or resolved_end <= start_frame
        or resolved_end - start_frame < 2
    ):
        raise ValueError("source frame crop must contain at least two frames and lie inside the source trajectory")
    qpos = complete_qpos[start_frame:resolved_end].copy()
    qvel = complete_qvel[start_frame:resolved_end].copy()
    substeps = _constant_transition_substeps(
        source_frequency_hz=frequency,
        physics_timestep=float(model.opt.timestep),
        transition_count=int(qpos.shape[0]) - 1,
    )
    phase_schema = load_primitive_phase_schema(phase_schema_path)
    phase_id = load_transition_phase_plan(
        phase_plan_path,
        phase_schema=phase_schema,
        source_motion_path=normalized,
        transition_count=int(qpos.shape[0]) - 1,
        start_frame=start_frame,
        end_frame_exclusive=resolved_end,
        source_total_frames=total_frames,
    )
    target = MotionTarget(
        qpos=qpos,
        qvel=qvel,
        phase_id=phase_id,
        transition_substeps=substeps,
        source_path=source,
        source_motion_path=normalized,
        source_sha256=file_sha256(source),
        source_frequency_hz=frequency,
        source_start_frame=start_frame,
        source_end_frame_exclusive=resolved_end,
        source_total_frames=total_frames,
        phase_schema_fingerprint=phase_schema.fingerprint,
        task_id=phase_schema.task_id,
    )
    validated = target.validated(model)
    _validate_model_quaternions(model, validated.qpos)
    return validated


def load_full_action_policy_controls(
    physical_rollout_shard: str | Path,
    *,
    metadata_path: str | Path,
    teacher_checkpoint: str | Path,
    model: mujoco.MjModel,
    target: MotionTarget,
    rollout_uid: int | None = None,
) -> PolicyControlImport:
    """Load actual full-354 policy controls captured by ``distill_collect``.

    The normalized policy arrays in the shard are deliberately ignored.  The
    loader accepts only the collector's verified ``teacher_ctrl_physical`` ABI,
    exact actuator metadata, a current checkpoint content hash, contiguous
    source-frame coordinates, and transition phases matching the explicit
    primitive plan.
    """

    if rollout_uid is not None and (type(rollout_uid) is not int or rollout_uid < 0):
        raise ValueError("rollout_uid must be a non-negative integer when supplied")
    target = target.validated(model)
    shard_path = Path(physical_rollout_shard).expanduser().resolve()
    metadata_file = Path(metadata_path).expanduser().resolve()
    if not shard_path.is_file() or shard_path.suffix.casefold() != ".npz":
        raise FileNotFoundError(f"physical policy rollout shard does not exist: {shard_path}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"physical policy rollout metadata does not exist: {metadata_file}")
    metadata = load_json_strict(metadata_file)
    if not isinstance(metadata, Mapping):
        raise ValueError("physical policy rollout metadata must contain a JSON object")
    if metadata.get("collector") != "teacher_lookahead_rollout":
        raise ValueError("physical policy controls require teacher_lookahead_rollout metadata")
    capture = metadata.get("physical_capture")
    if not isinstance(capture, Mapping) or capture.get("schema_version") != "physical_capture_spec_v2":
        raise ValueError("physical policy controls require physical_capture_spec_v2 metadata")
    names = complete_actuator_names(model)
    metadata_names = [str(value) for value in metadata.get("actuator_names", ())]
    capture_names = [str(value) for value in capture.get("actuator_names", ())]
    if metadata_names != list(names) or capture_names != list(names):
        raise ValueError("physical policy rollout actuator names/order differ from the exact runtime model")
    ranges = np.asarray(metadata.get("actuator_ctrlrange"), dtype=np.float64)
    expected_ranges = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    if ranges.shape != expected_ranges.shape or not np.array_equal(ranges, expected_ranges):
        raise ValueError("physical policy rollout ctrlrange differs from the exact runtime model")
    expected_schema = actuator_schema_hash(names)
    if metadata.get("action_schema_hash") != expected_schema:
        raise ValueError("physical policy rollout action schema hash differs from runtime actuator order")

    checkpoint = checkpoint_content_fingerprint(teacher_checkpoint)
    recorded_checkpoint = metadata.get("teacher_checkpoint_content")
    if not isinstance(recorded_checkpoint, Mapping):
        raise ValueError("physical policy rollout metadata lacks checkpoint content inventory")
    for field in ("sha256", "num_files", "num_bytes", "files"):
        if recorded_checkpoint.get(field) != checkpoint.get(field):
            raise ValueError("physical policy rollout checkpoint content differs from the supplied checkpoint")
    if metadata.get("teacher_checkpoint_fingerprint") != checkpoint["sha256"]:
        raise ValueError("physical policy rollout compact checkpoint fingerprint mismatch")
    teacher_target = metadata.get("teacher_action_target")
    if teacher_target == "mean":
        teacher_action_mode = "deterministic_mean"
    elif teacher_target == "sample":
        teacher_action_mode = "stochastic_sample"
    else:
        raise ValueError("physical policy rollout teacher_action_target must be mean or sample")

    with np.load(shard_path, allow_pickle=False) as shard:
        fields = set(shard.files)
        required = {
            "teacher_ctrl_physical",
            "rollout_uid",
            "motion_uid",
            "subtraj_step_no",
            "done",
            "sim_pre_qpos",
            "sim_pre_qvel",
        }
        missing = required - fields
        if missing:
            normalized_only = "teacher_action" in fields and "teacher_ctrl_physical" not in fields
            if normalized_only:
                raise ValueError(
                    "policy shard contains only normalized/signed action; actual teacher_ctrl_physical is required"
                )
            raise ValueError(f"physical policy rollout shard is missing fields: {sorted(missing)}")
        all_rollout_uid = np.asarray(shard["rollout_uid"], dtype=np.int64)
        if rollout_uid is None:
            unique_rollouts = np.unique(all_rollout_uid)
            if unique_rollouts.shape != (1,):
                raise ValueError("physical rollout shard has multiple rollout_uid values; select one explicitly")
            selected_rollout_uid = int(unique_rollouts[0])
        else:
            selected_rollout_uid = int(rollout_uid)
        row_mask = all_rollout_uid == selected_rollout_uid
        if not np.any(row_mask):
            raise ValueError("selected rollout_uid is absent from the physical rollout shard")
        step_no = np.asarray(shard["subtraj_step_no"], dtype=np.int64)[row_mask]
        expected_steps = np.arange(
            target.source_start_frame,
            target.source_end_frame_exclusive - 1,
            dtype=np.int64,
        )
        selected_rows = np.flatnonzero(row_mask)
        step_mask = np.isin(step_no, expected_steps)
        selected_rows = selected_rows[step_mask]
        selected_steps = step_no[step_mask]
        if selected_rows.shape != expected_steps.shape:
            raise ValueError("selected policy rollout does not cover the exact primitive source-frame interval")
        order = np.argsort(selected_steps, kind="stable")
        selected_rows = selected_rows[order]
        selected_steps = selected_steps[order]
        if not np.array_equal(selected_steps, expected_steps):
            raise ValueError("selected policy rollout source-frame coordinates are duplicated or non-contiguous")
        controls = np.asarray(shard["teacher_ctrl_physical"], dtype=np.float64)[selected_rows]
        motion_uid = np.asarray(shard["motion_uid"], dtype=np.int64)[selected_rows]
        phase_id = None if "phase_id" not in fields else np.asarray(shard["phase_id"])[selected_rows]
        done = np.asarray(shard["done"], dtype=bool)[selected_rows]
        pre_qpos = np.asarray(shard["sim_pre_qpos"], dtype=np.float64)[selected_rows]
        pre_qvel = np.asarray(shard["sim_pre_qvel"], dtype=np.float64)[selected_rows]
    expected_motion_uid = stable_motion_uid(target.source_motion_path)
    if np.any(motion_uid != expected_motion_uid):
        raise ValueError("selected policy controls belong to a different stable source motion")
    collector_phase_id_match = phase_id is not None
    if phase_id is not None and (
        phase_id.shape != target.phase_id.shape
        or not np.issubdtype(phase_id.dtype, np.integer)
        or not np.array_equal(phase_id.astype(np.int32), target.phase_id)
    ):
        raise ValueError("policy rollout phase_id differs from the explicit primitive transition plan")
    if np.any(done[:-1]):
        raise ValueError("selected policy rollout terminated before the primitive interval ended")
    if pre_qpos.shape != (target.transition_count, int(model.nq)) or pre_qvel.shape != (
        target.transition_count,
        int(model.nv),
    ):
        raise ValueError("policy pre-transition state dimensions differ from the exact runtime model")
    pre_position_error = np.asarray(
        [
            _position_difference(model, observed, expected)
            for observed, expected in zip(pre_qpos, target.qpos[:-1], strict=True)
        ]
    )
    pre_velocity_error = pre_qvel - target.qvel[:-1]
    control_hash = _array_sha256(controls)
    portable_checkpoint_content = {
        "schema_version": checkpoint["schema_version"],
        "sha256": checkpoint["sha256"],
        "num_files": checkpoint["num_files"],
        "num_bytes": checkpoint["num_bytes"],
        "files": checkpoint["files"],
    }
    controller_binding = {
        "schema_version": "full_354_policy_controller_binding_v1",
        # Deliberately omit supplied/resolved workstation paths: the controller
        # address depends on the checkpoint directory's bytes and inventory.
        "checkpoint_content": portable_checkpoint_content,
        "checkpoint_sha256": checkpoint["sha256"],
        "teacher_action_mode": teacher_action_mode,
    }
    rollout_binding = {
        "schema_version": "full_354_policy_physical_rollout_binding_v1",
        "physical_rollout_shard_sha256": file_sha256(shard_path),
        "physical_rollout_metadata_sha256": file_sha256(metadata_file),
        "physical_control_sequence_sha256": control_hash,
        "selected_rollout_uid": selected_rollout_uid,
        "source_motion_uid": int(expected_motion_uid),
        "source_frame_interval": {
            "start_frame": target.source_start_frame,
            "end_frame_exclusive": target.source_end_frame_exclusive,
            "source_total_frames": target.source_total_frames,
        },
        "transition_count": target.transition_count,
        "collector_phase_id_match": collector_phase_id_match,
        "phase_schema_fingerprint": target.phase_schema_fingerprint,
        "policy_pre_state_tracking": {
            "position_rmse": float(np.sqrt(np.mean(np.square(pre_position_error)))),
            "velocity_rmse": float(np.sqrt(np.mean(np.square(pre_velocity_error)))),
            "position_abs_max": float(np.max(np.abs(pre_position_error))),
            "velocity_abs_max": float(np.max(np.abs(pre_velocity_error))),
        },
    }
    return PolicyControlImport(
        planner=ScriptedPhysicalControlPlanner(model, controls),
        controller_binding=controller_binding,
        rollout_binding=rollout_binding,
    )


def build_synthetic_motion_target(
    model: mujoco.MjModel,
    *,
    applied_ctrl: Any,
    phase_id: Any,
    transition_substeps: Any,
    task_id: str = "P00_synthetic_fixture",
    source_motion_path: str = "synthetic/primitive-fixture",
    initial_qpos: Any | None = None,
    initial_qvel: Any | None = None,
) -> MotionTarget:
    """Build a declared synthetic target by forward-simulating physical ctrl.

    This helper is intended for toy validation and explicitly labelled
    synthetic primitive experiments.  The supplied controls generate only the
    target trajectory; a producer rollout must independently optimize and
    reapply its own physical controls before it can become evidence.
    """

    names = complete_actuator_names(model)
    resolve_muscle_channel_contract(model, names)
    ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    ctrl = np.asarray(applied_ctrl, dtype=np.float64)
    phases = np.asarray(phase_id)
    substeps = np.asarray(transition_substeps)
    if ctrl.ndim != 2 or ctrl.shape[0] <= 0 or ctrl.shape[1] != int(model.nu):
        raise ValueError(f"synthetic physical ctrl must have shape [T,{int(model.nu)}]")
    if np.any(ctrl < ctrlrange[:, 0]) or np.any(ctrl > ctrlrange[:, 1]) or not np.all(np.isfinite(ctrl)):
        raise ValueError("synthetic physical ctrl is outside the model ctrlrange")
    if phases.shape != (ctrl.shape[0],) or not np.issubdtype(phases.dtype, np.integer):
        raise ValueError("synthetic phase_id must be an integer vector with shape [T]")
    if substeps.shape != (ctrl.shape[0],) or not np.issubdtype(substeps.dtype, np.integer) or np.any(substeps <= 0):
        raise ValueError("synthetic transition_substeps must be positive integers with shape [T]")
    data = mujoco.MjData(model)
    if initial_qpos is not None:
        qpos = np.asarray(initial_qpos, dtype=np.float64)
        if qpos.shape != (int(model.nq),):
            raise ValueError("synthetic initial_qpos has wrong model width")
        data.qpos[:] = qpos
    if initial_qvel is not None:
        qvel = np.asarray(initial_qvel, dtype=np.float64)
        if qvel.shape != (int(model.nv),):
            raise ValueError("synthetic initial_qvel has wrong model width")
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    qpos_rows = [np.asarray(data.qpos, dtype=np.float64).copy()]
    qvel_rows = [np.asarray(data.qvel, dtype=np.float64).copy()]
    for transition_ctrl, count in zip(ctrl, substeps, strict=True):
        data.ctrl[:] = transition_ctrl
        for _ in range(int(count)):
            mujoco.mj_step(model, data)
        qpos_rows.append(np.asarray(data.qpos, dtype=np.float64).copy())
        qvel_rows.append(np.asarray(data.qvel, dtype=np.float64).copy())
    digest = hashlib.sha256()
    digest.update(np.asarray(ctrl, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(phases, dtype="<i4").tobytes(order="C"))
    digest.update(np.asarray(substeps, dtype="<i4").tobytes(order="C"))
    phase_fingerprint = canonical_json_sha256(
        {"task_id": str(task_id), "phase_ids": sorted({int(value) for value in phases.tolist()})}
    )
    return MotionTarget(
        qpos=np.asarray(qpos_rows),
        qvel=np.asarray(qvel_rows),
        phase_id=phases.astype(np.int32),
        transition_substeps=substeps.astype(np.int32),
        source_path=None,
        source_motion_path=source_motion_path,
        source_sha256=digest.hexdigest(),
        source_frequency_hz=1.0 / (float(model.opt.timestep) * float(np.median(substeps))),
        source_start_frame=0,
        source_end_frame_exclusive=int(ctrl.shape[0]) + 1,
        source_total_frames=int(ctrl.shape[0]) + 1,
        phase_schema_fingerprint=phase_fingerprint,
        task_id=task_id,
    ).validated(model)


def _validate_canonical_controller_copy(path: Path, canonical_binding: Mapping[str, Any]) -> None:
    embedded = load_canonical_control_artifact(
        path / "canonical_control.json",
        expected_width=int(canonical_binding["action_dim"]),
        require_path_binding=False,
    )
    portable = {key: value for key, value in embedded.items() if key != "path"}
    if portable != dict(canonical_binding):
        raise ValueError("controller canonical control copy differs from optimizer binding")


def ensure_optimizer_artifact(
    model: mujoco.MjModel,
    *,
    controller_store: str | Path,
    config: PhysicalOptimizerConfig,
    controller_binding: Mapping[str, Any] | None = None,
    canonical_control_binding: Mapping[str, Any] | None = None,
    runtime_model_binding: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Save an exact MJB and content-addressed optimizer manifest."""

    config = config.validated()
    names = complete_actuator_names(model)
    resolve_muscle_channel_contract(model, names)
    ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    model_state = model.__getstate__()
    if not isinstance(model_state, bytes) or not model_state:
        raise ValueError("MuJoCo model has no canonical byte state")
    model_hash = hashlib.sha256(model_state).hexdigest()
    runtime_binding = _validate_runtime_model_binding(runtime_model_binding, model_hash=model_hash)
    store = Path(controller_store).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = store / f".tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model_path = save_compiled_model_artifact(model, temporary / "runtime_model.mjb")
        policy_binding = None if controller_binding is None else _validate_policy_controller_binding(controller_binding)
        if policy_binding is not None and canonical_control_binding is not None:
            raise ValueError("canonical tonic control and policy control are mutually exclusive")
        canonical_binding = None
        if canonical_control_binding is not None:
            reloaded = load_canonical_control_artifact(canonical_control_binding["path"], expected_width=int(model.nu))
            if dict(reloaded) != dict(canonical_control_binding):
                raise ValueError("canonical control binding differs from its path-bound artifact")
            canonical_binding = {key: value for key, value in dict(canonical_control_binding).items() if key != "path"}
            if canonical_binding.get("model_hash") != model_hash or canonical_binding.get("action_dim") != int(
                model.nu
            ):
                raise ValueError("canonical control artifact differs from runtime model")
            if canonical_binding.get("actuator_schema_hash") != actuator_schema_hash(names):
                raise ValueError("canonical control actuator ABI differs from runtime model")
            expected_ctrlrange_hash = ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={"actuator_names": list(names), "ctrlrange": ctrlrange.tolist()},
            )
            if canonical_binding.get("ctrlrange_schema_hash") != expected_ctrlrange_hash:
                raise ValueError("canonical control ctrlrange ABI differs from runtime model")
            shutil.copy2(str(canonical_control_binding["path"]), temporary / "canonical_control.json")
        source_kind = (
            "canonical_tonic_control"
            if canonical_binding is not None
            else ("trajectory_optimizer" if policy_binding is None else "full_action_teacher")
        )
        algorithm = (
            "train_only_canonical_tonic_hold_v1"
            if canonical_binding is not None
            else (OPTIMIZER_ALGORITHM if policy_binding is None else "full_354_policy_actual_ctrl_cpu_replay_v1")
        )
        initial_state_contract = (
            _CANONICAL_TONIC_INITIAL_STATE_CONTRACT
            if canonical_binding is not None
            else (_INFERRED_INITIAL_STATE_CONTRACT if policy_binding is None else _ZERO_INITIAL_STATE_CONTRACT)
        )
        payload: dict[str, Any] = {
            "schema_version": OPTIMIZER_MANIFEST_SCHEMA_VERSION,
            "source_kind": source_kind,
            "algorithm": algorithm,
            "initial_state_contract": initial_state_contract,
            "control_coordinates": "exact_transition_state.data.ctrl_model_units",
            "normalized_action_accepted": False,
            "emg_used": False,
            "success_decision": "closed_loop_tracking_and_forward_replay_qc_only",
            "shooting_proposal_residual_can_mark_success": False,
            "optimizer_config": (
                {"canonical_control_binding": canonical_binding, "canonical_control_filename": "canonical_control.json"}
                if canonical_binding is not None
                else (asdict(config) if policy_binding is None else None)
            ),
            "policy_controller_binding": policy_binding,
            "production_eligible": runtime_binding is not None,
            "runtime_model_binding": runtime_binding,
            "implementation_sha256": file_sha256(Path(__file__).resolve()),
            "model_hash": model_hash,
            "model_artifact_filename": model_path.name,
            "model_artifact_sha256": file_sha256(model_path),
            "model_nq": int(model.nq),
            "model_nv": int(model.nv),
            "model_nu": int(model.nu),
            "model_na": int(model.na),
            "physics_timestep": float(model.opt.timestep),
            "actuator_names": list(names),
            "actuator_schema_hash": actuator_schema_hash(names),
            "ctrlrange": np.asarray(ctrlrange, dtype=np.float64).tolist(),
            "ctrlrange_schema_hash": ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={"actuator_names": list(names), "ctrlrange": ctrlrange.tolist()},
            ),
            "transform_ctrlrange_schema_hash": ctrlrange_schema_hash(names, ctrlrange),
        }
        fingerprint = canonical_json_sha256(payload)
        payload["optimizer_fingerprint"] = fingerprint
        _write_json(temporary / "optimizer_manifest.json", payload)
        final = store / fingerprint
        if final.exists():
            existing = _load_optimizer_artifact(final, expected_fingerprint=fingerprint)
            if canonical_binding is not None:
                _validate_canonical_controller_copy(final, canonical_binding)
            return final, existing
        _publish_content_addressed_directory(
            temporary,
            final,
            lambda path: _load_optimizer_artifact(path, expected_fingerprint=fingerprint),
        )
        temporary = None
        existing = _load_optimizer_artifact(final, expected_fingerprint=fingerprint)
        if canonical_binding is not None:
            _validate_canonical_controller_copy(final, canonical_binding)
        return final, existing
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def produce_primitive_trial(
    model: mujoco.MjModel,
    target: MotionTarget,
    *,
    output_dir: str | Path,
    controller_store: str | Path,
    trial_id: str,
    optimizer_config: PhysicalOptimizerConfig,
    qc_config: RolloutQCConfig,
    seed: int,
    planner: PhysicalControlPlanner | None = None,
    controller_binding: Mapping[str, Any] | None = None,
    runtime_model_binding: Mapping[str, Any] | None = None,
    runtime_model_provenance: Mapping[str, Any] | None = None,
    policy_rollout_binding: Mapping[str, Any] | None = None,
    canonical_control_binding: Mapping[str, Any] | None = None,
) -> ProducerResult:
    """Optimize, roll out, replay, QC, and atomically publish one trial."""

    if not str(trial_id).strip():
        raise ValueError("trial_id must be non-empty")
    if type(seed) is not int or seed < 0:
        raise ValueError("optimizer seed must be a non-negative integer")
    if canonical_control_binding is not None and planner is not None:
        raise ValueError("canonical control binding and explicit planner are mutually exclusive")
    target = target.validated(model)
    if target.source_path is not None and runtime_model_binding is None:
        raise ValueError(
            "file-backed primitive production requires an exact ChinaJump TaskFactory runtime binding; "
            "the direct MyoFullBody diagnostic model is not production eligible"
        )
    optimizer_config = optimizer_config.validated()
    qc_config = qc_config.validated()
    names = complete_actuator_names(model)
    channel_contract = resolve_muscle_channel_contract(model, names)
    ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    contact_contract = resolve_foot_floor_contact_contract(
        model,
        allow_unavailable=target.task_id == "P00_synthetic_fixture",
    )
    target_contact_audit = audit_target_contact_semantics(
        model,
        target,
        config=qc_config,
        contact_contract=contact_contract,
    )
    controller_dir, controller_manifest = ensure_optimizer_artifact(
        model,
        controller_store=controller_store,
        config=optimizer_config,
        controller_binding=controller_binding,
        canonical_control_binding=canonical_control_binding,
        runtime_model_binding=runtime_model_binding,
    )
    validated_runtime_provenance = _validate_runtime_model_provenance(
        runtime_model_provenance,
        model_hash=_model_hash(model),
        runtime_model_binding=controller_manifest["runtime_model_binding"],
    )
    selected_planner = (
        planner
        if planner is not None
        else (
            CanonicalTonicControlPlanner(model, canonical_control_binding["control"])
            if canonical_control_binding is not None
            else ComputedMuscleCEMPlanner(model, optimizer_config)
        )
    )
    selected_planner.reset(seed)

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"primitive rollout output already exists; refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        rollout = execute_physical_rollout(
            model,
            target,
            selected_planner,
            contact_contract=contact_contract,
            min_contact_normal_force=qc_config.min_contact_normal_force,
        )
        if target.source_path is not None and rollout.initialization.contract != controller_manifest.get(
            "initial_state_contract"
        ):
            raise ValueError("production rollout initialization differs from its optimizer artifact contract")
        replay = replay_physical_rollout(
            model,
            rollout,
            actadr=channel_contract.actuator_actadr,
            contact_contract=contact_contract,
            min_contact_normal_force=qc_config.min_contact_normal_force,
        )
        qc = evaluate_rollout_qc(
            model,
            rollout,
            replay,
            ctrlrange=ctrlrange,
            expected_transition_count=target.transition_count,
            config=qc_config,
            task_id=target.task_id,
            target_contact_audit=target_contact_audit,
            contact_contract=contact_contract,
        )
        qc_path = staging / "rollout_qc.npz"
        _write_qc_arrays(
            qc_path,
            rollout=rollout,
            replay=replay,
            target_contact_audit=target_contact_audit,
        )
        trial_path: Path | None = None
        trial_sha256: str | None = None
        if rollout.transition_count > 0:
            trial_path = staging / "primitive_trial.npz"
            trial_sha256 = write_primitive_trial_npz(
                trial_path,
                model=model,
                actuator_names=names,
                applied_ctrl=rollout.applied_ctrl,
                phase_id=rollout.phase_id,
                success=bool(qc["passed"]),
                muscle_activation=rollout.muscle_activation,
                muscle_force=rollout.muscle_force,
                muscle_tendon_length=rollout.muscle_tendon_length,
                muscle_tendon_velocity=rollout.muscle_tendon_velocity,
            )
        manifest: dict[str, Any] = {
            "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
            "trial_id": str(trial_id),
            "task_id": target.task_id,
            "target_skill_id": "ChinaJump",
            "contains_target_skill_rollout": False,
            "source_motion_path": target.source_motion_path,
            "source_motion_uid": int(stable_motion_uid(target.source_motion_path)),
            "source_artifact_sha256": target.source_sha256,
            "source_frequency_hz": target.source_frequency_hz,
            "source_frame_interval": {
                "start_frame": target.source_start_frame,
                "end_frame_exclusive": target.source_end_frame_exclusive,
                "source_total_frames": target.source_total_frames,
            },
            "phase_schema_fingerprint": target.phase_schema_fingerprint,
            "contact_contract": contact_contract.as_dict(),
            "axial_rotation_signal_contract": target_contact_audit.axial_rotation.contract.as_dict(),
            "target_contact_semantics": target_contact_audit.semantics,
            "seed": int(seed),
            "optimizer_fingerprint": controller_manifest["optimizer_fingerprint"],
            "controller_source_kind": controller_manifest["source_kind"],
            "runtime_model_binding": controller_manifest["runtime_model_binding"],
            "runtime_model_provenance": validated_runtime_provenance,
            "policy_rollout_binding": (
                None if policy_rollout_binding is None else _validate_policy_rollout_binding(policy_rollout_binding)
            ),
            "optimizer_artifact": {
                "path": str(controller_dir),
                "manifest_sha256": file_sha256(controller_dir / "optimizer_manifest.json"),
                "model_artifact_sha256": controller_manifest["model_artifact_sha256"],
            },
            "initialization_contract": {
                "contract": rollout.initialization.contract,
                "muscle_state_source": "controller_inference_not_motion_observation",
                "target_acceleration_method": rollout.initialization.target_acceleration_method,
                "warmup_transition_count": 0,
                "initial_activation_sha256": _array_sha256(rollout.initialization.initial_activation),
                "initial_ctrl_sha256": _array_sha256(rollout.initialization.initial_ctrl),
                "initial_integration_state_sha256": _array_sha256(rollout.initial_integration_state),
                "activation_equals_ctrl": bool(
                    np.array_equal(
                        rollout.initialization.initial_activation,
                        rollout.initialization.initial_ctrl,
                    )
                ),
                "activation_dynamics_steady_state_only_not_mechanical_equilibrium": True,
                "solver": {
                    "kind": rollout.initialization.solver_kind,
                    "status": rollout.initialization.solver_status,
                    "iterations": rollout.initialization.solver_iterations,
                    "optimality": rollout.initialization.solver_optimality,
                    "linearized_acceleration_residual_norm": (
                        rollout.initialization.linearized_acceleration_residual_norm
                    ),
                    "linearized_residual_can_mark_success": False,
                },
                "forward_acceleration_error": {
                    "rmse": float(np.sqrt(np.mean(np.square(rollout.initialization.forward_acceleration_error)))),
                    "abs_max": float(np.max(np.abs(rollout.initialization.forward_acceleration_error))),
                    "diagnostic_only": True,
                },
                "initial_exact_contact": {
                    "left": rollout.initialization.initial_left_foot_floor_contact,
                    "right": rollout.initialization.initial_right_foot_floor_contact,
                    "left_normal_force": rollout.initialization.initial_left_foot_floor_normal_force,
                    "right_normal_force": rollout.initialization.initial_right_foot_floor_normal_force,
                },
                "constant_control_shadow_transition": {
                    "duration": rollout.initialization.shadow_transition_duration,
                    "substep_count": int(rollout.initialization.shadow_qpos.shape[0]),
                    "qpos_sha256": _array_sha256(rollout.initialization.shadow_qpos),
                    "qvel_sha256": _array_sha256(rollout.initialization.shadow_qvel),
                    "final_integration_state_sha256": _array_sha256(
                        rollout.initialization.shadow_final_integration_state
                    ),
                    "position_rmse": float(np.sqrt(np.mean(np.square(rollout.initialization.shadow_position_error)))),
                    "position_abs_max": float(np.max(np.abs(rollout.initialization.shadow_position_error))),
                    "velocity_rmse": float(np.sqrt(np.mean(np.square(rollout.initialization.shadow_velocity_error)))),
                    "velocity_abs_max": float(np.max(np.abs(rollout.initialization.shadow_velocity_error))),
                    "left_contact_by_substep": (rollout.initialization.shadow_left_foot_floor_contact.tolist()),
                    "right_contact_by_substep": (rollout.initialization.shadow_right_foot_floor_contact.tolist()),
                    "left_normal_force_by_substep": (
                        rollout.initialization.shadow_left_foot_floor_normal_force.tolist()
                    ),
                    "right_normal_force_by_substep": (
                        rollout.initialization.shadow_right_foot_floor_normal_force.tolist()
                    ),
                },
                "solver_or_seed_alone_can_mark_success": False,
                "instant_forward_acceleration_is_diagnostic_only": True,
                "shadow_tracking_and_contact_are_gated_by_rollout_qc": True,
            },
            "transition_contract": {
                "control": "data.ctrl immediately before s_t_to_s_t_plus_1",
                "activation": "data.act[model.actuator_actadr] after s_t_to_s_t_plus_1",
                "foot_floor_contact": (
                    "post-transition exact floor-vs-stable-foot-geom contact with summed positive "
                    "mujoco.mj_contactForce normal force"
                ),
                "physics_substeps": rollout.transition_substeps.tolist(),
                "mujoco_state_spec": "mjSTATE_INTEGRATION",
            },
            "requested_transition_count": target.transition_count,
            "recorded_transition_count": rollout.transition_count,
            "qc_config": asdict(qc_config),
            "qc": qc,
            "success": bool(qc["passed"]),
            "production_eligible": bool(
                qc["passed"]
                and target.task_id != "P00_synthetic_fixture"
                and controller_manifest["production_eligible"] is True
            ),
            "status": "success" if bool(qc["passed"]) else "failed_qc",
            "artifacts": {
                "primitive_trial": (
                    None if trial_path is None else {"filename": trial_path.name, "sha256": trial_sha256}
                ),
                "rollout_qc": {"filename": qc_path.name, "sha256": file_sha256(qc_path)},
            },
        }
        rollout_fingerprint = canonical_json_sha256(manifest)
        manifest["rollout_fingerprint"] = rollout_fingerprint
        _write_json(staging / "rollout_manifest.json", manifest)
        os.replace(staging, output)
        staging = None
        published_trial = None if trial_path is None else output / trial_path.name
        return ProducerResult(
            output_dir=output,
            controller_dir=controller_dir,
            controller_fingerprint=str(controller_manifest["optimizer_fingerprint"]),
            rollout_fingerprint=rollout_fingerprint,
            trial_path=published_trial,
            success=bool(qc["passed"]),
            qc=qc,
        )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def execute_physical_rollout(
    model: mujoco.MjModel,
    target: MotionTarget,
    planner: PhysicalControlPlanner,
    *,
    contact_contract: FootFloorContactContract | None = None,
    min_contact_normal_force: float = 1.0e-6,
) -> RolloutArrays:
    """Run actual transitions and retain exact physical simulator signals."""

    target = target.validated(model)
    names = complete_actuator_names(model)
    channel_contract = resolve_muscle_channel_contract(model, names)
    ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    contract = contact_contract or resolve_foot_floor_contact_contract(
        model,
        allow_unavailable=target.task_id == "P00_synthetic_fixture",
    )
    actadr = np.asarray(channel_contract.actuator_actadr, dtype=np.int32)
    data = mujoco.MjData(model)
    data.qpos[:] = target.qpos[0]
    data.qvel[:] = target.qvel[0]
    data.ctrl[:] = ctrlrange[:, 0]
    mujoco.mj_forward(model, data)
    initialization = planner.initialize(
        data,
        target,
        contact_contract=contract,
        min_contact_normal_force=min_contact_normal_force,
    )
    _require_first_target_state(model, data, target)
    shadow_count = int(target.transition_substeps[0])
    if (
        not isinstance(initialization, RolloutInitialization)
        or not str(initialization.contract).strip()
        or not str(initialization.target_acceleration_method).strip()
        or np.asarray(initialization.initial_activation).shape != (int(model.nu),)
        or np.asarray(initialization.initial_ctrl).shape != (int(model.nu),)
        or np.asarray(initialization.forward_acceleration_error).shape != (int(model.nv),)
        or np.asarray(initialization.shadow_position_error).shape != (int(model.nv),)
        or np.asarray(initialization.shadow_velocity_error).shape != (int(model.nv),)
        or np.asarray(initialization.shadow_qpos).shape != (shadow_count, int(model.nq))
        or np.asarray(initialization.shadow_qvel).shape != (shadow_count, int(model.nv))
        or np.asarray(initialization.shadow_left_foot_floor_contact).shape != (shadow_count,)
        or np.asarray(initialization.shadow_right_foot_floor_contact).shape != (shadow_count,)
        or np.asarray(initialization.shadow_left_foot_floor_normal_force).shape != (shadow_count,)
        or np.asarray(initialization.shadow_right_foot_floor_normal_force).shape != (shadow_count,)
        or np.asarray(initialization.shadow_final_integration_state).shape
        != (int(mujoco.mj_stateSize(model, _STATE_SPEC)),)
        or not np.all(np.isfinite(initialization.initial_activation))
        or not np.all(np.isfinite(initialization.initial_ctrl))
        or not np.all(np.isfinite(initialization.forward_acceleration_error))
        or not np.all(np.isfinite(initialization.shadow_qpos))
        or not np.all(np.isfinite(initialization.shadow_qvel))
        or not np.all(np.isfinite(initialization.shadow_position_error))
        or not np.all(np.isfinite(initialization.shadow_velocity_error))
        or not np.all(np.isfinite(initialization.shadow_left_foot_floor_normal_force))
        or not np.all(np.isfinite(initialization.shadow_right_foot_floor_normal_force))
        or not np.all(np.isfinite(initialization.shadow_final_integration_state))
        or not np.isfinite(initialization.linearized_acceleration_residual_norm)
        or initialization.linearized_acceleration_residual_norm < 0.0
    ):
        raise ValueError("physical planner returned an invalid rollout initialization contract")
    if not np.array_equal(initialization.initial_activation, np.asarray(data.act[actadr])):
        raise ValueError("planner initialization activation does not equal the simulator state")
    if not np.array_equal(initialization.initial_ctrl, np.asarray(data.ctrl)):
        raise ValueError("planner initialization control does not equal the simulator state")
    if np.any(initialization.initial_ctrl < ctrlrange[:, 0]) or np.any(initialization.initial_ctrl > ctrlrange[:, 1]):
        raise ValueError("planner initialization control lies outside physical ctrlrange")
    initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    initial_state = _capture_integration_state(model, data)
    previous_ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()

    control_rows: list[np.ndarray] = []
    activation_rows: list[np.ndarray] = []
    force_rows: list[np.ndarray] = []
    length_rows: list[np.ndarray] = []
    velocity_rows: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    position_rows: list[np.ndarray] = []
    velocity_error_rows: list[np.ndarray] = []
    residual_rows: list[float] = []
    left_contact_rows: list[bool] = []
    right_contact_rows: list[bool] = []
    left_normal_force_rows: list[float] = []
    right_normal_force_rows: list[float] = []

    for transition_index in range(target.transition_count):
        ctrl, proposal_residual = planner.plan(data, target, transition_index, previous_ctrl)
        physical_ctrl = np.asarray(ctrl, dtype=np.float64)
        if (
            physical_ctrl.shape != (int(model.nu),)
            or not np.all(np.isfinite(physical_ctrl))
            or not np.isfinite(proposal_residual)
        ):
            break
        if np.any(physical_ctrl < ctrlrange[:, 0]) or np.any(physical_ctrl > ctrlrange[:, 1]):
            break
        data.ctrl[:] = physical_ctrl
        # Copy from data.ctrl itself, after assignment, so the evidence is the
        # exact value held constant across this transition's physics steps.
        actual_transition_ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()
        for _ in range(int(target.transition_substeps[transition_index])):
            mujoco.mj_step(model, data)
        if not _finite_dynamic_state(data):
            break
        try:
            left_contact, right_contact, left_normal_force, right_normal_force = _foot_floor_contact_sample(
                model,
                data,
                contract=contract,
                min_normal_force=min_contact_normal_force,
            )
        except ValueError:
            break
        control_rows.append(actual_transition_ctrl)
        activation_rows.append(np.asarray(data.act[actadr], dtype=np.float64).copy())
        force_rows.append(np.asarray(data.actuator_force, dtype=np.float64).copy())
        length_rows.append(np.asarray(data.actuator_length, dtype=np.float64).copy())
        velocity_rows.append(np.asarray(data.actuator_velocity, dtype=np.float64).copy())
        qpos_rows.append(np.asarray(data.qpos, dtype=np.float64).copy())
        qvel_rows.append(np.asarray(data.qvel, dtype=np.float64).copy())
        position_rows.append(_position_difference(model, data.qpos, target.qpos[transition_index + 1]))
        velocity_error_rows.append(np.asarray(data.qvel - target.qvel[transition_index + 1], dtype=np.float64))
        residual_rows.append(float(proposal_residual))
        left_contact_rows.append(left_contact)
        right_contact_rows.append(right_contact)
        left_normal_force_rows.append(left_normal_force)
        right_normal_force_rows.append(right_normal_force)
        previous_ctrl = actual_transition_ctrl

    recorded = len(control_rows)
    actual_qpos = _rows(qpos_rows, width=int(model.nq))
    actual_qvel = _rows(qvel_rows, width=int(model.nv))
    recorded_substeps = np.asarray(target.transition_substeps[:recorded], dtype=np.int32).copy()
    (
        actual_root_position,
        actual_root_velocity,
        actual_com_position,
        actual_com_velocity,
    ) = _reconstruct_actual_vertical_kinematics(
        model,
        initial_integration_state=initial_state,
        actual_qpos=actual_qpos,
        actual_qvel=actual_qvel,
        transition_substeps=recorded_substeps,
        allow_unavailable=target.task_id == "P00_synthetic_fixture",
    )
    actual_axial_rotation = reconstruct_axial_rotation_evidence(
        model,
        initial_qpos=initial_qpos,
        post_transition_qpos=actual_qpos,
        transition_substeps=recorded_substeps,
        allow_unavailable=target.task_id.split("_", 1)[0] != "P08",
    )
    return RolloutArrays(
        applied_ctrl=_rows(control_rows, width=int(model.nu)),
        muscle_activation=_rows(activation_rows, width=int(model.nu)),
        muscle_force=_rows(force_rows, width=int(model.nu)),
        muscle_tendon_length=_rows(length_rows, width=int(model.nu)),
        muscle_tendon_velocity=_rows(velocity_rows, width=int(model.nu)),
        actual_qpos=actual_qpos,
        actual_qvel=actual_qvel,
        actual_root_vertical_position=actual_root_position,
        actual_root_vertical_velocity=actual_root_velocity,
        actual_com_vertical_position=actual_com_position,
        actual_com_vertical_velocity=actual_com_velocity,
        actual_axial_rotation=actual_axial_rotation,
        target_qpos=np.asarray(target.qpos[1 : recorded + 1], dtype=np.float64).copy(),
        target_qvel=np.asarray(target.qvel[1 : recorded + 1], dtype=np.float64).copy(),
        position_error=_rows(position_rows, width=int(model.nv)),
        velocity_error=_rows(velocity_error_rows, width=int(model.nv)),
        phase_id=np.asarray(target.phase_id[:recorded], dtype=np.int32).copy(),
        transition_substeps=recorded_substeps,
        initialization=initialization,
        initial_integration_state=initial_state,
        proposal_tracking_residual_norm=np.asarray(residual_rows, dtype=np.float64),
        left_foot_floor_contact=np.asarray(left_contact_rows, dtype=np.bool_),
        right_foot_floor_contact=np.asarray(right_contact_rows, dtype=np.bool_),
        left_foot_floor_normal_force=np.asarray(left_normal_force_rows, dtype=np.float64),
        right_foot_floor_normal_force=np.asarray(right_normal_force_rows, dtype=np.float64),
    )


def replay_physical_rollout(
    model: mujoco.MjModel,
    rollout: RolloutArrays,
    *,
    actadr: Sequence[int],
    contact_contract: FootFloorContactContract | None = None,
    min_contact_normal_force: float = 1.0e-6,
) -> ReplayArrays:
    """Forward replay the exact recorded controls from the initial state."""

    addresses = np.asarray(actadr, dtype=np.int32)
    if addresses.shape != (int(model.nu),):
        raise ValueError("forward replay activation addresses differ from model actuator width")
    contract = contact_contract or resolve_foot_floor_contact_contract(
        model,
        allow_unavailable=not bool(
            rollout.left_foot_floor_contact.size
            and (np.any(rollout.left_foot_floor_contact) or np.any(rollout.right_foot_floor_contact))
        ),
    )
    data = mujoco.MjData(model)
    _restore_integration_state(model, data, rollout.initial_integration_state)
    initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    activation_rows: list[np.ndarray] = []
    left_contact_rows: list[bool] = []
    right_contact_rows: list[bool] = []
    left_normal_force_rows: list[float] = []
    right_normal_force_rows: list[float] = []
    for ctrl, substeps in zip(rollout.applied_ctrl, rollout.transition_substeps, strict=True):
        data.ctrl[:] = ctrl
        for _ in range(int(substeps)):
            mujoco.mj_step(model, data)
        qpos_rows.append(np.asarray(data.qpos, dtype=np.float64).copy())
        qvel_rows.append(np.asarray(data.qvel, dtype=np.float64).copy())
        activation_rows.append(np.asarray(data.act[addresses], dtype=np.float64).copy())
        left_contact, right_contact, left_normal_force, right_normal_force = _foot_floor_contact_sample(
            model,
            data,
            contract=contract,
            min_normal_force=min_contact_normal_force,
        )
        left_contact_rows.append(left_contact)
        right_contact_rows.append(right_contact)
        left_normal_force_rows.append(left_normal_force)
        right_normal_force_rows.append(right_normal_force)
    replay_qpos = _rows(qpos_rows, width=int(model.nq))
    replay_axial_rotation = reconstruct_axial_rotation_evidence(
        model,
        initial_qpos=initial_qpos,
        post_transition_qpos=replay_qpos,
        transition_substeps=rollout.transition_substeps,
        allow_unavailable=not rollout.actual_axial_rotation.contract.available,
    )
    return ReplayArrays(
        qpos=replay_qpos,
        qvel=_rows(qvel_rows, width=int(model.nv)),
        muscle_activation=_rows(activation_rows, width=int(model.nu)),
        left_foot_floor_contact=np.asarray(left_contact_rows, dtype=np.bool_),
        right_foot_floor_contact=np.asarray(right_contact_rows, dtype=np.bool_),
        left_foot_floor_normal_force=np.asarray(left_normal_force_rows, dtype=np.float64),
        right_foot_floor_normal_force=np.asarray(right_normal_force_rows, dtype=np.float64),
        axial_rotation=replay_axial_rotation,
    )


def evaluate_rollout_qc(
    model: mujoco.MjModel,
    rollout: RolloutArrays,
    replay: ReplayArrays,
    *,
    ctrlrange: Any,
    expected_transition_count: int,
    config: RolloutQCConfig,
    task_id: str,
    target_contact_audit: TargetContactAudit,
    contact_contract: FootFloorContactContract,
) -> dict[str, Any]:
    """Evaluate tracking, exact contact semantics, and deterministic replay."""

    config = config.validated()
    ranges = np.asarray(ctrlrange, dtype=np.float64)
    recorded = rollout.transition_count
    complete = recorded == int(expected_transition_count) and recorded > 0
    finite = all(
        np.all(np.isfinite(array))
        for array in (
            rollout.applied_ctrl,
            rollout.muscle_activation,
            rollout.actual_qpos,
            rollout.actual_qvel,
            rollout.actual_root_vertical_position,
            rollout.actual_root_vertical_velocity,
            rollout.actual_com_vertical_position,
            rollout.actual_com_vertical_velocity,
            rollout.actual_axial_rotation.position,
            rollout.actual_axial_rotation.velocity,
            rollout.actual_axial_rotation.root_yaw,
            rollout.actual_axial_rotation.root_xy,
            rollout.actual_axial_rotation.initial_root_xy,
            rollout.position_error,
            rollout.velocity_error,
            replay.qpos,
            replay.qvel,
            replay.muscle_activation,
            replay.axial_rotation.position,
            replay.axial_rotation.velocity,
            replay.axial_rotation.root_yaw,
            replay.axial_rotation.root_xy,
            replay.axial_rotation.initial_root_xy,
            rollout.left_foot_floor_normal_force,
            rollout.right_foot_floor_normal_force,
            replay.left_foot_floor_normal_force,
            replay.right_foot_floor_normal_force,
        )
    )
    finite = bool(
        finite
        and all(
            np.isfinite(value)
            for value in (
                rollout.actual_axial_rotation.initial_position,
                rollout.actual_axial_rotation.initial_root_yaw,
                replay.axial_rotation.initial_position,
                replay.axial_rotation.initial_root_yaw,
            )
        )
    )
    contact_shapes = all(
        array.shape == (recorded,)
        for array in (
            rollout.left_foot_floor_contact,
            rollout.right_foot_floor_contact,
            rollout.left_foot_floor_normal_force,
            rollout.right_foot_floor_normal_force,
            replay.left_foot_floor_contact,
            replay.right_foot_floor_contact,
            replay.left_foot_floor_normal_force,
            replay.right_foot_floor_normal_force,
        )
    )
    vertical_shapes = all(
        array.shape == (recorded,)
        for array in (
            rollout.actual_root_vertical_position,
            rollout.actual_root_vertical_velocity,
            rollout.actual_com_vertical_position,
            rollout.actual_com_vertical_velocity,
        )
    )
    axial_shapes = bool(
        all(
            evidence.position.shape == (recorded,)
            and evidence.velocity.shape == (recorded,)
            and evidence.root_yaw.shape == (recorded,)
            and evidence.root_xy.shape == (recorded, 2)
            and evidence.initial_root_xy.shape == (2,)
            for evidence in (rollout.actual_axial_rotation, replay.axial_rotation)
        )
    )
    target_axial_contract = target_contact_audit.axial_rotation.contract.as_dict()
    actual_axial_contract = rollout.actual_axial_rotation.contract.as_dict()
    replay_axial_contract = replay.axial_rotation.contract.as_dict()
    axial_contract_consistent = bool(target_axial_contract == actual_axial_contract == replay_axial_contract)
    actual_vertical_signal, actual_vertical_position, actual_vertical_velocity = _vertical_signal_for_task(
        task_id,
        root_position=rollout.actual_root_vertical_position,
        root_velocity=rollout.actual_root_vertical_velocity,
        com_position=rollout.actual_com_vertical_position,
        com_velocity=rollout.actual_com_vertical_velocity,
    )
    actual_contact_semantics = evaluate_task_contact_semantics(
        task_id=task_id,
        phase_id=rollout.phase_id,
        left_contact=rollout.left_foot_floor_contact,
        right_contact=rollout.right_foot_floor_contact,
        left_normal_force=rollout.left_foot_floor_normal_force,
        right_normal_force=rollout.right_foot_floor_normal_force,
        vertical_position=actual_vertical_position,
        vertical_velocity=actual_vertical_velocity,
        config=config,
        evidence_kind="actual_rollout_exact_contact",
        vertical_signal=actual_vertical_signal,
        **_axial_semantic_arguments(rollout.actual_axial_rotation),
    )
    task_family = str(task_id).split("_", 1)[0]
    if task_family == "P08":
        target_axial_direction = (
            target_contact_audit.semantics.get("exact", {}).get("metrics", {}).get("axial_rotation_direction")
        )
        actual_axial_direction = actual_contact_semantics.get("metrics", {}).get("axial_rotation_direction")
        axial_direction_consistent = bool(
            target_axial_direction in {-1.0, 1.0}
            and actual_axial_direction in {-1.0, 1.0}
            and target_axial_direction == actual_axial_direction
        )
    else:
        target_axial_direction = actual_axial_direction = None
        axial_direction_consistent = True
    if recorded > 0:
        position_rmse = float(np.sqrt(np.mean(np.square(rollout.position_error))))
        velocity_rmse = float(np.sqrt(np.mean(np.square(rollout.velocity_error))))
        position_abs = float(np.max(np.abs(rollout.position_error)))
        velocity_abs = float(np.max(np.abs(rollout.velocity_error)))
        saturation = np.isclose(rollout.applied_ctrl, ranges[:, 0], atol=1.0e-8, rtol=0.0) | np.isclose(
            rollout.applied_ctrl,
            ranges[:, 1],
            atol=1.0e-8,
            rtol=0.0,
        )
        saturation_fraction = float(np.mean(saturation))
        replay_position_abs = float(np.max(np.abs(replay.qpos - rollout.actual_qpos)))
        replay_velocity_abs = float(np.max(np.abs(replay.qvel - rollout.actual_qvel)))
        replay_activation_abs = float(np.max(np.abs(replay.muscle_activation - rollout.muscle_activation)))
        replay_left_force_abs = float(
            np.max(np.abs(replay.left_foot_floor_normal_force - rollout.left_foot_floor_normal_force))
        )
        replay_right_force_abs = float(
            np.max(np.abs(replay.right_foot_floor_normal_force - rollout.right_foot_floor_normal_force))
        )
        replay_axial_position_abs = float(
            np.max(np.abs(replay.axial_rotation.position - rollout.actual_axial_rotation.position))
        )
        replay_axial_velocity_abs = float(
            np.max(np.abs(replay.axial_rotation.velocity - rollout.actual_axial_rotation.velocity))
        )
        replay_axial_root_yaw_abs = float(
            np.max(np.abs(replay.axial_rotation.root_yaw - rollout.actual_axial_rotation.root_yaw))
        )
        replay_axial_root_xy_abs = float(
            np.max(np.abs(replay.axial_rotation.root_xy - rollout.actual_axial_rotation.root_xy))
        )
        replay_axial_initial_abs = float(
            max(
                abs(replay.axial_rotation.initial_position - rollout.actual_axial_rotation.initial_position),
                abs(replay.axial_rotation.initial_root_yaw - rollout.actual_axial_rotation.initial_root_yaw),
                np.max(np.abs(replay.axial_rotation.initial_root_xy - rollout.actual_axial_rotation.initial_root_xy)),
            )
        )
        proposal_residual_median = float(np.median(rollout.proposal_tracking_residual_norm))
        proposal_residual_max = float(np.max(rollout.proposal_tracking_residual_norm))
    else:
        position_rmse = velocity_rmse = position_abs = velocity_abs = float("inf")
        saturation_fraction = 1.0
        replay_position_abs = replay_velocity_abs = replay_activation_abs = float("inf")
        replay_left_force_abs = replay_right_force_abs = float("inf")
        replay_axial_position_abs = replay_axial_velocity_abs = float("inf")
        replay_axial_root_yaw_abs = replay_axial_root_xy_abs = replay_axial_initial_abs = float("inf")
        proposal_residual_median = proposal_residual_max = float("inf")
    initialization = rollout.initialization
    initialization_acceleration_rmse = float(np.sqrt(np.mean(np.square(initialization.forward_acceleration_error))))
    initialization_acceleration_abs = float(np.max(np.abs(initialization.forward_acceleration_error)))
    shadow_position_rmse = float(np.sqrt(np.mean(np.square(initialization.shadow_position_error))))
    shadow_position_abs = float(np.max(np.abs(initialization.shadow_position_error)))
    shadow_velocity_rmse = float(np.sqrt(np.mean(np.square(initialization.shadow_velocity_error))))
    shadow_velocity_abs = float(np.max(np.abs(initialization.shadow_velocity_error)))
    if target_contact_audit.semantics.get("gate_basis") == "exact_mj_forward_contact":
        expected_initial_left_contact = target_contact_audit.initial_left_foot_floor_contact
        expected_initial_right_contact = target_contact_audit.initial_right_foot_floor_contact
        expected_shadow_left_contact = bool(target_contact_audit.left_foot_floor_contact[0])
        expected_shadow_right_contact = bool(target_contact_audit.right_foot_floor_contact[0])
        initialization_contact_basis = "target_state0_and_state1_exact_contact"
    else:
        expected_initial_left_contact = target_contact_audit.initial_site_proxy_left_foot_contact
        expected_initial_right_contact = target_contact_audit.initial_site_proxy_right_foot_contact
        expected_shadow_left_contact = bool(target_contact_audit.site_proxy_left_foot_contact[0])
        expected_shadow_right_contact = bool(target_contact_audit.site_proxy_right_foot_contact[0])
        initialization_contact_basis = "target_state0_and_state1_site_proxy"
    known_initialization_contract = initialization.contract in {
        _INFERRED_INITIAL_STATE_CONTRACT,
        _ZERO_INITIAL_STATE_CONTRACT,
        _CANONICAL_TONIC_INITIAL_STATE_CONTRACT,
    }
    solver_contract_valid = bool(
        initialization.solver_status > 0
        if initialization.contract == _INFERRED_INITIAL_STATE_CONTRACT
        else initialization.solver_status == 0
    )
    contact_force_consistent = bool(
        initialization.initial_left_foot_floor_contact
        == (initialization.initial_left_foot_floor_normal_force >= config.min_contact_normal_force)
        and initialization.initial_right_foot_floor_contact
        == (initialization.initial_right_foot_floor_normal_force >= config.min_contact_normal_force)
        and np.array_equal(
            initialization.shadow_left_foot_floor_contact,
            initialization.shadow_left_foot_floor_normal_force >= config.min_contact_normal_force,
        )
        and np.array_equal(
            initialization.shadow_right_foot_floor_contact,
            initialization.shadow_right_foot_floor_normal_force >= config.min_contact_normal_force,
        )
    )
    p01_shadow_bilateral = bool(
        task_id.split("_", 1)[0] != "P01"
        or (
            np.all(initialization.shadow_left_foot_floor_contact)
            and np.all(initialization.shadow_right_foot_floor_contact)
        )
    )
    initialization_gate_required = task_id != "P00_synthetic_fixture"

    def initialization_gate(value: bool) -> bool:
        return bool(value or not initialization_gate_required)

    gates = {
        "complete_trajectory": bool(complete),
        "finite_dynamic_signals": bool(finite),
        "contact_array_shapes": bool(contact_shapes),
        "actual_vertical_signal_shapes": bool(vertical_shapes),
        "actual_replay_axial_signal_shapes": axial_shapes,
        "target_actual_vertical_signal_consistent": bool(
            target_contact_audit.semantics.get("vertical_signal") == actual_vertical_signal
        ),
        "target_actual_replay_axial_contract_consistent": axial_contract_consistent,
        "target_actual_axial_rotation_direction_consistent": axial_direction_consistent,
        "position_rmse": bool(position_rmse <= config.max_position_rmse),
        "velocity_rmse": bool(velocity_rmse <= config.max_velocity_rmse),
        "position_abs": bool(position_abs <= config.max_position_abs),
        "velocity_abs": bool(velocity_abs <= config.max_velocity_abs),
        "saturation_fraction": bool(saturation_fraction <= config.max_saturation_fraction),
        "initialization_contract_known": initialization_gate(known_initialization_contract),
        "initialization_solver_contract": initialization_gate(solver_contract_valid),
        "initialization_shadow_position_rmse": initialization_gate(shadow_position_rmse <= config.max_position_rmse),
        "initialization_shadow_position_abs": initialization_gate(shadow_position_abs <= config.max_position_abs),
        "initialization_shadow_velocity_rmse": initialization_gate(shadow_velocity_rmse <= config.max_velocity_rmse),
        "initialization_shadow_velocity_abs": initialization_gate(shadow_velocity_abs <= config.max_velocity_abs),
        "initialization_initial_left_contact_matches_target": initialization_gate(
            initialization.initial_left_foot_floor_contact == expected_initial_left_contact
        ),
        "initialization_initial_right_contact_matches_target": initialization_gate(
            initialization.initial_right_foot_floor_contact == expected_initial_right_contact
        ),
        "initialization_shadow_left_contact_matches_target": initialization_gate(
            bool(initialization.shadow_left_foot_floor_contact[-1]) == expected_shadow_left_contact
        ),
        "initialization_shadow_right_contact_matches_target": initialization_gate(
            bool(initialization.shadow_right_foot_floor_contact[-1]) == expected_shadow_right_contact
        ),
        "initialization_shadow_p01_bilateral_contact_all_substeps": initialization_gate(p01_shadow_bilateral),
        "initialization_contact_force_consistent": initialization_gate(contact_force_consistent),
        "forward_replay_position": bool(replay_position_abs <= config.replay_position_atol),
        "forward_replay_velocity": bool(replay_velocity_abs <= config.replay_velocity_atol),
        "forward_replay_activation": bool(replay_activation_abs <= config.replay_activation_atol),
        "forward_replay_left_contact_bool": bool(
            contact_shapes
            and np.array_equal(
                replay.left_foot_floor_contact,
                rollout.left_foot_floor_contact,
            )
        ),
        "forward_replay_right_contact_bool": bool(
            contact_shapes
            and np.array_equal(
                replay.right_foot_floor_contact,
                rollout.right_foot_floor_contact,
            )
        ),
        "forward_replay_left_contact_force": bool(replay_left_force_abs <= config.replay_contact_force_atol),
        "forward_replay_right_contact_force": bool(replay_right_force_abs <= config.replay_contact_force_atol),
        "forward_replay_axial_position": bool(replay_axial_position_abs <= config.replay_position_atol),
        "forward_replay_axial_velocity": bool(replay_axial_velocity_abs <= config.replay_velocity_atol),
        "forward_replay_axial_root_yaw": bool(replay_axial_root_yaw_abs <= config.replay_position_atol),
        "forward_replay_axial_root_xy": bool(replay_axial_root_xy_abs <= config.replay_position_atol),
        "forward_replay_axial_initial_state": bool(replay_axial_initial_abs <= config.replay_position_atol),
        "target_contact_phase_semantics": bool(target_contact_audit.semantics["passed"]),
        "actual_contact_phase_semantics": bool(actual_contact_semantics["passed"]),
    }
    metrics = {
        "position_rmse": position_rmse,
        "velocity_rmse": velocity_rmse,
        "position_abs_max": position_abs,
        "velocity_abs_max": velocity_abs,
        "saturation_fraction": saturation_fraction,
        "initialization_forward_acceleration_rmse": initialization_acceleration_rmse,
        "initialization_forward_acceleration_abs_max": initialization_acceleration_abs,
        "initialization_shadow_position_rmse": shadow_position_rmse,
        "initialization_shadow_position_abs_max": shadow_position_abs,
        "initialization_shadow_velocity_rmse": shadow_velocity_rmse,
        "initialization_shadow_velocity_abs_max": shadow_velocity_abs,
        "forward_replay_position_abs_max": replay_position_abs,
        "forward_replay_velocity_abs_max": replay_velocity_abs,
        "forward_replay_activation_abs_max": replay_activation_abs,
        "forward_replay_left_contact_force_abs_max": replay_left_force_abs,
        "forward_replay_right_contact_force_abs_max": replay_right_force_abs,
        "forward_replay_axial_position_abs_max": replay_axial_position_abs,
        "forward_replay_axial_velocity_abs_max": replay_axial_velocity_abs,
        "forward_replay_axial_root_yaw_abs_max": replay_axial_root_yaw_abs,
        "forward_replay_axial_root_xy_abs_max": replay_axial_root_xy_abs,
        "forward_replay_axial_initial_state_abs_max": replay_axial_initial_abs,
        "shooting_proposal_tracking_residual_median": proposal_residual_median,
        "shooting_proposal_tracking_residual_max": proposal_residual_max,
    }
    # Strict JSON artifacts forbid Infinity/NaN.  A missing metric is explicit
    # evidence of a zero-length/invalid rollout and every corresponding gate is
    # already false.
    json_metrics = {key: (None if not np.isfinite(value) else float(value)) for key, value in metrics.items()}
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "metrics": json_metrics,
        "recorded_transition_count": recorded,
        "expected_transition_count": int(expected_transition_count),
        "contact_contract": contact_contract.as_dict(),
        "axial_rotation_signal_contract": target_contact_audit.axial_rotation.contract.as_dict(),
        "target_contact_semantics": target_contact_audit.semantics,
        "actual_contact_semantics": actual_contact_semantics,
        "axial_rotation_direction_evidence": {
            "required": task_family == "P08",
            "target": target_axial_direction,
            "actual": actual_axial_direction,
        },
        "initialization_evidence": {
            "contract": initialization.contract,
            "gate_required": initialization_gate_required,
            "contact_expectation_basis": initialization_contact_basis,
            "expected_initial_left_contact": expected_initial_left_contact,
            "expected_initial_right_contact": expected_initial_right_contact,
            "expected_shadow_left_contact": expected_shadow_left_contact,
            "expected_shadow_right_contact": expected_shadow_right_contact,
            "initial_left_contact": initialization.initial_left_foot_floor_contact,
            "initial_right_contact": initialization.initial_right_foot_floor_contact,
            "shadow_left_contact_by_substep": initialization.shadow_left_foot_floor_contact.tolist(),
            "shadow_right_contact_by_substep": initialization.shadow_right_foot_floor_contact.tolist(),
            "shadow_transition_duration": initialization.shadow_transition_duration,
            "linearized_acceleration_residual_norm": (initialization.linearized_acceleration_residual_norm),
            "linearized_residual_can_mark_success": False,
            "instant_forward_acceleration_diagnostic_only": True,
        },
        "replay_contact_evidence": {
            "left_foot_floor_contact": replay.left_foot_floor_contact.tolist(),
            "right_foot_floor_contact": replay.right_foot_floor_contact.tolist(),
            "left_foot_floor_normal_force": replay.left_foot_floor_normal_force.tolist(),
            "right_foot_floor_normal_force": replay.right_foot_floor_normal_force.tolist(),
        },
        "replay_axial_rotation_evidence": {
            "signal_contract": replay_axial_contract,
            "position": replay.axial_rotation.position.tolist(),
            "velocity": replay.axial_rotation.velocity.tolist(),
            "root_yaw": replay.axial_rotation.root_yaw.tolist(),
            "root_xy": replay.axial_rotation.root_xy.tolist(),
            "initial_position": replay.axial_rotation.initial_position,
            "initial_root_yaw": replay.axial_rotation.initial_root_yaw,
            "initial_root_xy": replay.axial_rotation.initial_root_xy.tolist(),
        },
        "success_does_not_depend_on_shooting_proposal_residual": True,
    }


def preflight_physical_primitive_target(
    *,
    source_npz: str | Path,
    source_motion_path: str,
    phase_schema_path: str | Path,
    phase_plan_path: str | Path,
    controller_store: str | Path,
    optimizer_config: PhysicalOptimizerConfig,
    qc_config: RolloutQCConfig | None = None,
    target_skill_id: str = "ChinaJump",
    excluded_target_motion_paths: Sequence[str] = (),
    start_frame: int = 0,
    end_frame_exclusive: int | None = None,
    config_name: str = "config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2",
    hydra_overrides: Sequence[str] = (),
    verified_runtime_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all production inputs and save the exact runtime optimizer."""

    runtime = resolve_chinajump_runtime_model(
        config_name=config_name,
        hydra_overrides=hydra_overrides,
        verified_runtime_artifact=verified_runtime_artifact,
    )
    model = runtime.model
    target = load_retargeted_motion_target(
        source_npz,
        model=model,
        source_motion_path=source_motion_path,
        phase_schema_path=phase_schema_path,
        phase_plan_path=phase_plan_path,
        target_skill_id=target_skill_id,
        excluded_target_motion_paths=excluded_target_motion_paths,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
    )
    semantic_config = (
        RolloutQCConfig(
            max_position_rmse=1.0,
            max_velocity_rmse=1.0,
            max_position_abs=1.0,
            max_velocity_abs=1.0,
        )
        if qc_config is None
        else qc_config
    ).validated()
    contact_contract = resolve_foot_floor_contact_contract(model)
    target_contact_audit = audit_target_contact_semantics(
        model,
        target,
        config=semantic_config,
        contact_contract=contact_contract,
    )
    controller_dir, manifest = ensure_optimizer_artifact(
        model,
        controller_store=controller_store,
        config=optimizer_config,
        runtime_model_binding=runtime.binding,
    )
    return {
        "preflight_passed": bool(target_contact_audit.semantics["passed"]),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "model_na": int(model.na),
        "source_motion_path": target.source_motion_path,
        "source_sha256": target.source_sha256,
        "transition_count": target.transition_count,
        "source_frame_interval": {
            "start_frame": target.source_start_frame,
            "end_frame_exclusive": target.source_end_frame_exclusive,
            "source_total_frames": target.source_total_frames,
        },
        "phase_ids": sorted({int(value) for value in target.phase_id.tolist()}),
        "contact_contract": contact_contract.as_dict(),
        "target_contact_semantics": target_contact_audit.semantics,
        "semantic_qc_config": semantic_config.semantic_thresholds(),
        "optimizer_fingerprint": manifest["optimizer_fingerprint"],
        "optimizer_artifact": str(controller_dir),
        "runtime_model_mjb": str(controller_dir / "runtime_model.mjb"),
        "runtime_model_binding": runtime.binding,
        "runtime_model_provenance": _validate_runtime_model_provenance(
            runtime.provenance,
            model_hash=_model_hash(model),
            runtime_model_binding=runtime.binding,
        ),
    }


def _position_difference(model: mujoco.MjModel, actual_qpos: Any, target_qpos: Any) -> np.ndarray:
    difference = np.empty((int(model.nv),), dtype=np.float64)
    mujoco.mj_differentiatePos(
        model,
        difference,
        1.0,
        np.asarray(actual_qpos, dtype=np.float64),
        np.asarray(target_qpos, dtype=np.float64),
    )
    return difference


def _capture_integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty((int(mujoco.mj_stateSize(model, _STATE_SPEC)),), dtype=np.float64)
    mujoco.mj_getState(model, data, state, _STATE_SPEC)
    return state


def _restore_integration_state(model: mujoco.MjModel, data: mujoco.MjData, state: Any) -> None:
    values = np.asarray(state, dtype=np.float64)
    expected = int(mujoco.mj_stateSize(model, _STATE_SPEC))
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError("MuJoCo integration state has wrong shape or non-finite values")
    mujoco.mj_setState(model, data, values, _STATE_SPEC)
    mujoco.mj_forward(model, data)


def _finite_dynamic_state(data: mujoco.MjData) -> bool:
    return all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.act, data.ctrl, data.actuator_force))


def _model_hash(model: mujoco.MjModel) -> str:
    state = model.__getstate__()
    if not isinstance(state, bytes) or not state:
        raise ValueError("MuJoCo model has no canonical complete byte state")
    return hashlib.sha256(state).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(len(array.shape).to_bytes(4, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big", signed=False))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _constant_transition_substeps(
    *,
    source_frequency_hz: float,
    physics_timestep: float,
    transition_count: int,
) -> np.ndarray:
    if not np.isfinite(source_frequency_hz) or source_frequency_hz <= 0.0:
        raise ValueError("retargeted target frequency must be finite and positive")
    if not np.isfinite(physics_timestep) or physics_timestep <= 0.0:
        raise ValueError("MuJoCo physics timestep must be finite and positive")
    ratio = 1.0 / (float(source_frequency_hz) * float(physics_timestep))
    rounded = round(ratio)
    if rounded <= 0 or not np.isclose(ratio, rounded, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(
            "target frame interval is not an integer number of MuJoCo physics steps; "
            "resampling must be explicit before primitive production"
        )
    if int(transition_count) <= 0:
        raise ValueError("retargeted target must contain at least one transition")
    return np.full((int(transition_count),), rounded, dtype=np.int32)


def _validate_model_quaternions(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    normalized = np.asarray(qpos, dtype=np.float64).copy()
    for row in normalized:
        mujoco.mj_normalizeQuat(model, row)
    if not np.allclose(normalized, qpos, rtol=0.0, atol=1.0e-5):
        raise ValueError("target qpos contains non-unit joint quaternions; silent normalization is forbidden")


def _assert_not_target_skill_source(
    source_motion_path: str,
    source_path: Path | None,
    *,
    target_skill_id: str,
) -> None:
    target = str(target_skill_id).strip().casefold()
    if not target:
        raise ValueError("target_skill_id must be non-empty")
    relative_parts = {part.casefold() for part in PurePosixPath(source_motion_path).parts}
    resolved_parts = set() if source_path is None else {part.casefold() for part in source_path.resolve().parts}
    if target in relative_parts or target in resolved_parts:
        raise ValueError(
            f"primitive producer rejects target-skill source {target_skill_id!r}; use an independent primitive motion"
        )


def _rows(rows: Sequence[np.ndarray], *, width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, int(width)), dtype=np.float64)
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (len(rows), int(width)):
        raise ValueError("recorded simulator channel has inconsistent width")
    return values


def _write_qc_arrays(
    path: Path,
    *,
    rollout: RolloutArrays,
    replay: ReplayArrays,
    target_contact_audit: TargetContactAudit,
) -> None:
    temporary = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(QC_ARRAY_SCHEMA_VERSION),
            applied_ctrl=rollout.applied_ctrl.astype(np.float64),
            muscle_activation=rollout.muscle_activation.astype(np.float64),
            muscle_force=rollout.muscle_force.astype(np.float64),
            muscle_tendon_length=rollout.muscle_tendon_length.astype(np.float64),
            muscle_tendon_velocity=rollout.muscle_tendon_velocity.astype(np.float64),
            actual_qpos=rollout.actual_qpos.astype(np.float64),
            actual_qvel=rollout.actual_qvel.astype(np.float64),
            actual_root_vertical_position=rollout.actual_root_vertical_position.astype(np.float64),
            actual_root_vertical_velocity=rollout.actual_root_vertical_velocity.astype(np.float64),
            actual_com_vertical_position=rollout.actual_com_vertical_position.astype(np.float64),
            actual_com_vertical_velocity=rollout.actual_com_vertical_velocity.astype(np.float64),
            actual_axial_rotation_position=rollout.actual_axial_rotation.position.astype(np.float64),
            actual_axial_rotation_velocity=rollout.actual_axial_rotation.velocity.astype(np.float64),
            actual_axial_rotation_root_yaw=rollout.actual_axial_rotation.root_yaw.astype(np.float64),
            actual_axial_rotation_root_xy=rollout.actual_axial_rotation.root_xy.astype(np.float64),
            actual_axial_rotation_initial_position=np.asarray(
                rollout.actual_axial_rotation.initial_position, dtype=np.float64
            ),
            actual_axial_rotation_initial_root_yaw=np.asarray(
                rollout.actual_axial_rotation.initial_root_yaw, dtype=np.float64
            ),
            actual_axial_rotation_initial_root_xy=rollout.actual_axial_rotation.initial_root_xy.astype(np.float64),
            actual_axial_rotation_signal_contract_json=np.asarray(
                json.dumps(rollout.actual_axial_rotation.contract.as_dict(), sort_keys=True, allow_nan=False)
            ),
            target_qpos=rollout.target_qpos.astype(np.float64),
            target_qvel=rollout.target_qvel.astype(np.float64),
            position_error=rollout.position_error.astype(np.float64),
            velocity_error=rollout.velocity_error.astype(np.float64),
            phase_id=rollout.phase_id.astype(np.int32),
            transition_substeps=rollout.transition_substeps.astype(np.int32),
            initialization_contract=np.asarray(rollout.initialization.contract),
            initialization_target_acceleration_method=np.asarray(rollout.initialization.target_acceleration_method),
            initial_muscle_activation=rollout.initialization.initial_activation.astype(np.float64),
            initial_ctrl=rollout.initialization.initial_ctrl.astype(np.float64),
            initialization_solver_kind=np.asarray(rollout.initialization.solver_kind),
            initialization_solver_status=np.asarray(rollout.initialization.solver_status, dtype=np.int32),
            initialization_solver_iterations=np.asarray(rollout.initialization.solver_iterations, dtype=np.int32),
            initialization_solver_optimality=np.asarray(rollout.initialization.solver_optimality, dtype=np.float64),
            initialization_linearized_acceleration_residual_norm=np.asarray(
                rollout.initialization.linearized_acceleration_residual_norm, dtype=np.float64
            ),
            initialization_forward_acceleration_error=rollout.initialization.forward_acceleration_error.astype(
                np.float64
            ),
            initialization_shadow_transition_duration=np.asarray(
                rollout.initialization.shadow_transition_duration, dtype=np.float64
            ),
            initialization_initial_left_foot_floor_contact=np.asarray(
                rollout.initialization.initial_left_foot_floor_contact, dtype=np.bool_
            ),
            initialization_initial_right_foot_floor_contact=np.asarray(
                rollout.initialization.initial_right_foot_floor_contact, dtype=np.bool_
            ),
            initialization_initial_left_foot_floor_normal_force=np.asarray(
                rollout.initialization.initial_left_foot_floor_normal_force, dtype=np.float64
            ),
            initialization_initial_right_foot_floor_normal_force=np.asarray(
                rollout.initialization.initial_right_foot_floor_normal_force, dtype=np.float64
            ),
            initialization_shadow_qpos=rollout.initialization.shadow_qpos.astype(np.float64),
            initialization_shadow_qvel=rollout.initialization.shadow_qvel.astype(np.float64),
            initialization_shadow_position_error=rollout.initialization.shadow_position_error.astype(np.float64),
            initialization_shadow_velocity_error=rollout.initialization.shadow_velocity_error.astype(np.float64),
            initialization_shadow_left_foot_floor_contact=(
                rollout.initialization.shadow_left_foot_floor_contact.astype(np.bool_)
            ),
            initialization_shadow_right_foot_floor_contact=(
                rollout.initialization.shadow_right_foot_floor_contact.astype(np.bool_)
            ),
            initialization_shadow_left_foot_floor_normal_force=(
                rollout.initialization.shadow_left_foot_floor_normal_force.astype(np.float64)
            ),
            initialization_shadow_right_foot_floor_normal_force=(
                rollout.initialization.shadow_right_foot_floor_normal_force.astype(np.float64)
            ),
            initialization_shadow_final_integration_state=(
                rollout.initialization.shadow_final_integration_state.astype(np.float64)
            ),
            initial_integration_state=rollout.initial_integration_state.astype(np.float64),
            shooting_proposal_tracking_residual_norm=(rollout.proposal_tracking_residual_norm.astype(np.float64)),
            replay_qpos=replay.qpos.astype(np.float64),
            replay_qvel=replay.qvel.astype(np.float64),
            replay_muscle_activation=replay.muscle_activation.astype(np.float64),
            replay_axial_rotation_position=replay.axial_rotation.position.astype(np.float64),
            replay_axial_rotation_velocity=replay.axial_rotation.velocity.astype(np.float64),
            replay_axial_rotation_root_yaw=replay.axial_rotation.root_yaw.astype(np.float64),
            replay_axial_rotation_root_xy=replay.axial_rotation.root_xy.astype(np.float64),
            replay_axial_rotation_initial_position=np.asarray(replay.axial_rotation.initial_position, dtype=np.float64),
            replay_axial_rotation_initial_root_yaw=np.asarray(replay.axial_rotation.initial_root_yaw, dtype=np.float64),
            replay_axial_rotation_initial_root_xy=replay.axial_rotation.initial_root_xy.astype(np.float64),
            replay_axial_rotation_signal_contract_json=np.asarray(
                json.dumps(replay.axial_rotation.contract.as_dict(), sort_keys=True, allow_nan=False)
            ),
            target_left_foot_floor_contact=target_contact_audit.left_foot_floor_contact.astype(np.bool_),
            target_right_foot_floor_contact=target_contact_audit.right_foot_floor_contact.astype(np.bool_),
            target_left_foot_floor_normal_force=target_contact_audit.left_foot_floor_normal_force.astype(np.float64),
            target_right_foot_floor_normal_force=target_contact_audit.right_foot_floor_normal_force.astype(np.float64),
            target_initial_left_foot_floor_contact=np.asarray(
                target_contact_audit.initial_left_foot_floor_contact, dtype=np.bool_
            ),
            target_initial_right_foot_floor_contact=np.asarray(
                target_contact_audit.initial_right_foot_floor_contact, dtype=np.bool_
            ),
            target_initial_left_foot_floor_normal_force=np.asarray(
                target_contact_audit.initial_left_foot_floor_normal_force, dtype=np.float64
            ),
            target_initial_right_foot_floor_normal_force=np.asarray(
                target_contact_audit.initial_right_foot_floor_normal_force, dtype=np.float64
            ),
            target_site_proxy_left_foot_contact=target_contact_audit.site_proxy_left_foot_contact.astype(np.bool_),
            target_site_proxy_right_foot_contact=target_contact_audit.site_proxy_right_foot_contact.astype(np.bool_),
            target_initial_site_proxy_left_foot_contact=np.asarray(
                target_contact_audit.initial_site_proxy_left_foot_contact, dtype=np.bool_
            ),
            target_initial_site_proxy_right_foot_contact=np.asarray(
                target_contact_audit.initial_site_proxy_right_foot_contact, dtype=np.bool_
            ),
            target_site_proxy_left_clearance=target_contact_audit.site_proxy_left_clearance.astype(np.float64),
            target_site_proxy_right_clearance=target_contact_audit.site_proxy_right_clearance.astype(np.float64),
            target_root_vertical_position=target_contact_audit.root_vertical_position.astype(np.float64),
            target_root_vertical_velocity=target_contact_audit.root_vertical_velocity.astype(np.float64),
            target_com_vertical_position=target_contact_audit.com_vertical_position.astype(np.float64),
            target_com_vertical_velocity=target_contact_audit.com_vertical_velocity.astype(np.float64),
            target_axial_rotation_position=target_contact_audit.axial_rotation.position.astype(np.float64),
            target_axial_rotation_velocity=target_contact_audit.axial_rotation.velocity.astype(np.float64),
            target_axial_rotation_root_yaw=target_contact_audit.axial_rotation.root_yaw.astype(np.float64),
            target_axial_rotation_root_xy=target_contact_audit.axial_rotation.root_xy.astype(np.float64),
            target_axial_rotation_initial_position=np.asarray(
                target_contact_audit.axial_rotation.initial_position, dtype=np.float64
            ),
            target_axial_rotation_initial_root_yaw=np.asarray(
                target_contact_audit.axial_rotation.initial_root_yaw, dtype=np.float64
            ),
            target_axial_rotation_initial_root_xy=target_contact_audit.axial_rotation.initial_root_xy.astype(
                np.float64
            ),
            target_axial_rotation_signal_contract_json=np.asarray(
                json.dumps(target_contact_audit.axial_rotation.contract.as_dict(), sort_keys=True, allow_nan=False)
            ),
            actual_left_foot_floor_contact=rollout.left_foot_floor_contact.astype(np.bool_),
            actual_right_foot_floor_contact=rollout.right_foot_floor_contact.astype(np.bool_),
            actual_left_foot_floor_normal_force=rollout.left_foot_floor_normal_force.astype(np.float64),
            actual_right_foot_floor_normal_force=rollout.right_foot_floor_normal_force.astype(np.float64),
            replay_left_foot_floor_contact=replay.left_foot_floor_contact.astype(np.bool_),
            replay_right_foot_floor_contact=replay.right_foot_floor_contact.astype(np.bool_),
            replay_left_foot_floor_normal_force=replay.left_foot_floor_normal_force.astype(np.float64),
            replay_right_foot_floor_normal_force=replay.right_foot_floor_normal_force.astype(np.float64),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_optimizer_artifact(path: Path, *, expected_fingerprint: str) -> dict[str, Any]:
    manifest_path = path / "optimizer_manifest.json"
    model_path = path / "runtime_model.mjb"
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("existing optimizer artifact is incomplete")
    payload = load_json_strict(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError("optimizer manifest must contain a JSON object")
    result = dict(payload)
    fingerprint = result.pop("optimizer_fingerprint", None)
    if fingerprint != expected_fingerprint or canonical_json_sha256(result) != expected_fingerprint:
        raise ValueError("existing optimizer manifest fingerprint mismatch")
    if file_sha256(model_path) != payload.get("model_artifact_sha256"):
        raise ValueError("existing optimizer runtime MJB hash mismatch")
    if any(not _is_sha256(str(payload.get(field, ""))) for field in _SHA256_FIELDS):
        raise ValueError("optimizer manifest contains malformed SHA-256 fields")
    return dict(payload)


def _validate_runtime_model_binding(
    payload: Mapping[str, Any] | None,
    *,
    model_hash: str,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("runtime model binding must be a mapping")
    result = dict(payload)
    if result.get("schema_version") != "chinajump_taskfactory_runtime_model_binding_v1":
        raise ValueError("unsupported runtime model binding schema_version")
    if result.get("production_eligible") is not True:
        raise ValueError("runtime model binding is not production eligible")
    if result.get("num_env_model_hash_invariant") is not True:
        raise ValueError("runtime model binding has not proven num_env model-hash invariance")
    if result.get("construction_model_hash") != model_hash:
        raise ValueError("runtime model binding differs from the supplied complete MjModel hash")
    if result.get("declared_num_env_model_hash") != model_hash:
        raise ValueError("declared-production and construction model hashes differ")
    for field in (
        "resolved_config_sha256",
        "resolved_model_params_sha256",
        "construction_model_hash",
        "declared_num_env_model_hash",
    ):
        if not _is_sha256(str(result.get(field, ""))):
            raise ValueError(f"runtime model binding {field} must be SHA-256")
    model_params = result.get("resolved_model_params")
    if not isinstance(model_params, Mapping) or canonical_json_sha256(model_params) != result.get(
        "resolved_model_params_sha256"
    ):
        raise ValueError("runtime model binding resolved model-parameter hash mismatch")
    return result


def _validate_runtime_model_provenance(
    payload: Mapping[str, Any] | None,
    *,
    model_hash: str,
    runtime_model_binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate reporting-only provenance without mutating the binding ABI."""

    if payload is None:
        return None
    if runtime_model_binding is None:
        raise ValueError("runtime model provenance requires a production runtime binding")
    if not isinstance(payload, Mapping):
        raise ValueError("runtime model provenance must be a mapping")
    result = dict(payload)
    common_fields = {
        "schema_version",
        "source_kind",
        "verified_runtime_artifact",
        "model_hash",
        "config_name",
        "hydra_overrides",
    }
    source_kind = result.get("source_kind")
    if source_kind == "taskfactory_constructed":
        _require_exact_fields(result, common_fields, "TaskFactory runtime model provenance")
        if result.get("verified_runtime_artifact") is not None:
            raise ValueError("constructed runtime provenance may not name a verified artifact")
    elif source_kind == "verified_runtime_artifact_reuse":
        _require_exact_fields(
            result,
            common_fields
            | {
                "current_resolved_config_sha256",
                "current_resolved_model_params_sha256",
            },
            "reused runtime model provenance",
        )
        artifact = result.get("verified_runtime_artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("reused runtime provenance must identify its verified artifact")
        artifact_record = dict(artifact)
        _require_exact_fields(
            artifact_record,
            {
                "path",
                "optimizer_fingerprint",
                "optimizer_manifest_sha256",
                "runtime_model_mjb_sha256",
            },
            "verified runtime artifact provenance",
        )
        artifact_path = Path(str(artifact_record["path"]))
        if (
            not artifact_path.is_absolute()
            or artifact_path.name != artifact_record["optimizer_fingerprint"]
            or not _is_sha256(str(artifact_record["optimizer_fingerprint"]))
        ):
            raise ValueError("verified runtime artifact provenance path/fingerprint is malformed")
        for field in ("optimizer_manifest_sha256", "runtime_model_mjb_sha256"):
            if not _is_sha256(str(artifact_record[field])):
                raise ValueError(f"verified runtime artifact provenance {field} must be SHA-256")
        _load_optimizer_artifact(
            artifact_path,
            expected_fingerprint=str(artifact_record["optimizer_fingerprint"]),
        )
        if file_sha256(artifact_path / "optimizer_manifest.json") != artifact_record["optimizer_manifest_sha256"]:
            raise ValueError("verified runtime artifact provenance manifest hash is stale")
        if file_sha256(artifact_path / "runtime_model.mjb") != artifact_record["runtime_model_mjb_sha256"]:
            raise ValueError("verified runtime artifact provenance MJB hash is stale")
        if result.get("current_resolved_config_sha256") != runtime_model_binding.get("resolved_config_sha256"):
            raise ValueError("runtime provenance resolved config hash differs from the binding")
        if result.get("current_resolved_model_params_sha256") != runtime_model_binding.get(
            "resolved_model_params_sha256"
        ):
            raise ValueError("runtime provenance model-parameter hash differs from the binding")
    else:
        raise ValueError("runtime model provenance source_kind is unsupported")

    if result.get("schema_version") != "primitive_runtime_model_provenance_v1":
        raise ValueError("unsupported runtime model provenance schema_version")
    if result.get("model_hash") != model_hash:
        raise ValueError("runtime model provenance differs from the supplied complete MjModel hash")
    if result.get("config_name") != runtime_model_binding.get("config_name"):
        raise ValueError("runtime model provenance config_name differs from the binding")
    overrides = result.get("hydra_overrides")
    if not isinstance(overrides, list) or overrides != runtime_model_binding.get("hydra_overrides"):
        raise ValueError("runtime model provenance Hydra overrides differ from the binding")
    return result


def _validate_policy_controller_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    required = {
        "schema_version",
        "checkpoint_content",
        "checkpoint_sha256",
        "teacher_action_mode",
    }
    _require_exact_fields(result, required, "full-action policy controller binding")
    if result["schema_version"] != "full_354_policy_controller_binding_v1":
        raise ValueError("unsupported full-action policy controller binding schema_version")
    if not _is_sha256(str(result["checkpoint_sha256"])):
        raise ValueError("policy controller binding checkpoint_sha256 must be SHA-256")
    checkpoint = result["checkpoint_content"]
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != result["checkpoint_sha256"]:
        raise ValueError("policy controller binding checkpoint content/fingerprint mismatch")
    if result["teacher_action_mode"] not in {"deterministic_mean", "stochastic_sample"}:
        raise ValueError("policy controller binding teacher action mode is unsupported")
    return result


def _validate_policy_rollout_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    required = {
        "schema_version",
        "physical_rollout_shard_sha256",
        "physical_rollout_metadata_sha256",
        "physical_control_sequence_sha256",
        "selected_rollout_uid",
        "source_motion_uid",
        "source_frame_interval",
        "transition_count",
        "collector_phase_id_match",
        "phase_schema_fingerprint",
        "policy_pre_state_tracking",
    }
    _require_exact_fields(result, required, "full-action policy rollout binding")
    if result["schema_version"] != "full_354_policy_physical_rollout_binding_v1":
        raise ValueError("unsupported full-action policy rollout binding schema_version")
    for field in (
        "physical_rollout_shard_sha256",
        "physical_rollout_metadata_sha256",
        "physical_control_sequence_sha256",
    ):
        if not _is_sha256(str(result[field])):
            raise ValueError(f"policy rollout binding {field} must be SHA-256")
    if type(result["transition_count"]) is not int or result["transition_count"] <= 0:
        raise ValueError("policy rollout binding transition_count must be positive")
    if type(result["collector_phase_id_match"]) is not bool:
        raise ValueError("policy rollout binding collector_phase_id_match must be boolean")
    if not _is_sha256(str(result["phase_schema_fingerprint"])):
        raise ValueError("policy rollout binding phase schema fingerprint must be SHA-256")
    if type(result["selected_rollout_uid"]) is not int or type(result["source_motion_uid"]) is not int:
        raise ValueError("policy rollout binding rollout/motion UIDs must be integers")
    interval = result["source_frame_interval"]
    if not isinstance(interval, Mapping) or set(interval) != {
        "start_frame",
        "end_frame_exclusive",
        "source_total_frames",
    }:
        raise ValueError("policy rollout binding source frame interval is malformed")
    tracking = result["policy_pre_state_tracking"]
    if not isinstance(tracking, Mapping) or any(
        not np.isfinite(float(tracking.get(field, np.nan)))
        for field in ("position_rmse", "velocity_rmse", "position_abs_max", "velocity_abs_max")
    ):
        raise ValueError("policy rollout binding pre-state tracking metrics are malformed")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_exact_fields(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{context} fields differ from schema: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _optimizer_config_from_args(args: argparse.Namespace) -> PhysicalOptimizerConfig:
    return PhysicalOptimizerConfig(
        horizon=args.horizon,
        population=args.population,
        elite_count=args.elite_count,
        iterations=args.iterations,
        initial_std=args.initial_std,
        min_std=args.min_std,
        elite_momentum=args.elite_momentum,
        position_weight=args.position_weight,
        velocity_weight=args.velocity_weight,
        p08_axial_position_weight=args.p08_axial_position_weight,
        p08_axial_velocity_weight=args.p08_axial_velocity_weight,
        p08_position_abs_weight=args.p08_position_abs_weight,
        p08_root_orientation_weight=args.p08_root_orientation_weight,
        effort_weight=args.effort_weight,
        rate_weight=args.rate_weight,
        terminal_weight=args.terminal_weight,
        initial_activation_margin=args.initial_activation_margin,
        initial_forward_regularization=args.initial_forward_regularization,
        initial_forward_solver_tolerance=args.initial_forward_solver_tolerance,
        initial_forward_solver_max_iterations=args.initial_forward_solver_max_iterations,
        shooting_finite_difference_step=args.shooting_finite_difference_step,
        shooting_solver_tolerance=args.shooting_solver_tolerance,
        shooting_solver_max_iterations=args.shooting_solver_max_iterations,
    ).validated()


def _semantic_qc_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_contact_normal_force": args.min_contact_normal_force,
        "max_bilateral_contact_lag_frames": args.max_bilateral_contact_lag_frames,
        "min_low_flight_frames": args.min_low_flight_frames,
        "min_precontact_air_frames": args.min_precontact_air_frames,
        "min_landing_stabilization_frames": args.min_landing_stabilization_frames,
        "min_ready_hold_frames": args.min_ready_hold_frames,
        "min_phase_transitions": args.min_phase_transitions,
        "min_ready_frames": args.min_ready_frames,
        "max_stable_root_vertical_speed": args.max_stable_root_vertical_speed,
        "max_ready_com_vertical_speed": args.max_ready_com_vertical_speed,
        "max_post_impact_com_vertical_speed": args.max_post_impact_com_vertical_speed,
        "max_ready_hold_com_vertical_speed": args.max_ready_hold_com_vertical_speed,
        "min_com_vertical_excursion": args.min_com_vertical_excursion,
        "max_axial_neutral_speed": args.max_axial_neutral_speed,
        "min_axial_rotation_excursion": args.min_axial_rotation_excursion,
        "min_axial_signed_monotonic_fraction": args.min_axial_signed_monotonic_fraction,
        "max_axial_recenter_error": args.max_axial_recenter_error,
        "max_axial_root_yaw_excursion": args.max_axial_root_yaw_excursion,
        "max_axial_root_xy_displacement": args.max_axial_root_xy_displacement,
        "site_contact_baseline_quantile": args.site_contact_baseline_quantile,
        "site_contact_enter_height": args.site_contact_enter_height,
        "site_contact_exit_height": args.site_contact_exit_height,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--source-motion-path", required=True)
    parser.add_argument("--phase-schema", type=Path, required=True)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--controller-store", type=Path, required=True)
    parser.add_argument("--target-skill-id", default="ChinaJump")
    parser.add_argument(
        "--config-name",
        default="config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2",
        help=(
            "Fullbody-relative ChinaJump Hydra config used to construct the exact TaskFactory model "
            "or validate a verified runtime artifact's immutable identity."
        ),
    )
    parser.add_argument("--hydra-override", action="append", default=[])
    parser.add_argument(
        "--verified-runtime-artifact",
        type=Path,
        help=(
            "Fingerprint-named controller artifact whose verified MJB should be loaded CPU-only; "
            "the default constructs the exact TaskFactory runtime."
        ),
    )
    parser.add_argument("--exclude-target-motion", action="append", default=[])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame-exclusive", type=int)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--initial-std", type=float, default=0.12)
    parser.add_argument("--min-std", type=float, default=0.015)
    parser.add_argument("--elite-momentum", type=float, default=0.25)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.02)
    parser.add_argument(
        "--p08-axial-position-weight",
        type=float,
        default=0.0,
        help="P08-only squared error weight on the sum of the six named axial hinge positions.",
    )
    parser.add_argument(
        "--p08-axial-velocity-weight",
        type=float,
        default=0.0,
        help="P08-only squared error weight on the sum of the six named axial hinge velocities.",
    )
    parser.add_argument(
        "--p08-position-abs-weight",
        type=float,
        default=0.0,
        help="P08-only soft maximum-position-error weight used to rank CEM plans.",
    )
    parser.add_argument(
        "--p08-root-orientation-weight",
        type=float,
        default=0.0,
        help="P08-only mean squared root free-joint orientation-error weight.",
    )
    parser.add_argument("--effort-weight", type=float, default=1.0e-4)
    parser.add_argument("--rate-weight", type=float, default=5.0e-4)
    parser.add_argument("--terminal-weight", type=float, default=2.0)
    parser.add_argument("--initial-activation-margin", type=float, default=0.01)
    parser.add_argument("--initial-forward-regularization", type=float, default=1.0e-3)
    parser.add_argument("--initial-forward-solver-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--initial-forward-solver-max-iterations", type=int, default=1000)
    parser.add_argument("--shooting-finite-difference-step", type=float, default=0.05)
    parser.add_argument("--shooting-solver-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--shooting-solver-max-iterations", type=int, default=1000)
    parser.add_argument("--min-contact-normal-force", type=float, default=1.0e-6)
    parser.add_argument("--max-bilateral-contact-lag-frames", type=int, default=5)
    parser.add_argument("--min-low-flight-frames", type=int, default=2)
    parser.add_argument("--min-precontact-air-frames", type=int, default=2)
    parser.add_argument("--min-landing-stabilization-frames", type=int, default=10)
    parser.add_argument("--min-ready-hold-frames", type=int, default=10)
    parser.add_argument("--min-phase-transitions", type=int, default=2)
    parser.add_argument("--min-ready-frames", type=int, default=5)
    parser.add_argument("--max-stable-root-vertical-speed", type=float, default=0.20)
    parser.add_argument("--max-ready-com-vertical-speed", type=float, default=0.15)
    parser.add_argument("--max-post-impact-com-vertical-speed", type=float, default=0.20)
    parser.add_argument("--max-ready-hold-com-vertical-speed", type=float, default=0.15)
    parser.add_argument("--min-com-vertical-excursion", type=float, default=0.03)
    parser.add_argument("--max-axial-neutral-speed", type=float, default=0.60)
    parser.add_argument("--min-axial-rotation-excursion", type=float, default=0.12)
    parser.add_argument("--min-axial-signed-monotonic-fraction", type=float, default=0.90)
    parser.add_argument("--max-axial-recenter-error", type=float, default=0.01)
    parser.add_argument("--max-axial-root-yaw-excursion", type=float, default=0.35)
    parser.add_argument("--max-axial-root-xy-displacement", type=float, default=0.25)
    parser.add_argument("--site-contact-baseline-quantile", type=float, default=0.01)
    parser.add_argument("--site-contact-enter-height", type=float, default=0.035)
    parser.add_argument("--site-contact-exit-height", type=float, default=0.045)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce actual physical-control primitive trials with fail-closed QC."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Validate the exact 354-D runtime and target only.")
    _add_common_arguments(preflight)

    def add_rollout_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--output-dir", type=Path, required=True)
        command_parser.add_argument("--trial-id", required=True)
        command_parser.add_argument("--seed", type=int, required=True)
        command_parser.add_argument("--max-position-rmse", type=float, required=True)
        command_parser.add_argument("--max-velocity-rmse", type=float, required=True)
        command_parser.add_argument("--max-position-abs", type=float, required=True)
        command_parser.add_argument("--max-velocity-abs", type=float, required=True)
        command_parser.add_argument("--max-saturation-fraction", type=float, default=0.98)
        command_parser.add_argument("--replay-position-atol", type=float, default=1.0e-10)
        command_parser.add_argument("--replay-velocity-atol", type=float, default=1.0e-10)
        command_parser.add_argument("--replay-activation-atol", type=float, default=1.0e-10)
        command_parser.add_argument("--replay-contact-force-atol", type=float, default=1.0e-10)

    produce = commands.add_parser("produce", help="Optimize, roll out, replay, and publish one trial.")
    _add_common_arguments(produce)
    add_rollout_arguments(produce)
    produce.add_argument("--canonical-control-artifact", type=Path)
    import_policy = commands.add_parser(
        "import-policy",
        help="CPU-forward actual physical controls captured from an independent full-354 policy.",
    )
    _add_common_arguments(import_policy)
    add_rollout_arguments(import_policy)
    import_policy.add_argument("--physical-rollout-shard", type=Path, required=True)
    import_policy.add_argument("--physical-rollout-metadata", type=Path, required=True)
    import_policy.add_argument("--teacher-checkpoint", type=Path, required=True)
    import_policy.add_argument(
        "--rollout-uid",
        type=int,
        help="Required only when the physical shard contains more than one rollout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    optimizer_config = _optimizer_config_from_args(args)
    if args.command == "preflight":
        preflight_qc_config = RolloutQCConfig(
            max_position_rmse=1.0,
            max_velocity_rmse=1.0,
            max_position_abs=1.0,
            max_velocity_abs=1.0,
            **_semantic_qc_kwargs_from_args(args),
        ).validated()
        report = preflight_physical_primitive_target(
            source_npz=args.source_npz,
            source_motion_path=args.source_motion_path,
            phase_schema_path=args.phase_schema,
            phase_plan_path=args.phase_plan,
            controller_store=args.controller_store,
            optimizer_config=optimizer_config,
            qc_config=preflight_qc_config,
            target_skill_id=args.target_skill_id,
            excluded_target_motion_paths=args.exclude_target_motion,
            start_frame=args.start_frame,
            end_frame_exclusive=args.end_frame_exclusive,
            config_name=args.config_name,
            hydra_overrides=args.hydra_override,
            verified_runtime_artifact=args.verified_runtime_artifact,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 0 if bool(report["preflight_passed"]) else 2

    runtime = resolve_chinajump_runtime_model(
        config_name=args.config_name,
        hydra_overrides=args.hydra_override,
        verified_runtime_artifact=args.verified_runtime_artifact,
    )
    model = runtime.model
    target = load_retargeted_motion_target(
        args.source_npz,
        model=model,
        source_motion_path=args.source_motion_path,
        phase_schema_path=args.phase_schema,
        phase_plan_path=args.phase_plan,
        target_skill_id=args.target_skill_id,
        excluded_target_motion_paths=args.exclude_target_motion,
        start_frame=args.start_frame,
        end_frame_exclusive=args.end_frame_exclusive,
    )
    qc_config = RolloutQCConfig(
        max_position_rmse=args.max_position_rmse,
        max_velocity_rmse=args.max_velocity_rmse,
        max_position_abs=args.max_position_abs,
        max_velocity_abs=args.max_velocity_abs,
        max_saturation_fraction=args.max_saturation_fraction,
        replay_position_atol=args.replay_position_atol,
        replay_velocity_atol=args.replay_velocity_atol,
        replay_activation_atol=args.replay_activation_atol,
        replay_contact_force_atol=args.replay_contact_force_atol,
        **_semantic_qc_kwargs_from_args(args),
    ).validated()
    planner: PhysicalControlPlanner | None = None
    controller_binding: Mapping[str, Any] | None = None
    policy_rollout_binding: Mapping[str, Any] | None = None
    canonical_control_binding: Mapping[str, Any] | None = None
    if args.command == "import-policy":
        imported = load_full_action_policy_controls(
            args.physical_rollout_shard,
            metadata_path=args.physical_rollout_metadata,
            teacher_checkpoint=args.teacher_checkpoint,
            model=model,
            target=target,
            rollout_uid=args.rollout_uid,
        )
        planner = imported.planner
        controller_binding = imported.controller_binding
        policy_rollout_binding = imported.rollout_binding
    elif args.canonical_control_artifact is not None:
        canonical_control_binding = load_canonical_control_artifact(
            args.canonical_control_artifact, expected_width=int(model.nu)
        )
    result = produce_primitive_trial(
        model,
        target,
        output_dir=args.output_dir,
        controller_store=args.controller_store,
        trial_id=args.trial_id,
        optimizer_config=optimizer_config,
        qc_config=qc_config,
        seed=args.seed,
        planner=planner,
        controller_binding=controller_binding,
        runtime_model_binding=runtime.binding,
        runtime_model_provenance=runtime.provenance,
        policy_rollout_binding=policy_rollout_binding,
        canonical_control_binding=canonical_control_binding,
    )
    summary = {
        "success": result.success,
        "output_dir": str(result.output_dir),
        "trial_path": None if result.trial_path is None else str(result.trial_path),
        "controller_dir": str(result.controller_dir),
        "controller_fingerprint": result.controller_fingerprint,
        "rollout_fingerprint": result.rollout_fingerprint,
        "runtime_model_provenance": runtime.provenance,
        "qc": result.qc,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
