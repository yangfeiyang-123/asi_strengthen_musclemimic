import logging
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from loco_mujoco.core.utils import mj_jntid2qposid, mj_jntid2qvelid, mj_jntname2qposid, mj_jntname2qvelid
from loco_mujoco.core.utils.math import (
    calc_site_velocities,
    calculate_relative_site_quantities,
    quat_scalarfirst2scalarlast,
    quaternion_angular_distance,
)
from musclemimic.core.utils.site_mapping import create_site_mapper
from musclemimic.physiology.anatomical_groups import (
    load_anatomical_taxonomy,
    validate_taxonomy_against_model,
)
from musclemimic.physiology.continuity_groups import (
    build_fascicle_continuity_spec,
    load_fascicle_continuity_graph,
    resolve_fascicle_continuity_reward_gate,
    validate_continuity_graph_against_model,
)
from musclemimic.physiology.intra_muscle import (
    ordered_body_activation,
    robust_fascicle_continuity,
)
from musclemimic.physiology.runtime_binding import (
    resolve_muscle_activation_addresses,
    resolve_ordered_policy_muscle_layout,
)
from musclemimic.utils.finger_isolation import finger_joint_side

_LOGGER = logging.getLogger(__name__)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """

    def wrapper(self, *args, **kwargs):
        env = kwargs.get("env", None) if "env" in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)

    return wrapper


def quat_to_yaw(quat, backend):
    """Extract yaw from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return backend.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _select_reference_coordinates(
    traj_no,
    subtraj_step_no,
    effective_ref_stride,
    num_frames,
    *,
    is_bank: bool,
    backend,
):
    """Select a JIT-safe trajectory/frame pair for single or banked caches."""

    if is_bank:
        trajectory = backend.asarray(traj_no).astype(backend.int32)
        stride = effective_ref_stride[trajectory]
        frame_count = num_frames[trajectory]
    else:
        trajectory = None
        stride = effective_ref_stride
        frame_count = num_frames
    frame = backend.clip(
        backend.round(subtraj_step_no * stride).astype(backend.int32),
        0,
        frame_count - 1,
    )
    return trajectory, frame


def _ordered_muscle_activation_addresses(model: MjModel) -> np.ndarray:
    """Resolve MuJoCo muscle activation state without assuming id alignment.

    ``data.act`` is indexed by activation-state address, not actuator id.  A
    future mixed actuator model may therefore have ``nu != na`` and gaps in the
    mapping.  The reward deliberately accepts only native one-state muscle
    actuators and fails closed on an unsupported dynamics layout.
    """

    return resolve_muscle_activation_addresses(model)


def _plain_config_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise ValueError("intra_muscle_consistency must be a mapping")


def _finite_config_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"intra_muscle_consistency.{field} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"intra_muscle_consistency.{field} must be finite numeric")
    return result


def _positive_config_float(value: Any, field: str) -> float:
    result = _finite_config_float(value, field)
    if result <= 0.0:
        raise ValueError(f"intra_muscle_consistency.{field} must be positive")
    return result


def _required_contract_path(value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required when continuity is enabled")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = _REPOSITORY_ROOT / path
    return path.resolve(strict=True)


class TrajectoryBasedReward(Reward):
    """
    Base class for trajectory-based reward functions. These reward functions require a
    trajectory handler to compute the reward.

    """

    @property
    def requires_trajectory(self) -> bool:
        return True


class TargetVelocityTrajReward(TrajectoryBasedReward):
    """
    Reward function that computes the reward based on the deviation from the trajectory velocity. The trajectory
    velocity is provided as an observation in the environment. The reward is computed as the negative exponential
    of the squared difference between the current velocity and the goal velocity. The reward is computed for the
    x, y, and yaw velocities of the root.

    """

    def __init__(self, env: Any, w_exp=10.0, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)

        if float(kwargs.get("body_graph_w_sum", 0.0)) != 0.0:
            raise ValueError(
                "body_graph_w_sum must remain 0: the online body-graph Laplacian reward is not implemented"
            )
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
    def __call__(
        self,
        state: Union[np.ndarray, jnp.ndarray],
        action: Union[np.ndarray, jnp.ndarray],
        next_state: Union[np.ndarray, jnp.ndarray],
        absorbing: bool,
        info: Dict[str, Any],
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[float, Any]:
        """
        Computes a tracking reward based on the deviation from the trajectory velocity.
        Tracking is done on the x, y, and yaw velocities of the root.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        def calc_local_vel(_d):
            _lin_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[:3]
            _ang_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[3:]
            _root_quat = R.from_quat(
                quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7])
            )
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        traj_data = env.th.get_current_traj_data(carry, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp * backend.mean(backend.square(vel_local - traj_vel_local)))

        # set nan values to 0
        tracking_reward = backend.nan_to_num(tracking_reward, nan=0.0)

        reward_info = {"reward_total": tracking_reward}
        return tracking_reward, carry, reward_info


@struct.dataclass
class MimicRewardState:
    """
    State of MimicReward.
    """

    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]
    imitation_error_total: float = 0.0  # Raw weighted sum of distances for adaptive sampling
    last_foot_xpos: Union[np.ndarray, jnp.ndarray, None] = None


class MimicReward(TrajectoryBasedReward):
    """
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
    computed as the negative exponential of the squared difference between the current state and the trajectory state.
    The reward is computed for the joint positions, joint velocities, root position, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.

    """

    def __init__(
        self,
        env: Any,
        sites_for_mimic=None,
        joints_for_mimic=None,
        exclude_finger_joints=False,
        absolute_site_reward_sites=None,
        absolute_site_w_sum=0.0,
        absolute_site_w_exp=10.0,
        **kwargs,
    ):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None, taking all.
            joints_for_mimic (List[str], optional): List of joint names to mimic. Defaults to None, taking all.
            exclude_finger_joints (bool): Remove all name-identified right/left
                finger joints from qpos/qvel imitation terms. This is required
                when finger state is treated as a nuisance variable by the body
                policy. Wrist and forearm joints remain included.
            absolute_site_reward_sites (List[str], optional): Site names for optional absolute position reward.
            absolute_site_w_sum (float, optional): Weight for optional absolute site reward. Defaults to 0.0.
            absolute_site_w_exp (float, optional): Exponential weight for optional absolute site reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        # reward coefficients
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._root_pos_w_exp = kwargs.get("root_pos_w_exp", 10.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._root_pos_w_sum = kwargs.get("root_pos_w_sum", 0.0)
        # Dedicated global-root terms.  The generic qpos/qvel terms mix the
        # six root DoFs with every articulated joint, while root-relative site
        # rewards are invariant to a globally spinning pelvis.  Keep these
        # disabled by default so existing checkpoints/configs are unchanged.
        self._root_orientation_w_exp = kwargs.get("root_orientation_w_exp", 8.0)
        self._root_orientation_w_sum = kwargs.get("root_orientation_w_sum", 0.0)
        self._root_ang_vel_w_exp = kwargs.get("root_ang_vel_w_exp", 0.5)
        self._root_ang_vel_w_sum = kwargs.get("root_ang_vel_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._action_saturation_coeff = kwargs.get("action_saturation_coeff", 0.0)
        self._action_saturation_margin_fraction = kwargs.get("action_saturation_margin_fraction", 0.02)
        if not 0.0 < self._action_saturation_margin_fraction < 0.5:
            raise ValueError("action_saturation_margin_fraction must be in the open interval (0, 0.5)")
        self._activation_energy_coeff = kwargs.get("activation_energy_coeff", 0.0)
        self._muscle_activation_addresses = _ordered_muscle_activation_addresses(env._model)
        self._configure_fascicle_continuity(
            env,
            kwargs.get("intra_muscle_consistency"),
        )
        # Root velocity tracking: [vx_local, vy_local, yaw_rate]
        self._root_vel_w_exp = kwargs.get("root_vel_w_exp", 10.0)
        self._root_vel_w_sum = kwargs.get("root_vel_w_sum", 0.2)
        self._absolute_site_w_sum = absolute_site_w_sum
        self._absolute_site_w_exp = absolute_site_w_exp
        self._absolute_site_names = list(absolute_site_reward_sites or [])

        # Contact tracking reward exponential scales (weights come from carry)
        self._foot_contact_height_w_exp = kwargs.get("foot_contact_height_w_exp", 80.0)
        self._foot_contact_velocity_w_exp = kwargs.get("foot_contact_velocity_w_exp", 8.0)
        self._body_graph_w_exp = kwargs.get("body_graph_w_exp", 20.0)
        self._contact_tracking_data = None
        self._foot_site_ids = None
        # Pre-converted JAX arrays (set by attach_contact_tracking); None when contact tracking is disabled
        self._ctd_stance_mask = None
        self._ctd_foot_z = None
        self._ctd_eff_stride = None
        self._ctd_num_frames = None
        self._ctd_body_laplacian = None
        self._ctd_phase_global = None
        self._ctd_phase_id = None
        self._ctd_phase_local = None
        self._ctd_time_to_impact_s = None
        self._ctd_time_from_impact_s = None
        self._ctd_impact_flag = None
        self._ctd_racket_position_world = None
        self._ctd_racket_quaternion_world = None
        self._ctd_racket_linear_velocity_world = None
        self._ctd_racket_angular_velocity_world = None
        self._ctd_stringbed_normal_world = None
        self._ctd_stringbed_center_world = None
        self._ctd_racket_reference_confidence = None
        self._ctd_racket_reference_source = None
        self._ctd_reference_bundle_content_fingerprint = None
        self._ctd_is_bank = False
        self._ctd_num_trajectories = 1
        self._ctd_event_reference_bank_fingerprint = None

        # Parallel environment reward calculation mode
        # True: use mean(exp(-beta * dist)) - better for parallel environments
        # False: use exp(-beta * mean(dist)) - current behavior (backward compatible)
        self._use_mean_exp_reward = kwargs.get("use_mean_exp_reward", False)
        self._exclude_finger_joints = bool(exclude_finger_joints)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._right_hand_rel_index = (
            list(rel_site_names).index("right_hand_mimic") if "right_hand_mimic" in rel_site_names else None
        )
        self._rel_site_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in rel_site_names]
        )
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])
        self._absolute_site_ids = np.array([], dtype=int)
        if self._absolute_site_w_sum > 0.0:
            if not self._absolute_site_names:
                raise ValueError("absolute site reward sites must be provided when absolute_site_w_sum > 0")
            absolute_site_ids = []
            for site_name in self._absolute_site_names:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if site_id < 0:
                    raise ValueError(f"absolute site reward site not found: {site_name}")
                absolute_site_ids.append(site_id)
            self._absolute_site_ids = np.array(absolute_site_ids, dtype=int)

        # determine qpos and qvel indices
        quat_in_qpos = []
        qpos_ind = []
        qvel_ind = []
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if self._exclude_finger_joints and finger_joint_side(jnt_name) is not None:
                continue
            if joints_for_mimic is None or jnt_name in joints_for_mimic:
                qposid = mj_jntid2qposid(i, model)
                qvelid = mj_jntid2qvelid(i, model)
                qpos_ind.append(qposid)
                qvel_ind.append(qvelid)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    quat_in_qpos.append(qposid[3:])
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)

        # Handle case where there are no free joints (e.g., bimanual models)
        if len(quat_in_qpos) > 0:
            quat_in_qpos = np.concatenate(quat_in_qpos)
        else:
            quat_in_qpos = np.array([], dtype=int)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

        # calc mask for the root free joint velocities (handle case where it doesn't exist)
        self._free_joint_qpos_ind = None
        self._joint_qpos_mask = np.ones(len(self._qpos_ind), dtype=bool)
        self._joint_qvel_mask = np.ones(len(self._qvel_ind), dtype=bool)
        try:
            self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
            self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
            self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
            self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
            # Masks for excluding root from joint errors
            self._joint_qpos_mask = ~np.isin(self._qpos_ind, self._free_joint_qpos_ind)
            self._joint_qvel_mask = ~np.isin(self._qvel_ind, self._free_joint_qvel_ind)
        except (KeyError, ValueError):
            # For bimanual models without a free joint, create empty mask
            self._free_joint_qvel_ind = np.array([], dtype=int)
            self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)

        # Initialize site mapper for trajectory index mapping
        env_sites_for_mimic = getattr(env, "sites_for_mimic", [])
        traj_site_names = env.th.traj.info.site_names if (hasattr(env, "th") and env.th is not None) else None
        self._site_mapper = create_site_mapper(model, env.__class__.__name__, env_sites_for_mimic, traj_site_names)

        # Root XY indices for offset correction. When episodes start at random XY positions,
        # trajectory qpos is in world frame while simulation resets to origin. We subtract
        # the init XY offset so qpos values are compared in local frame.
        self._root_qpos_ids_xy = None
        self._root_xy_in_qpos_ind = None
        root_joint_name = self._info_props.get("root_free_joint_xml_name")
        if root_joint_name:
            try:
                root_qpos_ids = np.array(mj_jntname2qposid(root_joint_name, model))
                if root_qpos_ids.size >= 2:
                    self._root_qpos_ids_xy = root_qpos_ids[:2]
                    xy_in_ind = np.where(np.isin(self._qpos_ind, self._root_qpos_ids_xy))[0]
                    if xy_in_ind.size == 2:
                        self._root_xy_in_qpos_ind = xy_in_ind
            except Exception:
                pass

    def _configure_fascicle_continuity(self, env: Any, raw_config: Any) -> None:
        """Resolve all continuity files and static arrays before entering JIT."""

        config = _plain_config_mapping(raw_config)
        mode = str(config.get("mode", "off")).strip().lower()
        if mode not in {"off", "diagnostics", "reward"}:
            raise ValueError("intra_muscle_consistency.mode must be off, diagnostics, or reward")
        coefficient = _finite_config_float(config.get("coefficient", 0.0), "coefficient")
        if mode in {"off", "diagnostics"} and coefficient != 0.0:
            raise ValueError(f"intra_muscle_consistency mode={mode} requires coefficient=0")
        if mode == "reward" and coefficient <= 0.0:
            raise ValueError("intra_muscle_consistency reward mode requires coefficient > 0")

        self._fascicle_continuity_mode = mode
        self._fascicle_continuity_compute = mode != "off"
        self._fascicle_continuity_reward_active = mode == "reward"
        self._fascicle_continuity_coefficient = coefficient
        self._fascicle_continuity_scale = _positive_config_float(
            config.get("scale", 0.05),
            "scale",
        )
        self._fascicle_continuity_huber_delta = _positive_config_float(
            config.get("huber_delta", 1.0),
            "huber_delta",
        )
        raw_clip = config.get("raw_penalty_clip")
        self._fascicle_continuity_raw_penalty_clip = (
            None if raw_clip is None else _positive_config_float(raw_clip, "raw_penalty_clip")
        )
        self._fascicle_continuity_spec = None
        self._fascicle_continuity_reward_spec = None
        self._fascicle_continuity_measured_chain_count = 0
        self._fascicle_continuity_measured_edge_count = 0
        if mode == "off":
            return

        if str(config.get("signal", "activation")) != "activation":
            raise ValueError("fascicle continuity primary signal must be activation")
        if str(config.get("method", "robust_fascicle_continuity_v1")) != "robust_fascicle_continuity_v1":
            raise ValueError("unsupported intra_muscle_consistency method")
        compatibility = str(config.get("runtime_compatibility", "portable_muscle_channel_abi"))
        taxonomy_path = _required_contract_path(
            config.get("taxonomy_path"),
            "intra_muscle_consistency.taxonomy_path",
        )
        continuity_path = _required_contract_path(
            config.get("continuity_path"),
            "intra_muscle_consistency.continuity_path",
        )
        taxonomy = load_anatomical_taxonomy(taxonomy_path)
        if taxonomy.stable_model_binding["target"] != {
            "environment": "MyoFullBody",
            "disable_fingers": True,
            "expected_action_dim": 354,
        }:
            raise ValueError("online continuity requires the no-finger MyoFullBody 354 taxonomy")
        validate_taxonomy_against_model(
            taxonomy,
            env._model,
            compatibility=compatibility,
        )
        policy_layout = resolve_ordered_policy_muscle_layout(env, model=env._model)
        if policy_layout.actuator_names != taxonomy.actuator_names:
            raise ValueError("policy action order differs from the continuity taxonomy")
        graph = load_fascicle_continuity_graph(continuity_path, taxonomy=taxonomy)
        if graph.taxonomy_binding["runtime_compatibility"] != compatibility:
            raise ValueError("continuity graph runtime compatibility differs from reward config")
        validate_continuity_graph_against_model(graph, taxonomy, env._model)

        expected_taxonomy = config.get("expected_taxonomy_fingerprint")
        expected_graph = config.get("expected_continuity_fingerprint")
        if expected_taxonomy is not None and str(expected_taxonomy) != taxonomy.fingerprint:
            raise ValueError("configured expected taxonomy fingerprint differs from loaded taxonomy")
        if expected_graph is not None and str(expected_graph) != graph.graph_fingerprint:
            raise ValueError("configured expected continuity fingerprint differs from loaded graph")
        if mode == "reward" and (not expected_taxonomy or not expected_graph):
            raise ValueError("reward mode requires pinned taxonomy and continuity fingerprints")
        require_verified = config.get("require_verified_training_chains", True)
        if not isinstance(require_verified, bool):
            raise ValueError("require_verified_training_chains must be boolean")
        resolve_fascicle_continuity_reward_gate(
            graph,
            enabled=mode == "reward",
            require_verified_training_chains=require_verified,
        )
        if mode == "reward":
            expected_calibration = str(config.get("expected_calibration_fingerprint", "") or "").strip()
            generation = getattr(graph, "generation", None)
            promotion = None if not isinstance(generation, Mapping) else generation.get("training_promotion")
            if not expected_calibration:
                raise ValueError("reward mode requires a pinned calibration fingerprint")
            if not isinstance(promotion, Mapping):
                raise ValueError("reward graph lacks training-promotion calibration evidence")
            if promotion.get("calibration_fingerprint") != expected_calibration:
                raise ValueError("reward graph calibration differs from the pinned config")
            try:
                calibrated_coefficient = float(promotion["selected_reward_coefficient"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("reward graph lacks a calibrated coefficient") from error
            if not np.isclose(
                coefficient,
                calibrated_coefficient,
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError("reward coefficient differs from graph promotion evidence")
        self._fascicle_continuity_spec = build_fascicle_continuity_spec(
            graph,
            taxonomy,
        )
        self._fascicle_continuity_measured_chain_count = len(self._fascicle_continuity_spec.chain_ids)
        self._fascicle_continuity_measured_edge_count = int(
            np.sum(np.asarray(self._fascicle_continuity_spec.edge_mask))
        )
        if mode == "reward":
            self._fascicle_continuity_reward_spec = build_fascicle_continuity_spec(
                graph,
                taxonomy,
                training_enabled_only=True,
            )
        _LOGGER.info(
            "Fascicle continuity %s: graph=%s fingerprint=%s chains=%d edges=%d training_chains=%d",
            mode,
            graph.graph_id,
            graph.graph_fingerprint,
            self._fascicle_continuity_measured_chain_count,
            self._fascicle_continuity_measured_edge_count,
            graph.training_enabled_chain_count,
        )

    def attach_contact_tracking(self, contact_data, foot_site_names, model):
        """Attach contact tracking data for contact-preserving reward terms.

        Pre-converts numpy arrays to JAX for JIT-safe indexing inside __call__.
        """
        n_foot_sites = len(foot_site_names)
        self._ctd_is_bank = np.asarray(contact_data.stance_mask).ndim == 3
        n_cache_feet = contact_data.foot_points.shape[2 if self._ctd_is_bank else 1]
        if n_foot_sites != n_cache_feet:
            raise ValueError(
                f"foot_sites count ({n_foot_sites}) does not match "
                f"tracking cache foot count ({n_cache_feet}). "
                f"Config sites: {foot_site_names}, cache labels: {contact_data.foot_labels}"
            )
        self._contact_tracking_data = contact_data
        self._foot_site_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in foot_site_names], dtype=np.int32
        )
        self._ctd_stance_mask = jnp.asarray(contact_data.stance_mask, dtype=jnp.float32)
        # ``foot_points`` is either [T, F, 3] for a single reference or
        # [N, T, F, 3] for an exact-order multi-motion bank.  Ellipsis keeps
        # every leading axis and always selects the Cartesian z coordinate.
        self._ctd_foot_z = jnp.asarray(contact_data.foot_points[..., 2], dtype=jnp.float32)
        self._ctd_eff_stride = jnp.float32(contact_data.effective_ref_stride)
        self._ctd_num_frames = jnp.int32(contact_data.num_frames)
        if contact_data.body_laplacian is not None:
            self._ctd_body_laplacian = jnp.asarray(contact_data.body_laplacian, dtype=jnp.float32)
        else:
            self._ctd_body_laplacian = None
        for field, dtype in (
            ("phase_global", jnp.float32),
            ("phase_id", jnp.int16),
            ("phase_local", jnp.float32),
            ("time_to_impact_s", jnp.float32),
            ("time_from_impact_s", jnp.float32),
            ("impact_flag", jnp.bool_),
            ("racket_position_world", jnp.float32),
            ("racket_quaternion_world", jnp.float32),
            ("racket_linear_velocity_world", jnp.float32),
            ("racket_angular_velocity_world", jnp.float32),
            ("stringbed_normal_world", jnp.float32),
            ("stringbed_center_world", jnp.float32),
            ("racket_reference_confidence", jnp.float32),
        ):
            value = getattr(contact_data, field, None)
            setattr(
                self,
                f"_ctd_{field}",
                None if value is None else jnp.asarray(value, dtype=dtype),
            )
        self._ctd_racket_reference_source = getattr(contact_data, "racket_reference_source", None)
        self._ctd_reference_bundle_content_fingerprint = getattr(
            contact_data, "reference_bundle_content_fingerprint", None
        )
        self._ctd_num_trajectories = int(getattr(contact_data, "num_trajectories", 1))
        self._ctd_event_reference_bank_fingerprint = getattr(contact_data, "event_reference_bank_fingerprint", None)

    def init_state(
        self, env: Any, key: Any, model: Union[MjModel, Model], data: Union[MjData, Data], backend: ModuleType
    ):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            MimicRewardState: The initialized reward state.

        """
        foot_xpos = None
        if self._foot_site_ids is not None:
            foot_xpos = data.site_xpos[self._foot_site_ids]
        return MimicRewardState(
            last_qvel=data.qvel,
            last_action=backend.zeros(env.info.action_space.shape[0]),
            last_foot_xpos=foot_xpos,
        )

    def reset(self, env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any, backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    @check_traj_provided
    def __call__(
        self,
        state: Union[np.ndarray, jnp.ndarray],
        action: Union[np.ndarray, jnp.ndarray],
        next_state: Union[np.ndarray, jnp.ndarray],
        absorbing: bool,
        info: Dict[str, Any],
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[float, Any]:
        """
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        # Ensure site mapper matches actual trajectory order (handles creation-before-trajectory case)
        if self._site_mapper.requires_mapping and hasattr(env, "th") and env.th is not None:
            # Always attach (idempotent) to avoid stale mappings
            self._site_mapper.attach_trajectory_sites(env.th.traj.info.site_names)

        # get current reward state
        reward_state = carry.reward_state

        # Get dynamic reward weights from carry (set by reward curriculum)
        # This allows the curriculum to adjust velocity reward weights during training
        qvel_w_sum = carry.qvel_w_sum
        root_vel_w_sum = carry.root_vel_w_sum

        # get trajectory data
        # get all quantities from trajectory
        traj_data_single = env.th.get_current_traj_data(carry, backend)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._qpos_ind], traj_data_single.qvel[self._qvel_ind]

        # Subtract init XY offset from trajectory qpos to compare in local frame
        xy_offset = None
        if self._root_qpos_ids_xy is not None and hasattr(carry.traj_state, "subtraj_step_no_init"):
            init_data = env.th.get_traj_data_at(
                carry.traj_state.traj_no, carry.traj_state.subtraj_step_no_init, carry, backend
            )
            xy_offset = init_data.qpos[self._root_qpos_ids_xy]
            if self._root_xy_in_qpos_ind is not None:
                if backend == np:
                    qpos_traj = qpos_traj.copy()
                    qpos_traj[self._root_xy_in_qpos_ind] -= xy_offset
                else:
                    qpos_traj = qpos_traj.at[self._root_xy_in_qpos_ind].add(-xy_offset)

        # Handle quaternion joints if they exist
        qpos_quat_traj = qpos_traj[self._quat_in_qpos]
        if qpos_quat_traj.size > 0:
            qpos_quat_traj = qpos_quat_traj.reshape(-1, 4)

        if len(self._rel_site_ids) > 1:
            # For trajectory data access, use trajectory indices to handle memory-optimized environments
            if self._site_mapper.requires_mapping:
                traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            else:
                traj_site_indices = None

            site_rpos_traj, site_rangles_traj, site_rvel_traj = calculate_relative_site_quantities(
                traj_data_single,
                self._rel_site_ids,
                self._rel_body_ids,
                model.body_rootid,
                backend,
                trajectory_site_indices=traj_site_indices,
            )

        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]

        # Handle quaternion joints if they exist
        qpos_quat = qpos[self._quat_in_qpos]
        if qpos_quat.size > 0:
            qpos_quat = qpos_quat.reshape(-1, 4)

        if len(self._rel_site_ids) > 1:
            # For MyoBimanualArm, we use model site IDs for current data (not trajectory indices)
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
                data, self._rel_site_ids, self._rel_body_ids, model.body_rootid, backend
            )

        # Calculate distances and rewards
        if self._use_mean_exp_reward:
            # Better for parallel environments: mean(exp(-beta * dist))
            # Apply exp first, then mean across joint/site dimensions
            qpos_dists = backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos])

            # Add quaternion distance to maintain same structure as original
            if qpos_quat.size > 0:
                quat_dists = quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend)
                qpos_dists = qpos_dists + backend.mean(quat_dists)  # Add mean quat dist like original

            qpos_reward = backend.mean(backend.exp(-self._qpos_w_exp * qpos_dists))

            qvel_dists = backend.square(qvel - qvel_traj)
            qvel_reward = backend.mean(backend.exp(-self._qvel_w_exp * qvel_dists))

            if len(self._rel_site_ids) > 1:
                rpos_dists = backend.square(site_rpos - site_rpos_traj)
                rpos_reward = backend.mean(backend.exp(-self._rpos_w_exp * rpos_dists))

                rangles_dists = backend.square(site_rangles - site_rangles_traj)
                rangles_reward = backend.mean(backend.exp(-self._rquat_w_exp * rangles_dists))

                rvel_rot_dists = backend.square(site_rvel[:, :3] - site_rvel_traj[:, :3])
                rvel_rot_reward = backend.mean(backend.exp(-self._rvel_w_exp * rvel_rot_dists))

                rvel_lin_dists = backend.square(site_rvel[:, 3:] - site_rvel_traj[:, 3:])
                rvel_lin_reward = backend.mean(backend.exp(-self._rvel_w_exp * rvel_lin_dists))

            # Compute raw scalar distances for adaptive sampling (mean of per-element squared dists)
            raw_qpos_dist = backend.mean(qpos_dists)
            raw_qvel_dist = backend.mean(qvel_dists)
            if len(self._rel_site_ids) > 1:
                raw_rpos_dist = backend.mean(rpos_dists)
                raw_rangles_dist = backend.mean(rangles_dists)
                raw_rvel_rot_dist = backend.mean(rvel_rot_dists)
                raw_rvel_lin_dist = backend.mean(rvel_lin_dists)
            else:
                raw_rpos_dist = raw_rangles_dist = raw_rvel_rot_dist = raw_rvel_lin_dist = 0.0
        else:
            # Backward compatible: exp(-beta * mean(dist)) - original structure
            qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))

            # Add quaternion distance only if quaternion joints exist
            if qpos_quat.size > 0:
                qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))

            qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
            if len(self._rel_site_ids) > 1:
                rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
                rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
                rvel_rot_dist = backend.mean(backend.square(site_rvel[:, :3] - site_rvel_traj[:, :3]))
                rvel_lin_dist = backend.mean(backend.square(site_rvel[:, 3:] - site_rvel_traj[:, 3:]))

            # calculate rewards
            qpos_reward = backend.exp(-self._qpos_w_exp * qpos_dist)
            qvel_reward = backend.exp(-self._qvel_w_exp * qvel_dist)
            if len(self._rel_site_ids) > 1:
                rpos_reward = backend.exp(-self._rpos_w_exp * rpos_dist)
                rangles_reward = backend.exp(-self._rquat_w_exp * rangles_dist)
                rvel_rot_reward = backend.exp(-self._rvel_w_exp * rvel_rot_dist)
                rvel_lin_reward = backend.exp(-self._rvel_w_exp * rvel_lin_dist)

            # Use existing scalar distances for adaptive sampling
            raw_qpos_dist = qpos_dist
            raw_qvel_dist = qvel_dist
            if len(self._rel_site_ids) > 1:
                raw_rpos_dist = rpos_dist
                raw_rangles_dist = rangles_dist
                raw_rvel_rot_dist = rvel_rot_dist
                raw_rvel_lin_dist = rvel_lin_dist
            else:
                raw_rpos_dist = raw_rangles_dist = raw_rvel_rot_dist = raw_rvel_lin_dist = 0.0

        # Root position tracking reward.
        root_pos_reward = 0.0
        raw_root_pos_dist = 0.0
        offset_xyz = None
        if self._free_joint_qpos_ind is not None:
            root_xyz = data.qpos[self._free_joint_qpos_ind[:3]]
            traj_root_xyz = traj_data_single.qpos[self._free_joint_qpos_ind[:3]]
            if xy_offset is not None:
                offset_xyz = backend.concatenate([xy_offset, backend.zeros(1, dtype=xy_offset.dtype)])
                traj_root_xyz = traj_root_xyz - offset_xyz
            raw_root_pos_dist = backend.mean(backend.square(root_xyz - traj_root_xyz))
            root_pos_reward = backend.exp(-self._root_pos_w_exp * raw_root_pos_dist)

        # Full SO(3) root-orientation tracking.  This is intentionally separate
        # from qpos_reward: otherwise a globally rotated but internally correct
        # pose can retain nearly all of the dominant root-relative site reward.
        root_orientation_reward = 0.0
        root_orientation_error = 0.0
        raw_root_orientation_dist = 0.0
        if self._free_joint_qpos_ind is not None:
            root_quat = data.qpos[self._free_joint_qpos_ind[3:7]]
            traj_root_quat = traj_data_single.qpos[self._free_joint_qpos_ind[3:7]]
            root_quat = root_quat / backend.maximum(backend.linalg.norm(root_quat), 1e-8)
            traj_root_quat = traj_root_quat / backend.maximum(backend.linalg.norm(traj_root_quat), 1e-8)
            root_quat_dot = backend.abs(backend.dot(root_quat, traj_root_quat))
            root_orientation_error = 2.0 * backend.arccos(backend.clip(root_quat_dot, 0.0, 1.0))
            raw_root_orientation_dist = backend.square(root_orientation_error)
            root_orientation_reward = backend.exp(-self._root_orientation_w_exp * raw_root_orientation_dist)

        absolute_site_reward = 0.0
        raw_absolute_site_dist = 0.0
        if self._absolute_site_w_sum > 0.0:
            cur_abs_sites = data.site_xpos[self._absolute_site_ids]
            if self._site_mapper.requires_mapping:
                traj_abs_site_indices = self._site_mapper.model_ids_to_traj_indices(self._absolute_site_ids)
                ref_abs_sites = traj_data_single.site_xpos[traj_abs_site_indices]
            else:
                ref_abs_sites = traj_data_single.site_xpos[self._absolute_site_ids]
            if xy_offset is not None:
                if offset_xyz is None:
                    offset_xyz = backend.concatenate([xy_offset, backend.zeros(1, dtype=xy_offset.dtype)])
                ref_abs_sites = ref_abs_sites - offset_xyz
            raw_absolute_site_dist = backend.mean(backend.square(cur_abs_sites - ref_abs_sites))
            absolute_site_reward = backend.exp(-self._absolute_site_w_exp * raw_absolute_site_dist)

        # Compute total raw imitation error for adaptive sampling (weighted sum of raw distances)
        imitation_error_total = (
            self._qpos_w_sum * raw_qpos_dist
            + qvel_w_sum * raw_qvel_dist
            + self._root_pos_w_sum * raw_root_pos_dist
            + self._rpos_w_sum * raw_rpos_dist
            + self._rquat_w_sum * raw_rangles_dist
            + self._rvel_w_sum * raw_rvel_rot_dist
            + self._rvel_w_sum * raw_rvel_lin_dist
            + self._absolute_site_w_sum * raw_absolute_site_dist
            + self._root_orientation_w_sum * raw_root_orientation_dist
        )

        # Root velocity tracking reward
        # Always compute if free joint exists (weight from carry handles enable/disable)
        root_vel_reward = 0.0
        root_ang_vel_reward = 0.0
        root_ang_vel_error = 0.0
        root_ang_vel_norm = 0.0
        ref_root_ang_vel_norm = 0.0
        if self._free_joint_qpos_ind is not None:
            if backend == np:
                R = np_R
            else:
                R = jnp_R

            def calc_root_local_vel(_d):
                lin_vel_global = _d.qvel[self._free_joint_qvel_ind][:3]
                ang_vel_global = _d.qvel[self._free_joint_qvel_ind][3:]
                root_quat = R.from_quat(quat_scalarfirst2scalarlast(_d.qpos[self._free_joint_qpos_ind][3:7]))
                lin_vel_local = root_quat.as_matrix().T @ lin_vel_global
                # Include all 6 DOF: XYZ linear velocity + XYZ angular velocity
                return backend.concatenate([lin_vel_local, ang_vel_global])

            vel_local = calc_root_local_vel(data)
            traj_vel_local = calc_root_local_vel(traj_data_single)
            root_vel_dist = backend.mean(backend.square(vel_local - traj_vel_local))
            root_vel_reward = backend.exp(-self._root_vel_w_exp * root_vel_dist)

            root_ang_vel = data.qvel[self._free_joint_qvel_ind][3:]
            traj_root_ang_vel = traj_data_single.qvel[self._free_joint_qvel_ind][3:]
            root_ang_vel_delta = root_ang_vel - traj_root_ang_vel
            root_ang_vel_dist = backend.mean(backend.square(root_ang_vel_delta))
            root_ang_vel_reward = backend.exp(-self._root_ang_vel_w_exp * root_ang_vel_dist)
            root_ang_vel_error = backend.linalg.norm(root_ang_vel_delta)
            root_ang_vel_norm = backend.linalg.norm(root_ang_vel)
            ref_root_ang_vel_norm = backend.linalg.norm(traj_root_ang_vel)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(
                action,
                lower_bound=env.mdp_info.action_space.low,
                upper_bound=env.mdp_info.action_space.high,
                backend=backend,
            )
        else:
            out_of_bound_reward = 0.0

        # joint acceleration penalty
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_penalty = -acceleration_norm
        else:
            acceleration_penalty = 0.0

        # joint torque penalty
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_penalty = -torque_norm
        else:
            torque_penalty = 0.0

        # Always compute policy/activation diagnostics.  Promotion uses these
        # even when their reward coefficients are zero, so turning a penalty
        # off must not silently turn observability off as well.
        action_rate_norm = backend.mean(backend.square(action - reward_state.last_action))
        action_low = backend.asarray(env.mdp_info.action_space.low)
        action_high = backend.asarray(env.mdp_info.action_space.high)
        action_range = backend.maximum(action_high - action_low, 1e-6)
        action_margin = self._action_saturation_margin_fraction * action_range
        action_saturation_fraction = backend.mean(
            (action <= action_low + action_margin) | (action >= action_high - action_margin)
        )

        # Penalize only the part of an action entering the same boundary band
        # used by the saturation diagnostic.  The boundary depth is normalized
        # to zero at the inner edge and one at (or beyond) the physical bound.
        # Raw Gaussian policy samples are intentionally *not* allowed to grow
        # this term above one: out-of-bounds magnitude already has its own
        # penalty, and an unbounded 2%-band normalization can otherwise make a
        # modest raw overshoot saturate the entire reward at -1.
        lower_margin_violation = backend.maximum(
            (action_low + action_margin - action) / action_margin,
            0.0,
        )
        upper_margin_violation = backend.maximum(
            (action - (action_high - action_margin)) / action_margin,
            0.0,
        )
        action_boundary_depth = backend.minimum(
            lower_margin_violation + upper_margin_violation,
            1.0,
        )
        action_saturation_cost = backend.mean(backend.square(action_boundary_depth))
        action_saturation_penalty = -action_saturation_cost if self._action_saturation_coeff > 0.0 else 0.0

        # action rate penalty
        if self._action_rate_coeff > 0.0:
            action_rate_penalty = -action_rate_norm
        else:
            action_rate_penalty = 0.0

        # activation energy penalty
        if self._muscle_activation_addresses.size > 0:
            muscle_activation = data.act[self._muscle_activation_addresses]
            activation_energy = backend.mean(backend.square(muscle_activation))
        else:
            activation_energy = 0.0
        if self._activation_energy_coeff > 0.0:
            activation_energy_penalty = -activation_energy
        else:
            activation_energy_penalty = 0.0

        fascicle_continuity_loss = 0.0
        fascicle_continuity_training_loss = 0.0
        fascicle_continuity_violation_fraction = 0.0
        fascicle_continuity_mean_abs_difference = 0.0
        fascicle_continuity_max_abs_difference = 0.0
        fascicle_continuity_active_chain_fraction = 0.0
        weighted_fascicle_continuity_penalty = 0.0
        if self._fascicle_continuity_compute:
            ordered_activation = ordered_body_activation(
                data,
                self._fascicle_continuity_spec,
                backend=backend,
            )
            continuity_metrics = robust_fascicle_continuity(
                ordered_activation,
                self._fascicle_continuity_spec,
                scale=self._fascicle_continuity_scale,
                huber_delta=self._fascicle_continuity_huber_delta,
            )
            fascicle_continuity_loss = continuity_metrics.loss
            fascicle_continuity_violation_fraction = continuity_metrics.violation_fraction
            fascicle_continuity_mean_abs_difference = continuity_metrics.mean_abs_edge_difference
            fascicle_continuity_max_abs_difference = continuity_metrics.max_abs_edge_difference
            fascicle_continuity_active_chain_fraction = continuity_metrics.active_chain_fraction
            if self._fascicle_continuity_reward_active:
                reward_continuity_metrics = robust_fascicle_continuity(
                    ordered_activation,
                    self._fascicle_continuity_reward_spec,
                    scale=self._fascicle_continuity_scale,
                    huber_delta=self._fascicle_continuity_huber_delta,
                )
                fascicle_continuity_training_loss = reward_continuity_metrics.loss
                penalty_loss = fascicle_continuity_training_loss
                if self._fascicle_continuity_raw_penalty_clip is not None:
                    penalty_loss = backend.minimum(
                        penalty_loss,
                        self._fascicle_continuity_raw_penalty_clip,
                    )
                weighted_fascicle_continuity_penalty = -self._fascicle_continuity_coefficient * penalty_loss

        # total penalties (coefficient applied once here)
        total_penalties_before_clip = (
            self._action_out_of_bounds_coeff * out_of_bound_reward
            + self._joint_acc_coeff * acceleration_penalty
            + self._joint_torque_coeff * torque_penalty
            + self._action_rate_coeff * action_rate_penalty
            + self._action_saturation_coeff * action_saturation_penalty
            + self._activation_energy_coeff * activation_energy_penalty
            + weighted_fascicle_continuity_penalty
        )
        total_penalities = backend.maximum(total_penalties_before_clip, -1.0)

        # --- Contact tracking rewards (all JAX-native for JIT safety) ---
        # Entire block (including carry weight reads) is gated on contact tracking being
        # attached, so non-contact configs and carries without contact fields are unaffected.
        new_foot_xpos = reward_state.last_foot_xpos
        contact_reward = 0.0

        if self._ctd_stance_mask is not None and self._foot_site_ids is not None:
            ref_trajectory, ref_frame = _select_reference_coordinates(
                carry.traj_state.traj_no,
                carry.traj_state.subtraj_step_no,
                self._ctd_eff_stride,
                self._ctd_num_frames,
                is_bank=self._ctd_is_bank,
                backend=backend,
            )
            if self._ctd_is_bank:
                stance = self._ctd_stance_mask[ref_trajectory, ref_frame]
                ref_feet_z = self._ctd_foot_z[ref_trajectory, ref_frame]
            else:
                stance = self._ctd_stance_mask[ref_frame]
                ref_feet_z = self._ctd_foot_z[ref_frame]
            n_stance = jnp.sum(stance)

            actual_feet_z = data.site_xpos[self._foot_site_ids, 2]
            height_errs = jnp.abs(actual_feet_z - ref_feet_z) * stance
            height_err = jnp.where(n_stance > 0, jnp.sum(height_errs) / jnp.maximum(n_stance, 1.0), 0.0)
            foot_height_reward = jnp.exp(-self._foot_contact_height_w_exp * height_err)

            new_foot_xpos = data.site_xpos[self._foot_site_ids]
            if reward_state.last_foot_xpos is not None:
                dt = env.dt
                foot_disp = jnp.sqrt(jnp.sum(jnp.square(new_foot_xpos - reward_state.last_foot_xpos), axis=-1))
                foot_vel = foot_disp / jnp.maximum(dt, 1e-6)
                stance_vel = foot_vel * stance
                mean_vel = jnp.where(n_stance > 0, jnp.sum(stance_vel) / jnp.maximum(n_stance, 1.0), 0.0)
                foot_velocity_reward = jnp.exp(-self._foot_contact_velocity_w_exp * mean_vel)
            else:
                foot_velocity_reward = 1.0

            contact_reward = (
                carry.foot_contact_height_w_sum * foot_height_reward
                + carry.foot_contact_velocity_w_sum * foot_velocity_reward
            )

        # calculate total reward
        total_reward = (
            self._qpos_w_sum * qpos_reward
            + qvel_w_sum * qvel_reward
            + self._root_pos_w_sum * root_pos_reward
            + root_vel_w_sum * root_vel_reward
            + self._root_orientation_w_sum * root_orientation_reward
            + self._root_ang_vel_w_sum * root_ang_vel_reward
            + self._absolute_site_w_sum * absolute_site_reward
            + contact_reward
        )
        if len(self._rel_site_ids) > 1:
            total_reward = (
                total_reward
                + self._rpos_w_sum * rpos_reward
                + self._rquat_w_sum * rangles_reward
                + self._rvel_w_sum * rvel_rot_reward
                + self._rvel_w_sum * rvel_lin_reward
            )

        # Keep the positive imitation/task reward observable before any
        # regularizer is applied.  Coefficient calibration must not infer this
        # value back from the clipped final reward because the final clamp can
        # destroy that information.
        imitation_reward_total = total_reward
        total_reward = total_reward + total_penalities

        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        replace_kwargs = dict(
            last_qvel=data.qvel,
            last_action=action,
            imitation_error_total=imitation_error_total,
        )
        if new_foot_xpos is not None:
            replace_kwargs["last_foot_xpos"] = new_foot_xpos
        reward_state = reward_state.replace(**replace_kwargs)
        carry = carry.replace(reward_state=reward_state)

        # Diagnostic error metrics (raw errors, not exp-transformed)
        err_root_xyz = err_root_yaw = err_joint_pos = err_joint_vel = err_site_abs = err_rpos = 0.0
        err_right_hand_pos = 0.0
        if self._free_joint_qpos_ind is not None:
            # Root XYZ error (world frame, with offset correction)
            err_root_xyz = backend.sqrt(raw_root_pos_dist)
            # Root yaw error
            root_quat = data.qpos[self._free_joint_qpos_ind[3:7]]
            traj_root_quat = traj_data_single.qpos[self._free_joint_qpos_ind[3:7]]
            yaw_diff = quat_to_yaw(root_quat, backend) - quat_to_yaw(traj_root_quat, backend)
            err_root_yaw = backend.abs(backend.arctan2(backend.sin(yaw_diff), backend.cos(yaw_diff)))
        # Joint errors
        if np.any(self._joint_qpos_mask):
            err_joint_pos = backend.sqrt(
                backend.mean(backend.square(qpos[self._joint_qpos_mask] - qpos_traj[self._joint_qpos_mask]))
            )
        if np.any(self._joint_qvel_mask):
            err_joint_vel = backend.sqrt(
                backend.mean(backend.square(qvel[self._joint_qvel_mask] - qvel_traj[self._joint_qvel_mask]))
            )
        # Absolute site deviation (like terminal handler)
        if len(self._rel_site_ids) > 1:
            site_mapping = self._rel_site_ids
            cur_sites = data.site_xpos[site_mapping]
            if self._site_mapper.requires_mapping:
                traj_idx = self._site_mapper.model_ids_to_traj_indices(site_mapping)
                ref_sites = traj_data_single.site_xpos[traj_idx]
            else:
                ref_sites = traj_data_single.site_xpos[site_mapping]
            if xy_offset is not None:
                offset_xyz = backend.concatenate([xy_offset, backend.zeros(1, dtype=xy_offset.dtype)])
                ref_sites = ref_sites - offset_xyz
            err_site_abs = backend.mean(backend.linalg.norm(cur_sites - ref_sites, axis=-1))
            if self._right_hand_rel_index is not None:
                err_right_hand_pos = backend.linalg.norm(
                    cur_sites[self._right_hand_rel_index] - ref_sites[self._right_hand_rel_index]
                )
            # Relative site position error (RMSE of site_rpos)
            err_rpos = backend.sqrt(backend.mean(backend.square(site_rpos - site_rpos_traj)))

        # Build reward_info for logging/diagnostics
        reward_info = {
            "reward_total": total_reward,
            "reward_imitation_total": imitation_reward_total,
            "reward_qpos": qpos_reward,
            "reward_qvel": qvel_reward,
            "reward_root_pos": root_pos_reward,
            "reward_root_vel": root_vel_reward,
            "reward_root_orientation": root_orientation_reward,
            "reward_root_ang_vel": root_ang_vel_reward,
            "penalty_total": total_penalities,
            "penalty_total_before_clip": total_penalties_before_clip,
            "penalty_action_saturation": (self._action_saturation_coeff * action_saturation_penalty),
            "penalty_activation_energy": self._activation_energy_coeff * activation_energy_penalty,
            "penalty_fascicle_continuity": weighted_fascicle_continuity_penalty,
            "activation_energy": activation_energy,
            "fascicle_continuity_loss": fascicle_continuity_loss,
            "fascicle_continuity_training_loss": fascicle_continuity_training_loss,
            "fascicle_continuity_violation_fraction": fascicle_continuity_violation_fraction,
            "fascicle_continuity_mean_abs_difference": fascicle_continuity_mean_abs_difference,
            "fascicle_continuity_max_abs_difference": fascicle_continuity_max_abs_difference,
            "fascicle_continuity_active_chain_fraction": fascicle_continuity_active_chain_fraction,
            "fascicle_continuity_measured_chain_count": self._fascicle_continuity_measured_chain_count,
            "fascicle_continuity_measured_edge_count": self._fascicle_continuity_measured_edge_count,
            "action_saturation_fraction": action_saturation_fraction,
            "action_rate_mean_square": action_rate_norm,
            "err_root_xyz": err_root_xyz,
            "err_root_yaw": err_root_yaw,
            "err_root_rot": root_orientation_error,
            "err_root_ang_vel": root_ang_vel_error,
            "root_ang_vel": root_ang_vel_norm,
            "ref_root_ang_vel": ref_root_ang_vel_norm,
            "err_joint_pos": err_joint_pos,
            "err_joint_vel": err_joint_vel,
            "err_site_abs": err_site_abs,
            "err_rpos": err_rpos,
            "err_right_hand_pos": err_right_hand_pos,
            "reward_absolute_site": absolute_site_reward,
            "err_absolute_site": backend.sqrt(raw_absolute_site_dist),
        }
        if len(self._rel_site_ids) > 1:
            reward_info["reward_rpos"] = rpos_reward
            reward_info["reward_rquat"] = rangles_reward
            reward_info["reward_rvel_rot"] = rvel_rot_reward
            reward_info["reward_rvel_lin"] = rvel_lin_reward

        return total_reward, carry, reward_info


def _racket_grip_finger_reference(env, model):
    """Right-hand finger grip reference for a racket environment.

    Prefers values the environment already resolved (``grip_finger_*`` properties);
    falls back to reading the grip reference JSON directly so the reward also works
    for plain envs constructed in tests. Returns empty lists when fingers are absent.
    """
    names = getattr(env, "grip_finger_names", None)
    addrs = getattr(env, "grip_finger_qpos_addrs", None)
    targets = getattr(env, "grip_finger_targets", None)
    if names is not None and addrs is not None and targets is not None:
        return list(names), list(addrs), list(targets)
    try:
        from musclemimic.environments.humanoids.myofullbody_racket import grip_finger_reference

        n, a, t = grip_finger_reference(model)
        return list(n), list(a), list(t)
    except Exception:
        return [], [], []


class RacketMimicReward(MimicReward):
    """
    MimicReward plus a racket tracking term for rigid racket-holding environments
    (MyoFullBodyRacket). The body mimic sites only track the wrist, so forearm
    pronation/supination errors are amplified over the ~0.5 m racket lever arm
    without ever being penalized. This reward adds direct tracking of a racket
    site (default: stringbed center) in position and orientation.

    The stored trajectories contain no racket sites (they are retargeted for the
    bare-hand MyoFullBody). Because the racket is a jointless rigid child of the
    hand, the reference racket pose is derived at every step as
    ``ref_hand_site_pose ∘ fixed_offset``, where the fixed offset is computed
    once from the model at construction time. No trajectory regeneration and no
    observation change is needed, so bare-hand checkpoints remain loadable.

    Both racket quantities are expressed relative to the main mimic site (pelvis),
    consistent with the relative site tracking of the base class; the world-frame
    XY offset between trajectory and simulation cancels out.
    """

    def __init__(
        self,
        env: Any,
        racket_site_name: str = "racket_stringbed_center_site",
        racket_hand_site_name: str = "right_hand_mimic",
        racket_pos_w_sum: float = 0.3,
        racket_pos_w_exp: float = 50.0,
        racket_rot_w_sum: float = 0.15,
        racket_rot_w_exp: float = 5.0,
        racket_linvel_w_sum: float = 0.0,
        racket_linvel_w_exp: float = 0.5,
        racket_angvel_w_sum: float = 0.0,
        racket_angvel_w_exp: float = 0.2,
        stringbed_normal_w_sum: float = 0.0,
        stringbed_normal_w_exp: float = 5.0,
        impact_timing_w_sum: float = 0.0,
        impact_timing_w_exp: float = 50.0,
        impact_phase: float = 0.55,
        impact_phase_window: float = 0.08,
        racket_phase_multipliers=None,
        racket_reference_source: str = "derived_rigid",
        event_cache_trajectory_no: int | None = None,
        finger_grip_w_sum: float = 0.2,
        finger_grip_w_exp: float = 10.0,
        joints_for_mimic=None,
        **kwargs,
    ):
        """
        Args:
            env (Any): Environment instance (must contain the rigid racket).
            racket_site_name (str): Racket site to track.
            racket_hand_site_name (str): Mimic site (present in the trajectory data)
                on the body the racket is rigidly attached to.
            racket_pos_w_sum (float): Summation weight of the racket position reward.
            racket_pos_w_exp (float): Exponential scale of the racket position reward.
            racket_rot_w_sum (float): Summation weight of the racket orientation reward.
            racket_rot_w_exp (float): Exponential scale of the racket orientation reward.
            racket_linvel_w_sum/racket_angvel_w_sum (float): Optional velocity
                tracking weights. Both default to zero, preserving Stage-2 v1.
            stringbed_normal_w_sum (float): Optional explicit string-bed normal
                alignment term. Defaults to zero.
            impact_timing_w_sum (float): Optional impact-window position term.
                Defaults to zero; ``impact_phase`` and ``impact_phase_window``
                define its normalized reference-motion window.
            racket_phase_multipliers (sequence, optional): Non-negative values
                sampled uniformly over motion phase and linearly interpolated.
                ``None`` is exactly the legacy constant multiplier of one.
            racket_reference_source (str): ``derived_rigid`` preserves the v1
                hand-offset reference. ``event_cache`` consumes independently
                measured/fused per-frame racket and six-phase event arrays.
            event_cache_trajectory_no (int, optional): Explicit trajectory
                binding for a single-motion event cache. Multi-trajectory
                handlers are rejected until a per-trajectory cache registry is
                supplied.
            finger_grip_w_sum (float): Summation weight of the finger-grip hold reward.
                Only active when the environment has finger joints (fingers enabled).
            finger_grip_w_exp (float): Exponential scale of the finger-grip reward.
            joints_for_mimic (list, optional): Base-class joint mimic set. Defaults to
                all joints *except* the right-hand finger joints, which are instead
                pinned to the grip pose by the finger-grip term (the trajectories carry
                no finger reference, so the base qpos term would otherwise pull them to
                an open hand). Pass an explicit list to override.
            **kwargs: Forwarded to :class:`MimicReward`.
        """
        model = env._model
        self._racket_env = env
        self._racket_reference_mode = str(racket_reference_source)
        if self._racket_reference_mode not in {"derived_rigid", "event_cache"}:
            raise ValueError("racket_reference_source must be 'derived_rigid' or 'event_cache'")
        self._event_cache_trajectory_no = event_cache_trajectory_no
        if self._racket_reference_mode == "event_cache":
            if event_cache_trajectory_no is not None and int(event_cache_trajectory_no) != 0:
                raise ValueError("event_cache_trajectory_no, when supplied for a single cache, must be 0")
            if getattr(env, "th", None) is None:
                raise ValueError("event_cache racket reference requires a trajectory handler")

        # Right-hand finger joints (empty when fingers are disabled). We keep these
        # out of the base qpos/qvel mimic and hold them at the grip pose separately.
        grip_names, grip_addrs, grip_targets = _racket_grip_finger_reference(env, model)
        self._finger_grip_addrs = np.asarray(grip_addrs, dtype=int)
        self._finger_grip_targets = np.asarray(grip_targets, dtype=float)
        self._finger_grip_w_sum = finger_grip_w_sum
        self._finger_grip_w_exp = finger_grip_w_exp

        if joints_for_mimic is None and len(grip_names) > 0:
            grip_set = set(grip_names)
            joints_for_mimic = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(model.njnt)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in grip_set
            ]

        super().__init__(env, joints_for_mimic=joints_for_mimic, **kwargs)

        self._racket_pos_w_sum = racket_pos_w_sum
        self._racket_pos_w_exp = racket_pos_w_exp
        self._racket_rot_w_sum = racket_rot_w_sum
        self._racket_rot_w_exp = racket_rot_w_exp
        self._racket_linvel_w_sum = float(racket_linvel_w_sum)
        self._racket_linvel_w_exp = float(racket_linvel_w_exp)
        self._racket_angvel_w_sum = float(racket_angvel_w_sum)
        self._racket_angvel_w_exp = float(racket_angvel_w_exp)
        self._stringbed_normal_w_sum = float(stringbed_normal_w_sum)
        self._stringbed_normal_w_exp = float(stringbed_normal_w_exp)
        self._impact_timing_w_sum = float(impact_timing_w_sum)
        self._impact_timing_w_exp = float(impact_timing_w_exp)
        self._impact_phase = float(impact_phase)
        self._impact_phase_window = float(impact_phase_window)
        if not 0.0 <= self._impact_phase <= 1.0:
            raise ValueError("impact_phase must lie in [0, 1]")
        if self._impact_phase_window <= 0.0:
            raise ValueError("impact_phase_window must be positive")
        if any(
            value < 0.0
            for value in (
                self._racket_linvel_w_sum,
                self._racket_angvel_w_sum,
                self._stringbed_normal_w_sum,
                self._impact_timing_w_sum,
            )
        ):
            raise ValueError("optional racket reward weights must be non-negative")
        self._racket_phase_multipliers = _validate_racket_phase_multipliers(racket_phase_multipliers)
        self._racket_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, racket_site_name)
        if self._racket_site_id < 0:
            raise ValueError(
                f"racket site {racket_site_name!r} not found in the model. "
                "RacketMimicReward requires a racket-holding environment "
                "(e.g. MyoFullBodyRacket with enable_racket=True)."
            )
        self._racket_hand_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, racket_hand_site_name)
        if self._racket_hand_site_id < 0:
            raise ValueError(f"hand mimic site {racket_hand_site_name!r} not found in the model")
        # main site (index 0 of the relative mimic sites, pelvis by convention)
        self._racket_main_site_id = int(self._rel_site_ids[0])
        self._racket_hand_body_id = int(model.site_bodyid[self._racket_hand_site_id])
        self._racket_main_body_id = int(model.site_bodyid[self._racket_main_site_id])
        self._racket_site_body_id = int(model.site_bodyid[self._racket_site_id])
        self._racket_velocity_body_ids = np.asarray([self._racket_hand_body_id, self._racket_main_body_id], dtype=int)
        self._racket_velocity_root_ids = np.asarray(model.body_rootid[self._racket_velocity_body_ids], dtype=int)
        self._racket_traj_query_ids = np.array([self._racket_hand_site_id, self._racket_main_site_id], dtype=int)

        # The reference derivation is only valid if the racket is rigidly fixed to
        # the hand-site body: verify the kinematic chain has no joints in between.
        racket_bid = int(model.site_bodyid[self._racket_site_id])
        hand_bid = int(model.site_bodyid[self._racket_hand_site_id])
        b, njnt_on_chain = racket_bid, 0
        while b != 0 and b != hand_bid:
            njnt_on_chain += int(model.body_jntnum[b])
            b = int(model.body_parentid[b])
        if b != hand_bid or njnt_on_chain > 0:
            raise ValueError(
                f"racket site {racket_site_name!r} is not rigidly attached to the body of "
                f"{racket_hand_site_name!r} (ancestor found: {b == hand_bid}, "
                f"joints on chain: {njnt_on_chain}); the derived racket reference is invalid"
            )

        # Fixed transform hand site -> racket site, exact for any qpos (rigid chain).
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        hand_mat = d.site_xmat[self._racket_hand_site_id].reshape(3, 3)
        hand_pos = d.site_xpos[self._racket_hand_site_id]
        racket_mat = d.site_xmat[self._racket_site_id].reshape(3, 3)
        racket_pos = d.site_xpos[self._racket_site_id]
        self._racket_off_pos = hand_mat.T @ (racket_pos - hand_pos)
        self._racket_off_mat = hand_mat.T @ racket_mat

    def attach_contact_tracking(self, contact_data, foot_site_names, model):
        super().attach_contact_tracking(contact_data, foot_site_names, model)
        if self._racket_reference_mode != "event_cache":
            return
        required = (
            "phase_global",
            "phase_id",
            "phase_local",
            "time_to_impact_s",
            "time_from_impact_s",
            "impact_flag",
            "racket_quaternion_world",
            "racket_linear_velocity_world",
            "racket_angular_velocity_world",
            "stringbed_normal_world",
            "stringbed_center_world",
            "racket_reference_confidence",
        )
        missing = [name for name in required if getattr(contact_data, name, None) is None]
        if missing:
            raise ValueError("event_cache racket reference is incomplete; missing " + ", ".join(missing))
        sources = contact_data.racket_reference_source
        if isinstance(sources, str):
            sources = (sources,)
        if not sources or any(source not in {"measured", "fused"} for source in sources):
            raise ValueError(
                f"event_cache mainline requires an independent measured/fused racket reference, got {sources!r}"
            )
        env_trajectories = int(self._racket_env.th.n_trajectories)
        cache_trajectories = int(getattr(contact_data, "num_trajectories", 1))
        if env_trajectories != cache_trajectories:
            raise ValueError(
                "event cache count differs from trajectory handler: "
                f"cache={cache_trajectories} environment={env_trajectories}"
            )
        if cache_trajectories == 1 and self._event_cache_trajectory_no != 0:
            raise ValueError(
                "single event cache requires event_cache_trajectory_no=0; multi-motion banks must leave it null"
            )
        if cache_trajectories > 1 and self._event_cache_trajectory_no is not None:
            raise ValueError("multi-motion event bank selects by traj_no; fixed trajectory binding is forbidden")
        frame_counts = np.atleast_1d(contact_data.num_frames)
        strides = np.atleast_1d(contact_data.effective_ref_stride)
        for trajectory in range(env_trajectories):
            trajectory_length = int(self._racket_env.th.len_trajectory(trajectory))
            last_cache_frame = round((trajectory_length - 1) * float(strides[trajectory]))
            if last_cache_frame >= int(frame_counts[trajectory]):
                raise ValueError(
                    "event cache is shorter than the bound trajectory: "
                    f"traj_no={trajectory} last required frame={last_cache_frame}, "
                    f"cache frames={int(frame_counts[trajectory])}"
                )

    def derive_reference_racket_pose(self, hand_site_pos, hand_site_mat, backend=np):
        """Reference racket site pose implied by the rigid grip from a hand site pose."""
        pos = hand_site_pos + hand_site_mat @ self._racket_off_pos
        mat = hand_site_mat @ self._racket_off_mat
        return pos, mat

    def _phase_multiplier(
        self,
        env: Any,
        carry: Any,
        backend: ModuleType,
        *,
        motion_phase=None,
    ):
        if self._racket_phase_multipliers is None:
            return 1.0, motion_phase
        if motion_phase is None:
            length = env.th.len_trajectory(carry.traj_state.traj_no)
            motion_phase = carry.traj_state.subtraj_step_no / backend.maximum(length - 1, 1)
        phase = backend.clip(motion_phase, 0.0, 1.0)
        values = backend.asarray(self._racket_phase_multipliers)
        knots = backend.linspace(0.0, 1.0, values.shape[0])
        return backend.interp(phase, knots, values), phase

    @check_traj_provided
    def __call__(
        self,
        state: Union[np.ndarray, jnp.ndarray],
        action: Union[np.ndarray, jnp.ndarray],
        next_state: Union[np.ndarray, jnp.ndarray],
        absorbing: bool,
        info: Dict[str, Any],
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[float, Any]:
        total_reward, carry, reward_info = super().__call__(
            state, action, next_state, absorbing, info, env, model, data, carry, backend
        )

        # Finger-grip hold: keep the right hand closed on the racket at the grip pose.
        # Active only when the env has finger joints (fingers enabled); otherwise the
        # address set is empty and this contributes nothing.
        finger_active = self._finger_grip_addrs.size > 0 and self._finger_grip_w_sum > 0.0
        finger_extra_err = 0.0
        if finger_active:
            finger_qpos = data.qpos[self._finger_grip_addrs]
            raw_finger_dist = backend.mean(backend.square(finger_qpos - self._finger_grip_targets))
            finger_grip_reward = backend.exp(-self._finger_grip_w_exp * raw_finger_dist)
            finger_grip_reward = backend.nan_to_num(finger_grip_reward, nan=0.0)
            total_reward = total_reward + self._finger_grip_w_sum * finger_grip_reward
            finger_extra_err = self._finger_grip_w_sum * raw_finger_dist
            reward_info["reward_finger_grip"] = finger_grip_reward
            reward_info["err_finger_grip"] = backend.sqrt(raw_finger_dist)

        racket_terms_active = self._racket_reference_mode == "event_cache" or any(
            value > 0.0
            for value in (
                self._racket_pos_w_sum,
                self._racket_rot_w_sum,
                self._racket_linvel_w_sum,
                self._racket_angvel_w_sum,
                self._stringbed_normal_w_sum,
                self._impact_timing_w_sum,
            )
        )
        if not racket_terms_active:
            if finger_active:
                reward_state = carry.reward_state
                reward_state = reward_state.replace(
                    imitation_error_total=reward_state.imitation_error_total + finger_extra_err
                )
                carry = carry.replace(reward_state=reward_state)
                reward_info["reward_imitation_total"] = (
                    reward_info["reward_imitation_total"] + self._finger_grip_w_sum * finger_grip_reward
                )
            reward_info["reward_total"] = total_reward
            return total_reward, carry, reward_info

        R = np_R if backend == np else jnp_R

        traj_data_single = env.th.get_current_traj_data(carry, backend)
        if self._site_mapper.requires_mapping:
            traj_idx = self._site_mapper.model_ids_to_traj_indices(self._racket_traj_query_ids)
        else:
            traj_idx = self._racket_traj_query_ids

        ref_hand_pos = traj_data_single.site_xpos[traj_idx[0]]
        ref_hand_mat = traj_data_single.site_xmat[traj_idx[0]].reshape(3, 3)
        ref_main_pos = traj_data_single.site_xpos[traj_idx[1]]
        ref_main_mat = traj_data_single.site_xmat[traj_idx[1]].reshape(3, 3)
        event_frame = None
        event_motion_phase = None
        event_phase_id = None
        event_phase_local = None
        event_time_to_impact = None
        event_time_from_impact = None
        event_impact_flag = None
        event_reference_confidence = None
        if self._racket_reference_mode == "event_cache":
            if self._ctd_phase_id is None or self._ctd_stringbed_center_world is None:
                raise RuntimeError(
                    "event_cache reference was not attached; call attach_contact_tracking "
                    "with a complete single-motion event/racket cache"
                )
            event_trajectory, event_frame = _select_reference_coordinates(
                carry.traj_state.traj_no,
                carry.traj_state.subtraj_step_no,
                self._ctd_eff_stride,
                self._ctd_num_frames,
                is_bank=self._ctd_is_bank,
                backend=backend,
            )
            event_index = (event_trajectory, event_frame) if self._ctd_is_bank else event_frame
            ref_racket_pos = self._ctd_stringbed_center_world[event_index]
            ref_quat = self._ctd_racket_quaternion_world[event_index]
            ref_racket_mat = R.from_quat(quat_scalarfirst2scalarlast(ref_quat)).as_matrix()
            ref_stringbed_normal = self._ctd_stringbed_normal_world[event_index]
            event_motion_phase = self._ctd_phase_global[event_index]
            event_phase_id = self._ctd_phase_id[event_index]
            event_phase_local = self._ctd_phase_local[event_index]
            event_time_to_impact = self._ctd_time_to_impact_s[event_index]
            event_time_from_impact = self._ctd_time_from_impact_s[event_index]
            event_impact_flag = self._ctd_impact_flag[event_index]
            event_reference_confidence = self._ctd_racket_reference_confidence[event_index]
        else:
            ref_racket_pos, ref_racket_mat = self.derive_reference_racket_pose(ref_hand_pos, ref_hand_mat, backend)
            ref_stringbed_normal = ref_racket_mat[:, 2]

        cur_racket_pos = data.site_xpos[self._racket_site_id]
        cur_racket_mat = data.site_xmat[self._racket_site_id].reshape(3, 3)
        cur_main_pos = data.site_xpos[self._racket_main_site_id]
        cur_main_mat = data.site_xmat[self._racket_main_site_id].reshape(3, 3)

        # main-site-relative quantities: trajectory/simulation world XY offset cancels
        rpos_ref = ref_racket_pos - ref_main_pos
        rpos_cur = cur_racket_pos - cur_main_pos
        raw_racket_pos_dist = backend.mean(backend.square(rpos_cur - rpos_ref))

        rot_ref = ref_main_mat.T @ ref_racket_mat
        rot_cur = cur_main_mat.T @ cur_racket_mat
        rot_err_vec = R.from_matrix(rot_ref.T @ rot_cur).as_rotvec()
        raw_racket_rot_dist = backend.mean(backend.square(rot_err_vec))

        racket_pos_reward = backend.exp(-self._racket_pos_w_exp * raw_racket_pos_dist)
        racket_rot_reward = backend.exp(-self._racket_rot_w_exp * raw_racket_rot_dist)

        raw_racket_linvel_dist = 0.0
        raw_racket_angvel_dist = 0.0
        racket_linvel_reward = 0.0
        racket_angvel_reward = 0.0
        velocity_terms_active = self._racket_linvel_w_sum > 0.0 or self._racket_angvel_w_sum > 0.0
        if velocity_terms_active:
            trajectory_indices = traj_idx if self._site_mapper.requires_mapping else None
            ref_site_vel = calc_site_velocities(
                self._racket_traj_query_ids,
                traj_data_single,
                self._racket_velocity_body_ids,
                self._racket_velocity_root_ids,
                backend,
                trajectory_site_indices=trajectory_indices,
            )
            cur_site_vel = calc_site_velocities(
                self._racket_traj_query_ids,
                data,
                self._racket_velocity_body_ids,
                self._racket_velocity_root_ids,
                backend,
            )
            if self._racket_reference_mode == "event_cache":
                ref_relative_lin = self._ctd_racket_linear_velocity_world[event_index] - ref_site_vel[1, 3:]
                ref_relative_ang = self._ctd_racket_angular_velocity_world[event_index] - ref_site_vel[1, :3]
            else:
                ref_hand_ang = ref_site_vel[0, :3]
                ref_racket_lin = ref_site_vel[0, 3:] + backend.cross(ref_hand_ang, ref_racket_pos - ref_hand_pos)
                ref_relative_lin = ref_racket_lin - ref_site_vel[1, 3:]
                ref_relative_ang = ref_hand_ang - ref_site_vel[1, :3]

            cur_racket_body_vel = data.cvel[self._racket_site_body_id]
            cur_racket_lin = cur_racket_body_vel[3:] + backend.cross(
                cur_racket_body_vel[:3],
                cur_racket_pos - data.subtree_com[model.body_rootid[self._racket_site_body_id]],
            )
            cur_relative_lin = cur_racket_lin - cur_site_vel[1, 3:]
            cur_relative_ang = cur_racket_body_vel[:3] - cur_site_vel[1, :3]
            raw_racket_linvel_dist = backend.mean(backend.square(cur_relative_lin - ref_relative_lin))
            raw_racket_angvel_dist = backend.mean(backend.square(cur_relative_ang - ref_relative_ang))
            racket_linvel_reward = backend.exp(-self._racket_linvel_w_exp * raw_racket_linvel_dist)
            racket_angvel_reward = backend.exp(-self._racket_angvel_w_exp * raw_racket_angvel_dist)

        raw_stringbed_normal_dist = backend.mean(backend.square(cur_racket_mat[:, 2] - ref_stringbed_normal))
        stringbed_normal_reward = backend.exp(-self._stringbed_normal_w_exp * raw_stringbed_normal_dist)

        phase_multiplier, motion_phase = self._phase_multiplier(env, carry, backend, motion_phase=event_motion_phase)
        impact_timing_reward = 0.0
        if self._impact_timing_w_sum > 0.0:
            if motion_phase is None:
                length = env.th.len_trajectory(carry.traj_state.traj_no)
                motion_phase = carry.traj_state.subtraj_step_no / backend.maximum(length - 1, 1)
                motion_phase = backend.clip(motion_phase, 0.0, 1.0)
            if self._racket_reference_mode == "event_cache":
                phase_window = backend.exp(-0.5 * backend.square(event_time_to_impact / self._impact_phase_window))
            else:
                phase_window = backend.exp(
                    -0.5 * backend.square((motion_phase - self._impact_phase) / self._impact_phase_window)
                )
            impact_timing_reward = phase_window * backend.exp(-self._impact_timing_w_exp * raw_racket_pos_dist)

        legacy_racket_reward = self._racket_pos_w_sum * racket_pos_reward + self._racket_rot_w_sum * racket_rot_reward
        extra_racket_reward = (
            self._racket_linvel_w_sum * racket_linvel_reward
            + self._racket_angvel_w_sum * racket_angvel_reward
            + self._stringbed_normal_w_sum * stringbed_normal_reward
            + self._impact_timing_w_sum * impact_timing_reward
        )
        racket_reward = phase_multiplier * (legacy_racket_reward + extra_racket_reward)
        racket_reward = backend.nan_to_num(racket_reward, nan=0.0)

        total_reward = total_reward + racket_reward

        # keep adaptive trajectory sampling consistent with the extra tracking terms
        reward_state = carry.reward_state
        reward_state = reward_state.replace(
            imitation_error_total=reward_state.imitation_error_total
            + self._racket_pos_w_sum * raw_racket_pos_dist
            + self._racket_rot_w_sum * raw_racket_rot_dist
            + self._racket_linvel_w_sum * raw_racket_linvel_dist
            + self._racket_angvel_w_sum * raw_racket_angvel_dist
            + self._stringbed_normal_w_sum * raw_stringbed_normal_dist
            + self._impact_timing_w_sum * raw_racket_pos_dist
            + finger_extra_err
        )
        carry = carry.replace(reward_state=reward_state)

        reward_info["reward_racket_pos"] = racket_pos_reward
        reward_info["reward_racket_rot"] = racket_rot_reward
        reward_info["err_racket_pos"] = backend.sqrt(raw_racket_pos_dist)
        reward_info["err_racket_rot"] = backend.sqrt(raw_racket_rot_dist)
        reward_info["reward_racket_linvel"] = racket_linvel_reward
        reward_info["reward_racket_angvel"] = racket_angvel_reward
        reward_info["reward_stringbed_normal"] = stringbed_normal_reward
        reward_info["reward_impact_timing"] = impact_timing_reward
        reward_info["err_racket_linvel"] = backend.sqrt(raw_racket_linvel_dist)
        reward_info["err_racket_angvel"] = backend.sqrt(raw_racket_angvel_dist)
        reward_info["err_stringbed_normal"] = backend.sqrt(raw_stringbed_normal_dist)
        reward_info["racket_phase_multiplier"] = phase_multiplier
        if self._racket_reference_mode == "event_cache":
            reward_info.update(
                {
                    "phase_global": event_motion_phase,
                    "phase_id": event_phase_id,
                    "phase_local": event_phase_local,
                    "time_to_impact_s": event_time_to_impact,
                    "time_from_impact_s": event_time_from_impact,
                    "impact_flag": event_impact_flag,
                    "reference_confidence": event_reference_confidence,
                    "reference_cache_frame": event_frame,
                }
            )
        reward_info["reward_imitation_total"] = (
            reward_info["reward_imitation_total"]
            + (self._finger_grip_w_sum * finger_grip_reward if finger_active else 0.0)
            + racket_reward
        )
        reward_info["reward_total"] = total_reward

        return total_reward, carry, reward_info


def _validate_racket_phase_multipliers(values):
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("racket_phase_multipliers must contain at least two values")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("racket_phase_multipliers must be finite and non-negative")
    return tuple(float(value) for value in array)


TargetVelocityTrajReward.register()
MimicReward.register()
RacketMimicReward.register()
