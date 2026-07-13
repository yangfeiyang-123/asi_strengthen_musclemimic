from types import ModuleType
from typing import Any, Dict, Tuple, Union

import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from flax import struct
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quantities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from musclemimic.core.utils.site_mapping import create_site_mapper
from musclemimic.utils.finger_isolation import finger_joint_side


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


def quat_to_yaw(quat, backend):
    """Extract yaw from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return backend.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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

    def __init__(self, env: Any,
                 w_exp=10.0,
                 **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
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
            _root_quat = R.from_quat(quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7]))
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        traj_data = env.th.get_current_traj_data(carry, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(vel_local - traj_vel_local)))

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

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 joints_for_mimic=None,
                 exclude_finger_joints=False,
                 absolute_site_reward_sites=None,
                 absolute_site_w_sum=0.0,
                 absolute_site_w_exp=10.0,
                 **kwargs):
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
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._activation_energy_coeff = kwargs.get("activation_energy_coeff", 0.0)
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
            list(rel_site_names).index("right_hand_mimic")
            if "right_hand_mimic" in rel_site_names
            else None
        )
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
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
        env_sites_for_mimic = getattr(env, 'sites_for_mimic', [])
        traj_site_names = env.th.traj.info.site_names if (hasattr(env, 'th') and env.th is not None) else None
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


    def attach_contact_tracking(self, contact_data, foot_site_names, model):
        """Attach contact tracking data for contact-preserving reward terms.

        Pre-converts numpy arrays to JAX for JIT-safe indexing inside __call__.
        """
        n_foot_sites = len(foot_site_names)
        n_cache_feet = contact_data.foot_points.shape[1]
        if n_foot_sites != n_cache_feet:
            raise ValueError(
                f"foot_sites count ({n_foot_sites}) does not match "
                f"tracking cache foot count ({n_cache_feet}). "
                f"Config sites: {foot_site_names}, cache labels: {contact_data.foot_labels}"
            )
        self._contact_tracking_data = contact_data
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in foot_site_names
        ], dtype=np.int32)
        self._ctd_stance_mask = jnp.asarray(contact_data.stance_mask, dtype=jnp.float32)
        self._ctd_foot_z = jnp.asarray(contact_data.foot_points[:, :, 2], dtype=jnp.float32)
        self._ctd_eff_stride = jnp.float32(contact_data.effective_ref_stride)
        self._ctd_num_frames = jnp.int32(contact_data.num_frames)
        if contact_data.body_laplacian is not None:
            self._ctd_body_laplacian = jnp.asarray(contact_data.body_laplacian, dtype=jnp.float32)
        else:
            self._ctd_body_laplacian = None

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
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

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
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
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
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
        if self._site_mapper.requires_mapping and hasattr(env, 'th') and env.th is not None:
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
            
            site_rpos_traj, site_rangles_traj, site_rvel_traj =\
                calculate_relative_site_quantities(traj_data_single, self._rel_site_ids,
                                                self._rel_body_ids, model.body_rootid, backend,
                                                trajectory_site_indices=traj_site_indices)

        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        
        # Handle quaternion joints if they exist
        qpos_quat = qpos[self._quat_in_qpos]
        if qpos_quat.size > 0:
            qpos_quat = qpos_quat.reshape(-1, 4)
        
        if len(self._rel_site_ids) > 1:
            # For MyoBimanualArm, we use model site IDs for current data (not trajectory indices)
            site_rpos, site_rangles, site_rvel = (
                calculate_relative_site_quantities(data, self._rel_site_ids, self._rel_body_ids,
                                                model.body_rootid, backend))

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
                
                rvel_rot_dists = backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3])
                rvel_rot_reward = backend.mean(backend.exp(-self._rvel_w_exp * rvel_rot_dists))
                
                rvel_lin_dists = backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:])
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
                rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
                rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

            # calculate rewards
            qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
            qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
            if len(self._rel_site_ids) > 1:
                rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
                rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
                rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
                rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

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
            self._qpos_w_sum * raw_qpos_dist +
            qvel_w_sum * raw_qvel_dist +
            self._root_pos_w_sum * raw_root_pos_dist +
            self._rpos_w_sum * raw_rpos_dist +
            self._rquat_w_sum * raw_rangles_dist +
            self._rvel_w_sum * raw_rvel_rot_dist +
            self._rvel_w_sum * raw_rvel_lin_dist +
            self._absolute_site_w_sum * raw_absolute_site_dist
        )

        # Root velocity tracking reward
        # Always compute if free joint exists (weight from carry handles enable/disable)
        root_vel_reward = 0.0
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

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
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
        action_margin = 0.02 * backend.maximum(action_high - action_low, 1e-6)
        action_saturation_fraction = backend.mean(
            (action <= action_low + action_margin)
            | (action >= action_high - action_margin)
        )

        # action rate penalty
        if self._action_rate_coeff > 0.0:
            action_rate_penalty = -action_rate_norm
        else:
            action_rate_penalty = 0.0

        # activation energy penalty
        activation_energy = (
            backend.mean(backend.square(data.act)) if data.act.size > 0 else 0.0
        )
        if self._activation_energy_coeff > 0.0:
            activation_energy_penalty = -activation_energy
        else:
            activation_energy_penalty = 0.0

        # total penalties (coefficient applied once here)
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_penalty
                            + self._joint_torque_coeff * torque_penalty
                            + self._action_rate_coeff * action_rate_penalty
                            + self._activation_energy_coeff * activation_energy_penalty)
        total_penalities = backend.maximum(total_penalities, -1.0)

        # --- Contact tracking rewards (all JAX-native for JIT safety) ---
        # Entire block (including carry weight reads) is gated on contact tracking being
        # attached, so non-contact configs and carries without contact fields are unaffected.
        new_foot_xpos = reward_state.last_foot_xpos
        contact_reward = 0.0

        if self._ctd_stance_mask is not None and self._foot_site_ids is not None:
            ref_frame = jnp.clip(
                jnp.round(carry.traj_state.subtraj_step_no * self._ctd_eff_stride).astype(jnp.int32),
                0, self._ctd_num_frames - 1,
            )
            stance = self._ctd_stance_mask[ref_frame]
            n_stance = jnp.sum(stance)

            actual_feet_z = data.site_xpos[self._foot_site_ids, 2]
            ref_feet_z = self._ctd_foot_z[ref_frame]
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

            body_graph_reward = 0.0  # body-graph laplacian reward not yet implemented

            contact_reward = (
                carry.foot_contact_height_w_sum * foot_height_reward
                + carry.foot_contact_velocity_w_sum * foot_velocity_reward
                + carry.body_graph_w_sum * body_graph_reward
            )

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + qvel_w_sum * qvel_reward
                        + self._root_pos_w_sum * root_pos_reward
                        + root_vel_w_sum * root_vel_reward
                        + self._absolute_site_w_sum * absolute_site_reward
                        + contact_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

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
            err_joint_pos = backend.sqrt(backend.mean(backend.square(
                qpos[self._joint_qpos_mask] - qpos_traj[self._joint_qpos_mask])))
        if np.any(self._joint_qvel_mask):
            err_joint_vel = backend.sqrt(backend.mean(backend.square(
                qvel[self._joint_qvel_mask] - qvel_traj[self._joint_qvel_mask])))
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
                    cur_sites[self._right_hand_rel_index]
                    - ref_sites[self._right_hand_rel_index]
                )
            # Relative site position error (RMSE of site_rpos)
            err_rpos = backend.sqrt(backend.mean(backend.square(site_rpos - site_rpos_traj)))

        # Build reward_info for logging/diagnostics
        reward_info = {
            "reward_total": total_reward,
            "reward_qpos": qpos_reward,
            "reward_qvel": qvel_reward,
            "reward_root_pos": root_pos_reward,
            "reward_root_vel": root_vel_reward,
            "penalty_total": total_penalities,
            "penalty_activation_energy": self._activation_energy_coeff * activation_energy_penalty,
            "activation_energy": activation_energy,
            "action_saturation_fraction": action_saturation_fraction,
            "action_rate_mean_square": action_rate_norm,
            "err_root_xyz": err_root_xyz,
            "err_root_yaw": err_root_yaw,
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

    def __init__(self, env: Any,
                 racket_site_name: str = "racket_stringbed_center_site",
                 racket_hand_site_name: str = "right_hand_mimic",
                 racket_pos_w_sum: float = 0.3,
                 racket_pos_w_exp: float = 50.0,
                 racket_rot_w_sum: float = 0.15,
                 racket_rot_w_exp: float = 5.0,
                 finger_grip_w_sum: float = 0.2,
                 finger_grip_w_exp: float = 10.0,
                 joints_for_mimic=None,
                 **kwargs):
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
        self._racket_traj_query_ids = np.array(
            [self._racket_hand_site_id, self._racket_main_site_id], dtype=int
        )

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

    def derive_reference_racket_pose(self, hand_site_pos, hand_site_mat, backend=np):
        """Reference racket site pose implied by the rigid grip from a hand site pose."""
        pos = hand_site_pos + hand_site_mat @ self._racket_off_pos
        mat = hand_site_mat @ self._racket_off_mat
        return pos, mat

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
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

        if self._racket_pos_w_sum <= 0.0 and self._racket_rot_w_sum <= 0.0:
            if finger_active:
                reward_state = carry.reward_state
                reward_state = reward_state.replace(
                    imitation_error_total=reward_state.imitation_error_total + finger_extra_err
                )
                carry = carry.replace(reward_state=reward_state)
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
        ref_racket_pos, ref_racket_mat = self.derive_reference_racket_pose(
            ref_hand_pos, ref_hand_mat, backend
        )

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
        racket_reward = (self._racket_pos_w_sum * racket_pos_reward
                         + self._racket_rot_w_sum * racket_rot_reward)
        racket_reward = backend.nan_to_num(racket_reward, nan=0.0)

        total_reward = total_reward + racket_reward

        # keep adaptive trajectory sampling consistent with the extra tracking terms
        reward_state = carry.reward_state
        reward_state = reward_state.replace(
            imitation_error_total=reward_state.imitation_error_total
            + self._racket_pos_w_sum * raw_racket_pos_dist
            + self._racket_rot_w_sum * raw_racket_rot_dist
            + finger_extra_err
        )
        carry = carry.replace(reward_state=reward_state)

        reward_info["reward_racket_pos"] = racket_pos_reward
        reward_info["reward_racket_rot"] = racket_rot_reward
        reward_info["err_racket_pos"] = backend.sqrt(raw_racket_pos_dist)
        reward_info["err_racket_rot"] = backend.sqrt(raw_racket_rot_dist)
        reward_info["reward_total"] = total_reward

        return total_reward, carry, reward_info


TargetVelocityTrajReward.register()
MimicReward.register()
RacketMimicReward.register()
