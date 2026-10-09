"""Two-player forehand-clear rally RL environment.

Two MyoFullBody players stand in opposite backcourts and exchange forehand
clears over one shuttle.  Both players are driven every control step:
``step`` takes ``{"p1": action, "p2": action}`` (each action in [-1, 1] over
that player's 354 muscle actuators) and returns per-player observations and
rewards, PettingZoo-parallel style.

Player 2's observation is expressed in a mirrored frame (180-degree rotation
about the world z-axis through the net), so both players see themselves on
the -x half attacking toward +x.  A single shared policy can therefore drive
both sides (self-play) or two separate policies can be trained.

Episode flow: a feeder serve is launched toward the receiver's backcourt hit
window; players must alternate legal stringbed hits; the episode ends when
the shuttle lands (deep-clear landing regions are scored), when a player hits
out of turn, when a body falls, or when the rally/step caps are reached.

Physics: ``RallyBadmintonPhysics`` (v2 aero with skirt cross-flow and fin
damping, swept-crossing anti-tunneling, speed-dependent restitution, cork
angular-impulse closure) plus the restored physical shuttle inertia.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.double_play.src.build_double_play_scene import (
    DOUBLE_READY_KEYFRAME,
    default_double_play_scene_path,
)
from environment.double_play.src.rally_physics import (
    RallyBadmintonPhysics,
    RallyPhysicsConfig,
)
from environment.overall_environment.src.control_scaling import (
    normalized_action_to_model_ctrl,
)
from environment.overall_environment.src.incoming_shuttle_hit_env import (
    classify_return_net_crossing,
)
from environment.overall_environment.src.shuttle_feeder import (
    FeedConfig,
    FeedSample,
    HitWindow,
    launch_quat_from_velocity,
    sample_feed,
)
from environment.overall_environment.src.static_forehand_clear_env import (
    FlightRegion,
    classify_landing_region,
)
from environment.shuttlecock.src.shuttlecock_aero import (
    restore_shuttle_inertia,
    sample_randomized_aero_config,
)

PLAYERS = ("p1", "p2")
GROUND_REST_HEIGHT_M = 0.035
LANDED_HEIGHT_M = 0.06
BODY_FALL_ROOT_HEIGHT_M = 0.55
POSTURE_ROOT_HEIGHT_M = 0.85
Z180_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

# Serve generator tuned for backcourt-to-backcourt clears (players at |x|=4.6).
DOUBLE_PLAY_FEED_CONFIG = FeedConfig(
    launch_x_range=(3.8, 5.6),
    launch_y_range=(-0.8, 0.8),
    launch_z_range=(1.6, 2.2),
    speed_range=(21.0, 29.0),
    elevation_deg_range=(30.0, 50.0),
    azimuth_jitter_deg=4.0,
    net_clearance_height=1.75,
    intercept_time_range_s=(1.2, 2.6),
    intercept_vertical_velocity_range_m_s=(-8.5, -1.5),
    apex_height_range_m=(3.0, 7.5),
    target_height_tolerance_m=0.12,
    max_flight_time=3.5,
)
DOUBLE_PLAY_HIT_WINDOW = HitWindow(
    x_range=(-5.15, -3.95),
    y_range=(-0.8, 0.8),
    z_range=(1.6, 2.4),
)

# Deep-clear landing scores from the hitter's perspective.
CLEAR_REGION_SCORES: dict[str, float] = {
    FlightRegion.OPPONENT_BACK.value: 1.0,
    FlightRegion.OPPONENT_MID.value: 0.3,
    FlightRegion.NET_FRONT.value: -0.2,
    FlightRegion.OWN_SIDE.value: -1.0,
    FlightRegion.OUT.value: -0.75,
}

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    # dense shaping while it is my turn and the shuttle is inbound
    "approach": 1.0,
    # event rewards
    "hit_bonus": 5.0,
    "wrong_hitter": 5.0,  # penalty magnitude
    "crossed_net": 2.0,
    "invalid_net_crossing": 2.0,  # penalty magnitude
    "rally_continue": 2.0,  # my clear was successfully returned: rally goes on
    # flight-quality rewards, settled per hit
    "clear_apex": 1.0,
    "landing_region": 4.0,
    "miss": 3.0,  # penalty: the serve/return landed untouched on my side
    # regularizers
    "effort": 0.01,
    "posture": 0.25,
    "body_fall": 10.0,
}

APPROACH_SOFTNESS_M = 0.35
CLEAR_APEX_MIN_M = 2.5
CLEAR_APEX_FULL_M = 5.0


class RallyState(str, Enum):
    RALLY = "rally"
    DONE = "done"


@dataclass
class _PlayerBinding:
    name: str
    prefix: str
    half_sign: int
    mirrored: bool
    root_joint: str
    racket_body_name: str
    stringbed_site: str
    root_qadr: int = 0
    root_dadr: int = 0
    racket_body: int = 0
    stringbed_site_id: int = 0
    actuator_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    hinge_qpos_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    hinge_qvel_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


@dataclass
class _FlightRecord:
    hitter: str | None
    apex_m: float
    crossed: bool = False
    invalid_crossing: bool = False


def _mirror_vec(value: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(value, dtype=float).copy()
    mirrored[0] = -mirrored[0]
    mirrored[1] = -mirrored[1]
    return mirrored


def _mirror_quat(quat: np.ndarray) -> np.ndarray:
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, Z180_QUAT, np.asarray(quat, dtype=float))
    return _canonical_quat(rotated)


def _canonical_quat(quat: np.ndarray) -> np.ndarray:
    """Fix the quaternion double-cover sign so mirrored observations align."""
    quat = np.asarray(quat, dtype=float)
    return -quat if quat[0] < 0.0 else quat


class DoublePlayRallyEnv:
    """Two-player forehand-clear rally environment (CPU MuJoCo)."""

    def __init__(
        self,
        xml: str | Path | None = None,
        *,
        physics_config: RallyPhysicsConfig | None = None,
        control_substeps: int = 10,
        max_episode_steps: int = 600,
        max_rally_hits: int = 20,
        reward_weights: dict[str, float] | None = None,
        feed_config: FeedConfig | None = None,
        hit_window: HitWindow | None = None,
        serve_receiver: str = "random",  # "random" | "p1" | "p2"
        net_height_m: float = 1.55,
        min_net_clearance_m: float = 0.0,
        terminate_on_body_fall: bool = True,
        aero_domain_randomization: bool = False,
        seed: int = 0,
    ) -> None:
        self.xml_path = Path(xml) if xml is not None else default_double_play_scene_path()
        if not self.xml_path.is_file():
            raise FileNotFoundError(
                f"double-play scene XML not found: {self.xml_path}; "
                "run environment.double_play.src.build_double_play_scene first"
            )
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        restore_shuttle_inertia(self.model, "overall_shuttle")
        self.data = mujoco.MjData(self.model)
        self.physics = RallyBadmintonPhysics(physics_config)
        self.control_substeps = int(control_substeps)
        self.max_episode_steps = int(max_episode_steps)
        self.max_rally_hits = int(max_rally_hits)
        if self.control_substeps <= 0:
            raise ValueError("control_substeps must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.max_rally_hits <= 0:
            raise ValueError("max_rally_hits must be positive")
        weights = dict(DEFAULT_REWARD_WEIGHTS)
        if reward_weights:
            unknown = set(reward_weights) - set(weights)
            if unknown:
                raise ValueError(f"unknown reward weights: {sorted(unknown)}")
            weights.update({key: float(value) for key, value in reward_weights.items()})
        self.reward_weights = weights
        self.feed_config = feed_config if feed_config is not None else DOUBLE_PLAY_FEED_CONFIG
        self.hit_window = hit_window if hit_window is not None else DOUBLE_PLAY_HIT_WINDOW
        if serve_receiver not in {"random", *PLAYERS}:
            raise ValueError("serve_receiver must be 'random', 'p1', or 'p2'")
        self.serve_receiver_mode = serve_receiver
        self.net_height_m = float(net_height_m)
        self.min_net_clearance_m = float(min_net_clearance_m)
        self.terminate_on_body_fall = bool(terminate_on_body_fall)
        self.aero_domain_randomization = bool(aero_domain_randomization)
        self.rng = np.random.default_rng(seed)

        self.keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, DOUBLE_READY_KEYFRAME
        )
        if self.keyframe_id < 0:
            raise ValueError(f"missing keyframe {DOUBLE_READY_KEYFRAME!r} in {self.xml_path}")

        shuttle_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free"
        )
        if shuttle_joint < 0:
            raise ValueError("missing joint 'overall_shuttle_free'")
        self._shuttle_qadr = int(self.model.jnt_qposadr[shuttle_joint])
        self._shuttle_dadr = int(self.model.jnt_dofadr[shuttle_joint])

        self.players: dict[str, _PlayerBinding] = {
            "p1": _PlayerBinding(
                name="p1",
                prefix="",
                half_sign=-1,
                mirrored=False,
                root_joint="root",
                racket_body_name="overall_racket",
                stringbed_site="overall_stringbed_center_site",
            ),
            "p2": _PlayerBinding(
                name="p2",
                prefix="p2_",
                half_sign=1,
                mirrored=True,
                root_joint="p2_root",
                racket_body_name="p2_overall_racket",
                stringbed_site="p2_overall_stringbed_center_site",
            ),
        }
        self._racket_to_player = {
            binding.racket_body_name: name for name, binding in self.players.items()
        }
        self._bind_players()
        self._full_action = np.zeros(self.model.nu, dtype=float)

        self.state = RallyState.DONE
        self.step_index = 0
        self.rally_hits = 0
        self.serve_receiver: str | None = None
        self.expected_hitter: str | None = None
        self.last_hitter: str | None = None
        self.termination_reason: str | None = None
        self.feed: FeedSample | None = None
        self._flight: _FlightRecord | None = None
        self._approach_best: float = 0.0
        self._landing_region: str | None = None
        self._landing_xy: np.ndarray | None = None

    # ---- setup helpers ----------------------------------------------------

    def _bind_players(self) -> None:
        model = self.model
        for binding in self.players.values():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, binding.root_joint)
            if joint_id < 0:
                raise ValueError(f"missing joint {binding.root_joint!r}")
            binding.root_qadr = int(model.jnt_qposadr[joint_id])
            binding.root_dadr = int(model.jnt_dofadr[joint_id])
            binding.racket_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, binding.racket_body_name
            )
            if binding.racket_body < 0:
                raise ValueError(f"missing body {binding.racket_body_name!r}")
            binding.stringbed_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, binding.stringbed_site
            )
            if binding.stringbed_site_id < 0:
                raise ValueError(f"missing site {binding.stringbed_site!r}")

        actuator_owner: dict[str, list[int]] = {name: [] for name in PLAYERS}
        for actuator_id in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if name is None:
                raise ValueError("every actuator must be named")
            owner = "p2" if name.startswith("p2_") else "p1"
            actuator_owner[owner].append(actuator_id)
        p1_count, p2_count = len(actuator_owner["p1"]), len(actuator_owner["p2"])
        if p1_count == 0 or p1_count != p2_count:
            raise ValueError(f"asymmetric actuator split: p1={p1_count}, p2={p2_count}")

        hinge_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        hinge_qpos: dict[str, list[int]] = {name: [] for name in PLAYERS}
        hinge_qvel: dict[str, list[int]] = {name: [] for name in PLAYERS}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None or name == "overall_shuttle_free":
                continue
            if int(model.jnt_type[joint_id]) not in hinge_types:
                continue
            owner = "p2" if name.startswith("p2_") else "p1"
            hinge_qpos[owner].append(int(model.jnt_qposadr[joint_id]))
            hinge_qvel[owner].append(int(model.jnt_dofadr[joint_id]))
        if len(hinge_qpos["p1"]) != len(hinge_qpos["p2"]) or not hinge_qpos["p1"]:
            raise ValueError("asymmetric joint split between players")

        for name, binding in self.players.items():
            binding.actuator_indices = np.asarray(actuator_owner[name], dtype=int)
            binding.hinge_qpos_indices = np.asarray(hinge_qpos[name], dtype=int)
            binding.hinge_qvel_indices = np.asarray(hinge_qvel[name], dtype=int)

    # ---- public sizes -----------------------------------------------------

    @property
    def action_size(self) -> int:
        return int(self.players["p1"].actuator_indices.size)

    @property
    def observation_size(self) -> int:
        binding = self.players["p1"]
        hinges = binding.hinge_qpos_indices.size + binding.hinge_qvel_indices.size
        # root(1z+4quat+2xy+3linvel+3angvel) + shuttle(12) + racket(9) + task(6)
        return int(hinges + 13 + 12 + 9 + 6)

    # ---- serve ------------------------------------------------------------

    def _sample_serve(self, receiver: str) -> FeedSample:
        feed = sample_feed(self.rng, self.feed_config, self.hit_window)
        if receiver == "p2":
            feed = FeedSample(
                launch_pos=_mirror_vec(feed.launch_pos),
                launch_vel=_mirror_vec(feed.launch_vel),
                trajectory=feed.trajectory,
                intercept_index=feed.intercept_index,
                intercept_point=_mirror_vec(feed.intercept_point),
                intercept_velocity=_mirror_vec(feed.intercept_velocity),
                intercept_time_s=feed.intercept_time_s,
            )
        return feed

    # ---- lifecycle --------------------------------------------------------

    def reset(self, *, serve_receiver: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        if self.aero_domain_randomization:
            self.physics.cfg.aero = sample_randomized_aero_config(
                self.rng, base=self.physics.cfg.aero
            )

        mode = serve_receiver if serve_receiver is not None else self.serve_receiver_mode
        if mode == "random":
            receiver = str(self.rng.choice(PLAYERS))
        elif mode in PLAYERS:
            receiver = mode
        else:
            raise ValueError("serve_receiver must be 'random', 'p1', or 'p2'")
        self.serve_receiver = receiver
        self.feed = self._sample_serve(receiver)

        qadr, dadr = self._shuttle_qadr, self._shuttle_dadr
        self.data.qpos[qadr : qadr + 3] = self.feed.launch_pos
        self.data.qpos[qadr + 3 : qadr + 7] = launch_quat_from_velocity(self.feed.launch_vel)
        self.data.qvel[dadr : dadr + 3] = self.feed.launch_vel
        self.data.qvel[dadr + 3 : dadr + 6] = 0.0

        self.physics.reset()
        self.state = RallyState.RALLY
        self.step_index = 0
        self.rally_hits = 0
        self.expected_hitter = receiver
        self.last_hitter = None
        self.termination_reason = None
        self._flight = _FlightRecord(hitter=None, apex_m=float(self.data.qpos[qadr + 2]))
        self._approach_best = 0.0
        self._landing_region = None
        self._landing_xy = None

        mujoco.mj_forward(self.model, self.data)
        return self._observations(), self._info({})

    # ---- frame helpers ----------------------------------------------------

    def _shuttle_pos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self._shuttle_qadr : self._shuttle_qadr + 3], dtype=float)

    def _shuttle_vel(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self._shuttle_dadr : self._shuttle_dadr + 3], dtype=float)

    def _stringbed_pos(self, binding: _PlayerBinding) -> np.ndarray:
        return np.asarray(self.data.site_xpos[binding.stringbed_site_id], dtype=float)

    def _stringbed_normal(self, binding: _PlayerBinding) -> np.ndarray:
        mat = np.asarray(self.data.site_xmat[binding.stringbed_site_id], dtype=float).reshape(3, 3)
        return mat[:, 2]

    def _stringbed_vel(self, binding: _PlayerBinding) -> np.ndarray:
        vel6 = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, binding.racket_body, vel6, 0
        )
        omega, v_origin = vel6[:3], vel6[3:]
        origin = np.asarray(self.data.xpos[binding.racket_body], dtype=float)
        point = self._stringbed_pos(binding)
        return v_origin + np.cross(omega, point - origin)

    def _root_height(self, binding: _PlayerBinding) -> float:
        return float(self.data.qpos[binding.root_qadr + 2])

    # ---- observations -----------------------------------------------------

    def _observation_for(self, binding: _PlayerBinding) -> np.ndarray:
        data = self.data
        qadr = binding.root_qadr
        dadr = binding.root_dadr
        root_pos = np.asarray(data.qpos[qadr : qadr + 3], dtype=float)
        root_quat = np.asarray(data.qpos[qadr + 3 : qadr + 7], dtype=float)
        root_linvel = np.asarray(data.qvel[dadr : dadr + 3], dtype=float)
        root_angvel_local = np.asarray(data.qvel[dadr + 3 : dadr + 6], dtype=float)

        shuttle_pos = self._shuttle_pos()
        shuttle_vel = self._shuttle_vel()
        stringbed_pos = self._stringbed_pos(binding)
        stringbed_normal = self._stringbed_normal(binding)
        stringbed_vel = self._stringbed_vel(binding)

        if binding.mirrored:
            root_pos = _mirror_vec(root_pos)
            root_quat = _mirror_quat(root_quat)
            root_linvel = _mirror_vec(root_linvel)
        else:
            root_quat = _canonical_quat(root_quat)
        if binding.mirrored:
            shuttle_pos = _mirror_vec(shuttle_pos)
            shuttle_vel = _mirror_vec(shuttle_vel)
            stringbed_pos = _mirror_vec(stringbed_pos)
            stringbed_normal = _mirror_vec(stringbed_normal)
            stringbed_vel = _mirror_vec(stringbed_vel)
        # joint coordinates and the body-local root angular velocity are frame
        # invariant under the mirror rotation, so they are used raw.

        hinge_qpos = np.asarray(data.qpos, dtype=float)[binding.hinge_qpos_indices]
        hinge_qvel = np.asarray(data.qvel, dtype=float)[binding.hinge_qvel_indices]

        my_turn = 1.0 if self.expected_hitter == binding.name else 0.0
        elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        task = np.array(
            [
                shuttle_pos[0],
                shuttle_pos[1],
                shuttle_pos[2],
                my_turn,
                min(1.0, elapsed / 10.0),
                self.rally_hits / float(self.max_rally_hits),
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                [root_pos[2]],
                root_quat,
                root_pos[:2],
                root_linvel,
                root_angvel_local,
                hinge_qpos,
                hinge_qvel,
                shuttle_pos - root_pos,
                shuttle_vel,
                shuttle_pos - stringbed_pos,
                shuttle_vel - stringbed_vel,
                stringbed_pos - root_pos,
                stringbed_normal,
                stringbed_vel,
                task,
            ]
        )

    def _observations(self) -> dict[str, np.ndarray]:
        return {name: self._observation_for(binding) for name, binding in self.players.items()}

    # ---- step -------------------------------------------------------------

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        if self.state != RallyState.RALLY:
            raise RuntimeError("call reset() before step()")
        if set(actions) != set(PLAYERS):
            raise ValueError(f"actions must contain exactly {PLAYERS}, got {sorted(actions)}")
        for name in PLAYERS:
            action = np.asarray(actions[name], dtype=float)
            if action.shape != (self.action_size,):
                raise ValueError(
                    f"action[{name!r}] must have shape ({self.action_size},), got {action.shape}"
                )
            if not np.isfinite(action).all():
                raise ValueError(f"action[{name!r}] contains non-finite values")
            self._full_action[self.players[name].actuator_indices] = action
        ctrl = normalized_action_to_model_ctrl(self.model, self._full_action)
        self.data.ctrl[:] = ctrl

        rewards = {name: 0.0 for name in PLAYERS}
        weights = self.reward_weights
        events: list[dict[str, Any]] = []
        previous_shuttle_pos = self._shuttle_pos().copy()

        terminated = False
        truncated = False
        for _ in range(self.control_substeps):
            diag = self.physics.substep(self.model, self.data)
            if diag["event_racket"] is not None:
                events.append(diag["event"])

        self.step_index += 1
        shuttle_pos = self._shuttle_pos()

        # -- hit events -----------------------------------------------------
        for event in events:
            hitter = self._racket_to_player[str(event["racket"])]
            if hitter == self.expected_hitter:
                self._settle_flight(rewards, returned=True)
                rewards[hitter] += weights["hit_bonus"]
                self.rally_hits += 1
                self.last_hitter = hitter
                self.expected_hitter = "p2" if hitter == "p1" else "p1"
                self._flight = _FlightRecord(hitter=hitter, apex_m=float(shuttle_pos[2]))
                self._approach_best = 0.0
            else:
                rewards[hitter] -= weights["wrong_hitter"]
                terminated = True
                self.termination_reason = f"wrong_hitter_{hitter}"
        if self.rally_hits >= self.max_rally_hits and not terminated:
            truncated = True
            self.termination_reason = "max_rally_hits"

        # -- flight tracking ------------------------------------------------
        if self._flight is not None:
            self._flight.apex_m = max(self._flight.apex_m, float(shuttle_pos[2]))
            if self._flight.hitter is not None and not terminated:
                hitter_binding = self.players[self._flight.hitter]
                crossing = classify_return_net_crossing(
                    previous_shuttle_pos,
                    shuttle_pos,
                    player_half_sign=hitter_binding.half_sign,
                    net_x_m=0.0,
                    net_height_m=self.net_height_m,
                    min_clearance_m=self.min_net_clearance_m,
                )
                if bool(crossing["valid"]) and not self._flight.crossed:
                    self._flight.crossed = True
                    rewards[self._flight.hitter] += weights["crossed_net"]
                elif bool(crossing["crossed"]) and not bool(crossing["valid"]):
                    self._flight.invalid_crossing = True
                    rewards[self._flight.hitter] -= weights["invalid_net_crossing"]

        # -- approach shaping for the expected hitter -----------------------
        if not terminated and self.expected_hitter is not None:
            binding = self.players[self.expected_hitter]
            incoming = float(self._shuttle_vel()[0]) * binding.half_sign > 0.0 or (
                float(np.sign(shuttle_pos[0])) == binding.half_sign
            )
            if incoming:
                distance = float(np.linalg.norm(shuttle_pos - self._stringbed_pos(binding)))
                potential = float(np.exp(-distance / APPROACH_SOFTNESS_M))
                if potential > self._approach_best:
                    rewards[self.expected_hitter] += weights["approach"] * (
                        potential - self._approach_best
                    )
                    self._approach_best = potential

        # -- landing --------------------------------------------------------
        landed = float(shuttle_pos[2]) <= LANDED_HEIGHT_M
        if landed and not terminated:
            self._landing_xy = np.asarray(shuttle_pos[:2], dtype=float).copy()
            if self._flight is None or self._flight.hitter is None:
                receiver = self.expected_hitter or self.serve_receiver or "p1"
                rewards[receiver] -= weights["miss"]
                self.termination_reason = f"receiver_miss_{receiver}"
            else:
                self._settle_flight(rewards, returned=False)
                self.termination_reason = "landed"
            terminated = True

        # -- regularizers and falls -----------------------------------------
        for name, binding in self.players.items():
            player_ctrl = ctrl[binding.actuator_indices]
            rewards[name] -= weights["effort"] * float(np.mean(np.square(player_ctrl)))
            root_height = self._root_height(binding)
            rewards[name] -= weights["posture"] * max(0.0, POSTURE_ROOT_HEIGHT_M - root_height)
            if root_height < BODY_FALL_ROOT_HEIGHT_M:
                rewards[name] -= weights["body_fall"]
                if self.terminate_on_body_fall:
                    terminated = True
                    self.termination_reason = f"body_fall_{name}"

        if not terminated and self.step_index >= self.max_episode_steps:
            truncated = True
            if self.termination_reason is None:
                self.termination_reason = "max_steps"

        if terminated or truncated:
            self.state = RallyState.DONE

        info = self._info(
            {
                "events": events,
                "landed": landed,
                "landing_region": self._landing_region,
            }
        )
        return self._observations(), rewards, terminated, truncated, info

    # ---- flight settlement ------------------------------------------------

    def _settle_flight(self, rewards: dict[str, float], *, returned: bool) -> None:
        """Score the completed flight segment for its hitter.

        ``returned=True``: the opponent legally returned this hit (settled at
        the moment of their hit).  ``returned=False``: the shuttle landed.
        """
        flight = self._flight
        if flight is None or flight.hitter is None:
            return
        weights = self.reward_weights
        hitter = flight.hitter
        apex_quality = float(
            np.clip(
                (flight.apex_m - CLEAR_APEX_MIN_M) / (CLEAR_APEX_FULL_M - CLEAR_APEX_MIN_M),
                0.0,
                1.0,
            )
        )
        if flight.crossed:
            rewards[hitter] += weights["clear_apex"] * apex_quality
        if returned:
            rewards[hitter] += weights["rally_continue"]
        else:
            landing_xy = self._landing_xy
            if landing_xy is not None:
                region = classify_landing_region(
                    landing_xy,
                    player_half_sign=self.players[hitter].half_sign,
                    singles=False,
                )
                self._landing_region = region.value
                rewards[hitter] += weights["landing_region"] * CLEAR_REGION_SCORES[region.value]
        self._flight = None

    # ---- info -------------------------------------------------------------

    def _info(self, extra: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "state": self.state.value,
            "step_index": self.step_index,
            "rally_hits": self.rally_hits,
            "serve_receiver": self.serve_receiver,
            "expected_hitter": self.expected_hitter,
            "last_hitter": self.last_hitter,
            "termination_reason": self.termination_reason,
            "shuttle_pos": self._shuttle_pos().copy(),
            "shuttle_vel": self._shuttle_vel().copy(),
        }
        info.update(extra)
        return info
