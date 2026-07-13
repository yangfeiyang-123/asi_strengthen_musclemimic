"""
MyoFullBodyRacket environment - MyoFullBody holding a rigid badminton racket.

The racket asset (``environment/racket/assets/badminton_racket_rigid.xml``) is
attached as a *jointless* rigid child body of the right-hand third metacarpal
``thirdmc_r`` (the palm body used as ``body1`` of the Overall scene's hand-racket
soft weld). Because no joint is added, ``qpos``/``qvel``/``nu`` are identical to
plain :class:`MyoFullBody`, so existing retargeted free-hand trajectories,
observation/action spaces, and trained body policies transfer unchanged. The
rigid attachment is the physical limit of the downstream ``soft_weld_schedule``
``strong_weld`` stage, so a body policy pretrained here drops into the Overall
badminton (hitting) scene at its first curriculum stage.

The racket collision geoms are moved to a dedicated collision bit so the racket
never contacts the human body (or anything else) during trajectory imitation.
This keeps the model ``mjx.put_model``-compatible on both the ``jax`` and
``warp`` backends with no extra constraint/contact budget.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec

from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__, identifier="[MyoFullBodyRacket]")


# Repo root = .../musclemimic (parents: humanoids -> environments -> musclemimic pkg -> repo root)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RACKET_XML = _REPO_ROOT / "environment" / "racket" / "assets" / "badminton_racket_rigid.xml"

# Optimized right-hand finger angles for the standard forehand grip (same
# reference the Overall hitting scene uses): fingers wrap the handle, thumb lies
# on the wide face, palm stays slightly hollow. The stored trajectories were
# retargeted for the bare-hand MyoFullBody and contain no finger joints, so when
# the fingers are enabled the trajectory handler zero-fills them -> a flat, open
# hand that fights the racket. We instead pin the fingers to this closed grip
# both at reset and in the reward.
DEFAULT_GRIP_REFERENCE_JSON = _REPO_ROOT / "configs" / "right_hand_racket_grip_reference.json"

# Body the racket is rigidly fixed to. ``thirdmc_r`` (palm / 3rd metacarpal) matches
# the Overall scene's ``SOFT_WELD_BODY1``; it survives ``disable_fingers`` (only the
# finger joints/muscles are removed, the bodies remain, fixed to their parent).
DEFAULT_RACKET_ATTACH_BODY = "thirdmc_r"

# Racket butt-cap pose expressed in ``thirdmc_r`` local frame. Derived from
# ``configs/right_hand_racket_grip_reference.json`` (the standard right-hand
# forehand grip the Overall scene's hand-racket weld enforces: handle diagonal
# from the index-finger base to the heel of the palm, tiger mouth on the +45 deg
# bevel, thumb flat on the wide face, fingers wrapped, palm slightly hollow), so
# the pretrained grip pose matches the downstream hitting scene. Override via
# ``racket_grip_pos``/``racket_grip_quat``.
DEFAULT_RACKET_GRIP_POS = (-0.03196, -0.09715, 0.02468)
DEFAULT_RACKET_GRIP_QUAT = (0.594207, -0.036689, 0.609628, -0.52338)

# Collision bit for the racket geoms. Human body geoms only use bit 1, so bit 4
# (0b100) is disjoint and the racket produces no contact pairs with the person.
DEFAULT_RACKET_COLLISION_BIT = 4

# Attach prefix and resulting racket body name after ``spec.attach(..., prefix=...)``.
RACKET_ATTACH_PREFIX = "racket_"
RACKET_BODY_NAME = "racket_racket"


@lru_cache(maxsize=4)
def _load_grip_finger_angles(grip_json_path: str) -> tuple:
    """Return ((joint_name, angle), ...) for the right-hand finger grip pose.

    Read from ``right_hand_racket_grip_reference.json`` by joint *name* so it is
    robust to qpos layout. The JSON stores the full grip-scene qpos plus the
    ordered ``right_hand_joint_names``; the grip scene shares the MyoFullBody
    finger addresses, so the finger angle for each name is ``qpos[addr]``.
    """
    ref = json.loads(Path(grip_json_path).read_text())
    names = list(ref["right_hand_joint_names"])
    qpos = np.asarray(ref["qpos"], dtype=float)
    # The grip scene is a MyoFullBody variant; its right-hand finger joints sit at
    # the same contiguous qpos block the full-body model uses. We read them via a
    # throwaway load of the referenced scene to avoid hard-coding addresses.
    scene_xml = _REPO_ROOT / "assets" / "right_hand_racket_grip_scene.xml"
    if scene_xml.is_file():
        m = mujoco.MjModel.from_xml_path(str(scene_xml))
        d = mujoco.MjData(m)
        d.qpos[: len(qpos)] = qpos
        mujoco.mj_forward(m, d)
        out = []
        for n in names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                raise ValueError(f"grip reference joint {n!r} missing from {scene_xml}")
            out.append((n, float(d.qpos[m.jnt_qposadr[jid]])))
        return tuple(out)
    # Fallback: the reference JSON is emitted with finger angles contiguous right
    # after the 43-dim spine/arm block for the standard MyoFullBody skeleton.
    return tuple((n, float(v)) for n, v in zip(names, qpos[43 : 43 + len(names)]))


def grip_finger_reference(model, grip_json_path: str | Path = DEFAULT_GRIP_REFERENCE_JSON):
    """Map the grip pose onto ``model``'s right-hand finger joints.

    Returns ``(joint_names, qpos_addrs, target_angles)`` restricted to the finger
    joints actually present in ``model`` (empty arrays when fingers are disabled).
    """
    angles = _load_grip_finger_angles(str(grip_json_path))
    names, addrs, targets = [], [], []
    for name, angle in angles:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue  # fingers disabled -> joint not in model
        names.append(name)
        addrs.append(int(model.jnt_qposadr[jid]))
        targets.append(angle)
    return names, np.asarray(addrs, dtype=int), np.asarray(targets, dtype=float)


def inject_racket(
    spec: MjSpec,
    *,
    attach_body: str = DEFAULT_RACKET_ATTACH_BODY,
    grip_pos=DEFAULT_RACKET_GRIP_POS,
    grip_quat=DEFAULT_RACKET_GRIP_QUAT,
    collision_bit: int = DEFAULT_RACKET_COLLISION_BIT,
    mass_scale: float = 1.0,
    racket_xml_path: str | Path | None = None,
) -> MjSpec:
    """Attach the rigid racket to ``attach_body`` as a jointless child body.

    Args:
        spec: The MyoFullBody model specification to modify in place.
        attach_body: Body the racket is rigidly fixed to (default ``thirdmc_r``).
        grip_pos: Racket butt-cap position in ``attach_body`` local frame.
        grip_quat: Racket orientation quaternion (wxyz) in ``attach_body`` local frame.
        collision_bit: contype/conaffinity bit for the racket collision geoms.
        mass_scale: Multiplier on the racket body mass and inertia. Use <1 to ramp
            the racket load in from a light racket (Stage-2 mass curriculum).
        racket_xml_path: Racket asset path. Defaults to the repo rigid racket.

    Returns:
        The modified spec (same object).
    """
    xml_path = Path(racket_xml_path) if racket_xml_path is not None else DEFAULT_RACKET_XML
    if not xml_path.is_file():
        raise FileNotFoundError(f"racket asset not found: {xml_path}")
    if mass_scale <= 0.0:
        raise ValueError(f"mass_scale must be > 0, got {mass_scale}")

    hand = spec.body(attach_body)
    if hand is None:
        raise ValueError(f"attach body {attach_body!r} not present in MyoFullBody spec")

    racket_spec = mujoco.MjSpec.from_file(str(xml_path))

    # Scale the racket's mass and inertia (its inertial element lives on the root
    # body). Scaling both keeps the density-implied dynamics consistent.
    if mass_scale != 1.0:
        for body in racket_spec.worldbody.bodies:
            body.mass = float(body.mass) * mass_scale
            body.inertia = [float(v) * mass_scale for v in body.inertia]

    # Drop the free joint so the racket becomes a rigid extension of the hand
    # (no added DOF -> qpos/qvel unchanged).
    for joint in list(racket_spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            racket_spec.delete(joint)

    # The asset spawns the free racket at a world pose (e.g. pos="0 0 1.2");
    # with the freejoint that pos is just the initial qpos, but once the joint
    # is dropped it becomes a fixed offset composed with the attach frame,
    # floating the racket ~1.2 m away from the hand. Zero the root body pose so
    # the racket butt cap lands exactly on the grip frame.
    for body in racket_spec.worldbody.bodies:
        body.pos = [0.0, 0.0, 0.0]
        body.quat = [1.0, 0.0, 0.0, 0.0]

    # Move every colliding racket geom to the isolated collision bit; leave
    # visual-only geoms (contype=conaffinity=0) untouched.
    for geom in racket_spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = collision_bit
            geom.conaffinity = collision_bit

    frame = hand.add_frame(pos=list(grip_pos), quat=list(grip_quat))
    spec.attach(racket_spec, frame=frame, prefix=RACKET_ATTACH_PREFIX)
    logger.info(
        "Attached rigid racket to %r (collision bit %d, prefix %r)",
        attach_body,
        collision_bit,
        RACKET_ATTACH_PREFIX,
    )
    return spec


class _RacketConfigMixin:
    """Stores racket knobs and injects the racket after base spec changes.

    Both the CPU and MJX racket environments mix this in. It relies on the base
    ``_apply_spec_changes`` (finger disabling, mimic sites, muscle ctrl range)
    running first, then appends the rigid racket. It does not add joints, sites
    to the mimic set, or actuators, so observation/action specs are unchanged.
    """

    def _store_racket_params(
        self,
        *,
        enable_racket: bool,
        racket_attach_body: str,
        racket_grip_pos,
        racket_grip_quat,
        racket_collision_bit: int,
        racket_mass_scale: float,
        racket_xml_path,
    ) -> None:
        self._enable_racket = enable_racket
        self._racket_attach_body = racket_attach_body
        self._racket_grip_pos = racket_grip_pos
        self._racket_grip_quat = racket_grip_quat
        self._racket_collision_bit = racket_collision_bit
        self._racket_mass_scale = racket_mass_scale
        self._racket_xml_path = racket_xml_path

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        spec = super()._apply_spec_changes(spec)
        if getattr(self, "_enable_racket", True):
            spec = inject_racket(
                spec,
                attach_body=self._racket_attach_body,
                grip_pos=self._racket_grip_pos,
                grip_quat=self._racket_grip_quat,
                collision_bit=self._racket_collision_bit,
                mass_scale=self._racket_mass_scale,
                racket_xml_path=self._racket_xml_path,
            )
        return spec

    def _resolve_grip_fingers(self) -> None:
        """Cache the right-hand finger qpos addresses + grip targets for reset and
        reward. Empty when the fingers are disabled."""
        names, addrs, targets = grip_finger_reference(self._model)
        self._grip_finger_names = names
        self._grip_finger_qpos_addrs = addrs
        self._grip_finger_targets = targets

    @property
    def grip_finger_names(self) -> list:
        if not hasattr(self, "_grip_finger_names"):
            self._resolve_grip_fingers()
        return self._grip_finger_names

    @property
    def grip_finger_qpos_addrs(self):
        if not hasattr(self, "_grip_finger_qpos_addrs"):
            self._resolve_grip_fingers()
        return self._grip_finger_qpos_addrs

    @property
    def grip_finger_targets(self):
        if not hasattr(self, "_grip_finger_targets"):
            self._resolve_grip_fingers()
        return self._grip_finger_targets


class MyoFullBodyRacket(_RacketConfigMixin, MyoFullBody):
    """CPU MuJoCo MyoFullBody rigidly holding a badminton racket."""

    mjx_enabled = False

    # The rigid racket adds no joints/mimic sites, so SMPL->robot retargeting is
    # identical to MyoFullBody: reuse its robot conf and GMR cache (see
    # loco_mujoco.smpl.retargeting._resolve_retarget_env_name).
    retarget_as = "MyoFullBody"

    def __init__(
        self,
        *,
        enable_racket: bool = True,
        racket_attach_body: str = DEFAULT_RACKET_ATTACH_BODY,
        racket_grip_pos=DEFAULT_RACKET_GRIP_POS,
        racket_grip_quat=DEFAULT_RACKET_GRIP_QUAT,
        racket_collision_bit: int = DEFAULT_RACKET_COLLISION_BIT,
        racket_mass_scale: float = 1.0,
        racket_xml_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self._store_racket_params(
            enable_racket=enable_racket,
            racket_attach_body=racket_attach_body,
            racket_grip_pos=racket_grip_pos,
            racket_grip_quat=racket_grip_quat,
            racket_collision_bit=racket_collision_bit,
            racket_mass_scale=racket_mass_scale,
            racket_xml_path=racket_xml_path,
        )
        super().__init__(**kwargs)


class MjxMyoFullBodyRacket(_RacketConfigMixin, MjxMyoFullBody):
    """MJX (jax/warp) MyoFullBody rigidly holding a badminton racket."""

    mjx_enabled = True

    # See MyoFullBodyRacket.retarget_as.
    retarget_as = "MyoFullBody"

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        mjx_backend: str = "jax",
        *,
        enable_racket: bool = True,
        racket_attach_body: str = DEFAULT_RACKET_ATTACH_BODY,
        racket_grip_pos=DEFAULT_RACKET_GRIP_POS,
        racket_grip_quat=DEFAULT_RACKET_GRIP_QUAT,
        racket_collision_bit: int = DEFAULT_RACKET_COLLISION_BIT,
        racket_mass_scale: float = 1.0,
        racket_xml_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self._store_racket_params(
            enable_racket=enable_racket,
            racket_attach_body=racket_attach_body,
            racket_grip_pos=racket_grip_pos,
            racket_grip_quat=racket_grip_quat,
            racket_collision_bit=racket_collision_bit,
            racket_mass_scale=racket_mass_scale,
            racket_xml_path=racket_xml_path,
        )
        super().__init__(timestep=timestep, n_substeps=n_substeps, mjx_backend=mjx_backend, **kwargs)
