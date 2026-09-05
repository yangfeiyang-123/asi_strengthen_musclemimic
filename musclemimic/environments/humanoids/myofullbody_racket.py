"""
MyoFullBodyRacket environment - MyoFullBody holding a rigid badminton racket.

The racket asset (``environment/racket/assets/badminton_racket_rigid.xml``) is
attached as a *jointless* rigid child body of the right-hand third metacarpal
``thirdmc_r``. Because no joint is added, ``qpos``/``qvel``/``nu`` are identical to
plain :class:`MyoFullBody`, so existing retargeted free-hand trajectories,
observation/action spaces, and trained body policies transfer unchanged. The
attachment pose, racket inertia, stringbed transform, and contact semantics are
pinned by the same versioned exact-child contract consumed by the Overall
badminton scene.

The racket collision geoms are moved to a dedicated collision bit so the racket
never contacts the human body (or anything else) during trajectory imitation.
This keeps the model ``mjx.put_model``-compatible on both the ``jax`` and
``warp`` backends with no extra constraint/contact budget.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec

from environment.overall_environment.src.racket_attachment import (
    RacketAttachmentContract,
    canonical_json_fingerprint,
    load_racket_attachment_contract,
    validate_racket_spec_against_contract,
)
from musclemimic.badminton.racket_grip_preset import (
    DEFAULT_RACKET_GRIP_PRESET_PATH,
    RACKET_GRIP_PRESET_SCHEMA,
    load_racket_grip_preset,
)
from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__, identifier="[MyoFullBodyRacket]")


# Repo root = .../musclemimic (parents: humanoids -> environments -> musclemimic pkg -> repo root)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RACKET_ATTACHMENT_CONTRACT = load_racket_attachment_contract()
DEFAULT_RACKET_XML = _DEFAULT_RACKET_ATTACHMENT_CONTRACT.asset_path

# Visually accepted all-trajectory grip preset promoted from the interactive
# editor. It binds the default hand-racket translation/orientation and the 20
# right-hand finger targets. Bare-hand trajectories contain no finger joints,
# so reset and reward pin them to these targets when fingers are enabled.
DEFAULT_GRIP_REFERENCE_JSON = DEFAULT_RACKET_GRIP_PRESET_PATH

# Body the racket is rigidly fixed to. ``thirdmc_r`` (palm / 3rd metacarpal) matches
# the Overall scene's ``SOFT_WELD_BODY1``; it survives ``disable_fingers`` (only the
# finger joints/muscles are removed, the bodies remain, fixed to their parent).
DEFAULT_RACKET_ATTACH_BODY = _DEFAULT_RACKET_ATTACHMENT_CONTRACT.parent_body

# Racket butt-cap pose expressed in ``thirdmc_r`` local frame. Derived from
# the promoted all-trajectory grip preset, so the pretrained grip pose matches
# the downstream hitting scene. Override via ``racket_grip_pos`` /
# ``racket_grip_quat`` or another ``racket_grip_preset``.
DEFAULT_RACKET_GRIP_POS = _DEFAULT_RACKET_ATTACHMENT_CONTRACT.relative_position_m
DEFAULT_RACKET_GRIP_QUAT = _DEFAULT_RACKET_ATTACHMENT_CONTRACT.relative_quaternion_wxyz

# Collision bit for the racket geoms. Human body geoms only use bit 1, so bit 4
# (0b100) is disjoint and the racket produces no contact pairs with the person.
DEFAULT_RACKET_COLLISION_BIT = _DEFAULT_RACKET_ATTACHMENT_CONTRACT.racket_collision_bit

# Attach prefix and resulting racket body name after ``spec.attach(..., prefix=...)``.
RACKET_ATTACH_PREFIX = "racket_"
RACKET_BODY_NAME = "racket_racket"


@lru_cache(maxsize=4)
def _load_grip_finger_angles(grip_json_path: str) -> tuple:
    """Return ((joint_name, angle), ...) for the right-hand finger grip pose.

    New global presets store a direct joint-name mapping. Legacy
    ``right_hand_racket_grip_reference.json`` files store a full grip-scene qpos
    plus ordered names; both forms are mapped by joint name.
    """
    ref = json.loads(Path(grip_json_path).read_text())
    if ref.get("schema") == RACKET_GRIP_PRESET_SCHEMA:
        preset = load_racket_grip_preset(grip_json_path)
        return preset.finger_joint_angles_rad
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
    return tuple((n, float(v)) for n, v in zip(names, qpos[43 : 43 + len(names)], strict=False))


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
    attachment_contract: str | Path | RacketAttachmentContract | None = None,
    attach_body: str | None = None,
    grip_pos=None,
    grip_quat=None,
    collision_bit: int | None = None,
    mass_scale: float = 1.0,
    racket_xml_path: str | Path | None = None,
) -> MjSpec:
    """Attach the rigid racket to ``attach_body`` as a jointless child body.

    Args:
        spec: The MyoFullBody model specification to modify in place.
        attachment_contract: Contract object/path. ``None`` loads the canonical
            forehand-clear rigid-v2 contract.
        attach_body: Optional parent-body override.
        grip_pos: Optional racket-position override in ``attach_body`` local frame.
        grip_quat: Optional orientation override (wxyz) in that frame.
        collision_bit: Optional contype/conaffinity bit override.
        mass_scale: Multiplier on the racket body mass and inertia. Use <1 to ramp
            the racket load in from a light racket (Stage-2 mass curriculum).
        racket_xml_path: Racket asset path. Defaults to the repo rigid racket.

    Returns:
        The modified spec (same object).
    """
    if isinstance(attachment_contract, RacketAttachmentContract):
        contract = attachment_contract
        contract.verify_asset()
    else:
        contract = load_racket_attachment_contract(attachment_contract)
    attach_body = contract.parent_body if attach_body is None else attach_body
    grip_pos = contract.relative_position_m if grip_pos is None else grip_pos
    grip_quat = contract.relative_quaternion_wxyz if grip_quat is None else grip_quat
    collision_bit = contract.racket_collision_bit if collision_bit is None else collision_bit
    xml_path = Path(racket_xml_path) if racket_xml_path is not None else contract.asset_path
    if not xml_path.is_file():
        raise FileNotFoundError(f"racket asset not found: {xml_path}")
    if mass_scale <= 0.0:
        raise ValueError(f"mass_scale must be > 0, got {mass_scale}")
    if collision_bit <= 0:
        raise ValueError(f"collision_bit must be > 0, got {collision_bit}")

    hand = spec.body(attach_body)
    if hand is None:
        raise ValueError(f"attach body {attach_body!r} not present in MyoFullBody spec")

    racket_spec = mujoco.MjSpec.from_file(str(xml_path))
    if xml_path.resolve() == contract.asset_path:
        validate_racket_spec_against_contract(racket_spec, contract)

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
        "Attached rigid racket to %r (collision bit %d, prefix %r, contract %s)",
        attach_body,
        collision_bit,
        RACKET_ATTACH_PREFIX,
        contract.fingerprint,
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
        racket_attachment_contract: str | Path | RacketAttachmentContract | None,
        racket_grip_preset: str | Path | None,
        racket_attach_body: str | None,
        racket_grip_pos,
        racket_grip_quat,
        racket_collision_bit: int | None,
        racket_mass_scale: float,
        racket_xml_path,
    ) -> None:
        if racket_grip_preset is None and racket_attachment_contract is None:
            racket_grip_preset = DEFAULT_RACKET_GRIP_PRESET_PATH
        preset = (
            None
            if racket_grip_preset is None
            else load_racket_grip_preset(racket_grip_preset)
        )
        if preset is not None and racket_attachment_contract is None:
            racket_attachment_contract = preset.attachment_contract_path

        if isinstance(racket_attachment_contract, RacketAttachmentContract):
            contract = racket_attachment_contract
            contract.verify_asset()
        else:
            contract = load_racket_attachment_contract(racket_attachment_contract)
        if (
            preset is not None
            and preset.attachment_contract_fingerprint != contract.fingerprint
        ):
            raise ValueError(
                "racket_grip_preset is bound to attachment contract "
                f"{preset.attachment_contract_fingerprint}, but the environment uses "
                f"{contract.fingerprint}"
            )

        attach_body = contract.parent_body if racket_attach_body is None else racket_attach_body
        grip_pos = (
            contract.relative_position_m if racket_grip_pos is None else tuple(racket_grip_pos)
        )
        grip_quat = (
            contract.relative_quaternion_wxyz
            if racket_grip_quat is None
            else tuple(racket_grip_quat)
        )
        collision_bit = (
            contract.racket_collision_bit
            if racket_collision_bit is None
            else int(racket_collision_bit)
        )
        xml_path = contract.asset_path if racket_xml_path is None else Path(racket_xml_path)
        resolved_xml_path = xml_path.resolve()
        uses_canonical_contract = (
            bool(enable_racket)
            and attach_body == contract.parent_body
            and tuple(float(value) for value in grip_pos) == contract.relative_position_m
            and tuple(float(value) for value in grip_quat)
            == contract.relative_quaternion_wxyz
            and collision_bit == contract.racket_collision_bit
            and float(racket_mass_scale) == 1.0
            and resolved_xml_path == contract.asset_path
        )

        asset_sha256 = None
        if resolved_xml_path.is_file():
            asset_sha256 = hashlib.sha256(resolved_xml_path.read_bytes()).hexdigest()
        effective_payload = {
            "schema": "musclemimic.racket_attachment.effective.v1",
            "source_contract_fingerprint": contract.fingerprint,
            "enabled": bool(enable_racket),
            "attach_body": attach_body,
            "relative_position_m": [float(value) for value in grip_pos],
            "relative_quaternion_wxyz": [float(value) for value in grip_quat],
            "collision_bit": collision_bit,
            "mass_scale": float(racket_mass_scale),
            "racket_asset_path": str(resolved_xml_path),
            "racket_asset_sha256": asset_sha256,
        }
        self._enable_racket = enable_racket
        self._racket_grip_reference_json = (
            DEFAULT_GRIP_REFERENCE_JSON if preset is None else preset.source_path
        )
        self._racket_grip_preset_fingerprint = (
            None if preset is None else preset.fingerprint
        )
        self._racket_attachment_contract = contract
        self._racket_attachment_contract_fingerprint = (
            contract.fingerprint if uses_canonical_contract else None
        )
        self._racket_attachment_effective_fingerprint = (
            contract.fingerprint
            if uses_canonical_contract
            else canonical_json_fingerprint(effective_payload)
        )
        self._racket_attachment_uses_canonical_contract = uses_canonical_contract
        self._racket_attach_body = attach_body
        self._racket_grip_pos = grip_pos
        self._racket_grip_quat = grip_quat
        self._racket_collision_bit = collision_bit
        self._racket_mass_scale = racket_mass_scale
        self._racket_xml_path = resolved_xml_path

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        spec = super()._apply_spec_changes(spec)
        if getattr(self, "_enable_racket", True):
            spec = inject_racket(
                spec,
                attachment_contract=self._racket_attachment_contract,
                attach_body=self._racket_attach_body,
                grip_pos=self._racket_grip_pos,
                grip_quat=self._racket_grip_quat,
                collision_bit=self._racket_collision_bit,
                mass_scale=self._racket_mass_scale,
                racket_xml_path=self._racket_xml_path,
            )
        return spec

    @property
    def racket_attachment_contract_fingerprint(self) -> str | None:
        """Canonical contract fingerprint, or ``None`` when knobs override it."""

        return self._racket_attachment_contract_fingerprint

    @property
    def racket_attachment_effective_fingerprint(self) -> str:
        """Fingerprint of the actual attachment configuration in this model."""

        return self._racket_attachment_effective_fingerprint

    @property
    def racket_attachment_uses_canonical_contract(self) -> bool:
        return self._racket_attachment_uses_canonical_contract

    @property
    def racket_grip_preset_fingerprint(self) -> str | None:
        """Fingerprint of the global grip preset, or ``None`` for the legacy default."""

        return self._racket_grip_preset_fingerprint

    def _resolve_grip_fingers(self) -> None:
        """Cache the right-hand finger qpos addresses + grip targets for reset and
        reward. Empty when the fingers are disabled."""
        names, addrs, targets = grip_finger_reference(
            self._model,
            self._racket_grip_reference_json,
        )
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
        racket_attachment_contract: str | Path | RacketAttachmentContract | None = None,
        racket_grip_preset: str | Path | None = None,
        racket_attach_body: str | None = None,
        racket_grip_pos=None,
        racket_grip_quat=None,
        racket_collision_bit: int | None = None,
        racket_mass_scale: float = 1.0,
        racket_xml_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self._store_racket_params(
            enable_racket=enable_racket,
            racket_attachment_contract=racket_attachment_contract,
            racket_grip_preset=racket_grip_preset,
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
        racket_attachment_contract: str | Path | RacketAttachmentContract | None = None,
        racket_grip_preset: str | Path | None = None,
        racket_attach_body: str | None = None,
        racket_grip_pos=None,
        racket_grip_quat=None,
        racket_collision_bit: int | None = None,
        racket_mass_scale: float = 1.0,
        racket_xml_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self._store_racket_params(
            enable_racket=enable_racket,
            racket_attachment_contract=racket_attachment_contract,
            racket_grip_preset=racket_grip_preset,
            racket_attach_body=racket_attach_body,
            racket_grip_pos=racket_grip_pos,
            racket_grip_quat=racket_grip_quat,
            racket_collision_bit=racket_collision_bit,
            racket_mass_scale=racket_mass_scale,
            racket_xml_path=racket_xml_path,
        )
        super().__init__(timestep=timestep, n_substeps=n_substeps, mjx_backend=mjx_backend, **kwargs)
