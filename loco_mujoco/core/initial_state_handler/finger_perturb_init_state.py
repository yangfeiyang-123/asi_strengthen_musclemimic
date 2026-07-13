from types import ModuleType
from typing import Any, Tuple, Union

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.initial_state_handler.traj_init_state import TrajInitialStateHandler
from loco_mujoco.core.utils import assert_backend_is_supported


# Joint-name prefixes of the MyoFullBody hand (thumb CMC/MP/IP + fingers 2-5
# MCP/PM/MD flexion & abduction). Matching by prefix keeps this robust to qpos
# layout and to left/right suffixes.
_FINGER_JOINT_PREFIXES = (
    "cmc_",
    "mp_",
    "ip_",
    "mcp2",
    "mcp3",
    "mcp4",
    "mcp5",
    "pm2",
    "pm3",
    "pm4",
    "pm5",
    "md2",
    "md3",
    "md4",
    "md5",
)


def _normalize_finger_side(side: str) -> str:
    side = str(side).lower()
    aliases = {"r": "right", "l": "left"}
    side = aliases.get(side, side)
    if side not in {"right", "left", "both"}:
        raise ValueError(
            "finger_perturb_side must be one of 'right', 'left', or 'both', "
            f"got {side!r}"
        )
    return side


def _joint_matches_side(name: str, side: str) -> bool:
    if side == "both":
        return name.endswith("_r") or name.endswith("_l")
    suffix = "_r" if side == "right" else "_l"
    return name.endswith(suffix)


def _finger_state_addrs_and_ranges(model, prefixes, side="both"):
    """Return finger qpos/qvel addresses plus qpos limits for one or both hands."""
    side = _normalize_finger_side(side)
    qpos_addrs, qvel_addrs, low, high = [], [], [], []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name or not name.startswith(prefixes) or not _joint_matches_side(name, side):
            continue
        joint_type = int(model.jnt_type[jid])
        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            raise ValueError(
                f"finger perturb expects one-DoF hinge/slide joints, got {name!r} "
                f"with type={joint_type}"
            )
        qpos_addrs.append(int(model.jnt_qposadr[jid]))
        qvel_addrs.append(int(model.jnt_dofadr[jid]))
        if int(model.jnt_limited[jid]):
            low.append(float(model.jnt_range[jid, 0]))
            high.append(float(model.jnt_range[jid, 1]))
        else:
            low.append(-np.inf)
            high.append(np.inf)
    return (
        np.asarray(qpos_addrs, dtype=int),
        np.asarray(qvel_addrs, dtype=int),
        np.asarray(low, dtype=float),
        np.asarray(high, dtype=float),
    )


def _finger_qpos_addrs_and_ranges(model, prefixes):
    """Return (qpos_addrs, low, high) for all present finger joints.

    Empty arrays when the fingers are disabled (joints absent from the model),
    so the handler degrades to a plain trajectory reset with zero side effects.
    """
    qpos_addrs, _qvel_addrs, low, high = _finger_state_addrs_and_ranges(
        model, prefixes, side="both"
    )
    return qpos_addrs, low, high


class FingerPerturbInitialStateHandler(TrajInitialStateHandler):
    """Trajectory init handler that adds bounded random noise to the finger qpos.

    Stage-1R uses ``disable_fingers=false`` so the hand joints exist, while a
    separate ``BodyFingerIsolationWrapper`` keeps them out of the body policy
    interface. The retargeted trajectories carry no finger data, so the base
    :class:`TrajInitialStateHandler` zero-fills every finger joint at reset. This
    handler then applies uniform, joint-limit-clipped nuisance perturbations. The
    upper body / arms / legs remain exactly on the trajectory reset state.

    The perturbation is jit/vmap-safe: the jax branch draws from ``carry.key`` and
    threads the split key back, mirroring the goal-sampling reset pattern.

    Existing configs that only pass ``finger_perturb_scale`` retain the original
    behavior: both hands receive qpos noise and qvel is untouched.  New configs
    can select only the right hand and configure qpos/qvel independently.

    Args:
        finger_perturb_scale (float): Legacy qpos half-width. Used when
            ``finger_qpos_perturb_scale`` is omitted. Default ``0.15``.
        finger_perturb_side (str): ``"right"``, ``"left"``, or ``"both"``.
            Default ``"both"`` for backward compatibility.
        finger_qpos_perturb_scale (float | None): Explicit qpos half-width in
            radians. Overrides ``finger_perturb_scale`` when supplied.
        finger_qvel_perturb_scale (float): qvel half-width in rad/s. Default 0.
        finger_perturb_seed (int | None): Optional deterministic seed for the
            NumPy backend. JAX continues to use and advance ``carry.key``.
    """

    def __init__(
        self,
        env: Any,
        finger_perturb_scale: float = 0.15,
        finger_perturb_side: str = "both",
        finger_qpos_perturb_scale: float | None = None,
        finger_qvel_perturb_scale: float = 0.0,
        finger_perturb_seed: int | None = None,
        finger_perturb_rng_mode: str = "legacy_split",
        finger_perturb_stream_id: int = 17031,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self._finger_perturb_side = _normalize_finger_side(finger_perturb_side)
        self._finger_qpos_perturb_scale = float(
            finger_perturb_scale
            if finger_qpos_perturb_scale is None
            else finger_qpos_perturb_scale
        )
        self._finger_qvel_perturb_scale = float(finger_qvel_perturb_scale)
        self._finger_perturb_rng_mode = str(finger_perturb_rng_mode).lower()
        if self._finger_perturb_rng_mode not in {"legacy_split", "fold_in"}:
            raise ValueError(
                "finger_perturb_rng_mode must be 'legacy_split' or 'fold_in', "
                f"got {finger_perturb_rng_mode!r}"
            )
        self._finger_perturb_stream_id = int(finger_perturb_stream_id)
        for name, scale in (
            ("finger_qpos_perturb_scale", self._finger_qpos_perturb_scale),
            ("finger_qvel_perturb_scale", self._finger_qvel_perturb_scale),
        ):
            if not np.isfinite(scale) or scale < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value, got {scale}")
        # Retain this attribute for code that inspected the old handler.
        self._finger_perturb_scale = self._finger_qpos_perturb_scale
        qpos_addrs, qvel_addrs, low, high = _finger_state_addrs_and_ranges(
            env._model,
            _FINGER_JOINT_PREFIXES,
            side=self._finger_perturb_side,
        )
        self._finger_qpos_addrs = qpos_addrs
        self._finger_qvel_addrs = qvel_addrs
        self._finger_low = low
        self._finger_high = high
        self._np_rng = (
            None if finger_perturb_seed is None else np.random.default_rng(int(finger_perturb_seed))
        )

    def reset(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[MjData, Data], Any]:
        assert_backend_is_supported(backend)

        # First set the full trajectory state (fingers zero-filled by the base).
        data, carry = super().reset(env, model, data, carry, backend)

        qpos_addrs = self._finger_qpos_addrs
        qvel_addrs = self._finger_qvel_addrs
        qpos_scale = self._finger_qpos_perturb_scale
        qvel_scale = self._finger_qvel_perturb_scale
        if qpos_addrs.size == 0 or (qpos_scale <= 0.0 and qvel_scale <= 0.0):
            return data, carry

        if backend == np:
            random_uniform = np.random.uniform if self._np_rng is None else self._np_rng.uniform
            if qpos_scale > 0.0:
                noise = random_uniform(-qpos_scale, qpos_scale, size=qpos_addrs.shape)
                base = data.qpos[qpos_addrs]
                perturbed = np.clip(base + noise, self._finger_low, self._finger_high)
                data.qpos[qpos_addrs] = perturbed
            if qvel_scale > 0.0:
                noise = random_uniform(-qvel_scale, qvel_scale, size=qvel_addrs.shape)
                data.qvel[qvel_addrs] = data.qvel[qvel_addrs] + noise
            return data, carry

        import jax
        key = carry.key
        if self._finger_perturb_rng_mode == "fold_in":
            # Nuisance perturbation owns a deterministic independent stream and
            # must not advance the main reset key. Clean and perturbed rollouts
            # can therefore share all subsequent domain randomization exactly.
            stream_key = jax.random.fold_in(key, self._finger_perturb_stream_id)
            qpos_key = jax.random.fold_in(stream_key, 0)
            qvel_key = jax.random.fold_in(stream_key, 1)
        if qpos_scale > 0.0:
            if self._finger_perturb_rng_mode == "legacy_split":
                key, subkey = jax.random.split(key)
            else:
                subkey = qpos_key
            qpos_addrs_b = backend.asarray(qpos_addrs)
            noise = jax.random.uniform(
                subkey,
                shape=qpos_addrs.shape,
                minval=-qpos_scale,
                maxval=qpos_scale,
            )
            base = data.qpos[qpos_addrs_b]
            perturbed = backend.clip(
                base + noise,
                backend.asarray(self._finger_low),
                backend.asarray(self._finger_high),
            )
            data = data.replace(qpos=data.qpos.at[qpos_addrs_b].set(perturbed))
        if qvel_scale > 0.0:
            if self._finger_perturb_rng_mode == "legacy_split":
                key, subkey = jax.random.split(key)
            else:
                subkey = qvel_key
            qvel_addrs_b = backend.asarray(qvel_addrs)
            noise = jax.random.uniform(
                subkey,
                shape=qvel_addrs.shape,
                minval=-qvel_scale,
                maxval=qvel_scale,
            )
            data = data.replace(
                qvel=data.qvel.at[qvel_addrs_b].set(data.qvel[qvel_addrs_b] + noise)
            )
        if self._finger_perturb_rng_mode == "fold_in":
            return data, carry
        return data, carry.replace(key=key)
