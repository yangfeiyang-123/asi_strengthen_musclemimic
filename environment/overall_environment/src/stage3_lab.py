"""Stage-3 latent-action-barrier control and actuator ownership.

The task policy owns only the raw latent vector.  A frozen latent runtime
maps the current non-finger body state through the conditional prior and
decoder.  The production rigid-tool fixture is fingerless, so its final
354-D action is exactly the decoded body action.  A name-routed 416-D legacy
fixture remains readable for old experiments, but it is never selected
implicitly.

This module contains no PPO code and deliberately has no dependency on the
incoming-shuttle environments.  CPU evaluation and batched MJX training can
therefore share exactly the same control contract.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.distill.physical import (
    resolve_muscle_channel_contract,
    validate_unit_muscle_ctrlrange,
)


class LatentRuntimeProtocol(Protocol):
    """Minimal frozen runtime interface consumed by Stage 3."""

    state_dim: int
    latent_dim: int
    action_dim: int

    def prior_raw_numpy(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def decoder_numpy(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray: ...

    def prior_raw_jax(self, state: jax.Array) -> tuple[jax.Array, jax.Array]: ...

    def decoder_jax(self, state: jax.Array, latent: jax.Array) -> jax.Array: ...


class GripProviderProtocol(Protocol):
    action_size: int

    def action_numpy(self, lab_state: np.ndarray) -> np.ndarray: ...

    def action_jax(self, lab_state: jax.Array) -> jax.Array: ...


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_teacher_body_ctrlrange(model: Any, controller: "Stage3LABController") -> str | None:
    """Install the Stage-2 teacher control ranges into the full Stage-3 model.

    Decoder actions are normalized in the teacher's actuator ranges.  Applying
    them using whichever ranges happen to be present in the hitting scene can
    silently change all 354 body controls.  Runtime checkpoints therefore
    carry the exact name-aligned teacher ranges; this function applies them
    before either CPU scaling or ``mjx.put_model``.
    """
    value = getattr(controller.runtime, "body_ctrlrange", None)
    if value is None:
        return None
    ranges = np.asarray(value, dtype=float)
    expected = (controller.router.body_size, 2)
    if ranges.shape != expected:
        raise ValueError(f"teacher body_ctrlrange must have shape {expected}, got {ranges.shape}")
    validate_unit_muscle_ctrlrange(
        controller.router.body_actuator_names,
        ranges,
    )
    model.actuator_ctrlrange[controller.router.body_indices] = ranges
    model.actuator_ctrllimited[controller.router.body_indices] = True
    return effective_ctrlrange_hash(model, controller.router.all_actuator_names)


def effective_ctrlrange_hash(model: Any, actuator_names: Sequence[str]) -> str:
    """Hash full ordered control bounds and limited flags after runtime patching."""
    ranges = np.asarray(model.actuator_ctrlrange, dtype=float)
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    if ranges.shape != (len(actuator_names), 2) or limited.shape != (len(actuator_names),):
        raise ValueError("model actuator control arrays do not match the named action schema")
    return _stable_hash(
        {
            "schema_version": "effective_ctrlrange_v1",
            "actuator_names": list(actuator_names),
            "ctrlrange": ranges.tolist(),
            "ctrllimited": limited.astype(int).tolist(),
        }
    )


def stage3_attachment_report(
    model: Any,
    scene_xml: str | Path,
    *,
    contract_path: str | Path | None = None,
    human_root_body_name: str = "Full Body",
    racket_body_name: str = "overall_racket",
    shuttle_body_name: str = "overall_shuttle",
) -> dict[str, Any]:
    """Validate the production jointless-racket topology against its contract.

    The racket lives inside the human kinematic tree, so the human collision
    set explicitly excludes the racket subtree.  The report checks compiled
    masks *and* explicit pairs: an XML ``exclude`` is useful provenance but is
    not accepted as a substitute for an actually collision-incompatible model.
    Racket--shuttle collision is checked separately and must remain enabled.
    """
    import mujoco

    from environment.overall_environment.src.racket_attachment import (
        DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH,
        load_racket_attachment_contract,
    )

    scene_path = Path(scene_xml)
    root = ET.parse(scene_path).getroot()
    custom_text = {
        node.attrib.get("name", ""): node.attrib.get("data", "")
        for node in root.findall("./custom/text")
    }
    repository_root = Path(__file__).resolve().parents[3]
    embedded_contract_path = custom_text.get(
        "overall_racket_attachment_contract_path"
    )
    selected_contract_path: str | Path = (
        DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH
        if contract_path is None and not embedded_contract_path
        else (embedded_contract_path if contract_path is None else contract_path)
    )
    selected_contract_path = Path(selected_contract_path).expanduser()
    if not selected_contract_path.is_absolute():
        selected_contract_path = repository_root / selected_contract_path
    contract = load_racket_attachment_contract(selected_contract_path)

    def named_id(obj_type: Any, name: str) -> int:
        index = int(mujoco.mj_name2id(model, obj_type, name))
        if index < 0:
            raise ValueError(f"Stage-3 attachment model is missing {name!r}")
        return index

    human_root_id = named_id(mujoco.mjtObj.mjOBJ_BODY, human_root_body_name)
    racket_body_id = named_id(mujoco.mjtObj.mjOBJ_BODY, racket_body_name)
    parent_body_id = named_id(mujoco.mjtObj.mjOBJ_BODY, contract.parent_body)
    shuttle_body_id = named_id(mujoco.mjtObj.mjOBJ_BODY, shuttle_body_name)

    def is_descendant(body_id: int, root_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == root_id:
                return True
            current = int(model.body_parentid[current])
        return current == root_id

    racket_bodies = {
        body_id for body_id in range(int(model.nbody)) if is_descendant(body_id, racket_body_id)
    }
    human_bodies = {
        body_id
        for body_id in range(int(model.nbody))
        if is_descendant(body_id, human_root_id) and body_id not in racket_bodies
    }
    shuttle_bodies = {
        body_id for body_id in range(int(model.nbody)) if is_descendant(body_id, shuttle_body_id)
    }
    human_geoms = {
        index
        for index, body_id in enumerate(np.asarray(model.geom_bodyid))
        if int(body_id) in human_bodies
    }
    racket_geoms = {
        index
        for index, body_id in enumerate(np.asarray(model.geom_bodyid))
        if int(body_id) in racket_bodies
    }
    shuttle_geoms = {
        index
        for index, body_id in enumerate(np.asarray(model.geom_bodyid))
        if int(body_id) in shuttle_bodies
    }
    proxy_geom_name = f"overall_{contract.stringbed_proxy_geom_name}"
    proxy_geom_id = named_id(mujoco.mjtObj.mjOBJ_GEOM, proxy_geom_name)
    if proxy_geom_id not in racket_geoms:
        raise ValueError(
            f"Stage-3 stringbed proxy {proxy_geom_name!r} is outside the racket subtree"
        )
    proxy_geoms = {proxy_geom_id}
    frame_geoms = racket_geoms - proxy_geoms

    def mask_pair_count(first: set[int], second: set[int]) -> int:
        count = 0
        for first_geom in first:
            for second_geom in second:
                first_type = int(model.geom_contype[first_geom])
                first_affinity = int(model.geom_conaffinity[first_geom])
                second_type = int(model.geom_contype[second_geom])
                second_affinity = int(model.geom_conaffinity[second_geom])
                if (first_type & second_affinity) or (second_type & first_affinity):
                    count += 1
        return count

    def explicit_pair_count(first: set[int], second: set[int]) -> int:
        count = 0
        for pair_id in range(int(model.npair)):
            geom1 = int(model.pair_geom1[pair_id])
            geom2 = int(model.pair_geom2[pair_id])
            if (geom1 in first and geom2 in second) or (geom2 in first and geom1 in second):
                count += 1
        return count

    human_racket_mask_pairs = mask_pair_count(human_geoms, racket_geoms)
    human_racket_explicit_pairs = explicit_pair_count(human_geoms, racket_geoms)
    racket_shuttle_mask_pairs = mask_pair_count(racket_geoms, shuttle_geoms)
    racket_shuttle_explicit_pairs = explicit_pair_count(racket_geoms, shuttle_geoms)
    proxy_shuttle_mask_pairs = mask_pair_count(proxy_geoms, shuttle_geoms)
    proxy_shuttle_explicit_pairs = explicit_pair_count(proxy_geoms, shuttle_geoms)
    frame_shuttle_mask_pairs = mask_pair_count(frame_geoms, shuttle_geoms)
    frame_shuttle_explicit_pairs = explicit_pair_count(frame_geoms, shuttle_geoms)

    ground_geoms: set[int] = set()
    for ground_name in ("floor", "overall_floor_collision"):
        ground_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_name)
        )
        if ground_id >= 0:
            ground_geoms.add(ground_id)
    proxy_ground_mask_pairs = mask_pair_count(proxy_geoms, ground_geoms)
    proxy_ground_explicit_pairs = explicit_pair_count(proxy_geoms, ground_geoms)

    racket_equality_constraints = 0
    body_equality_types = {
        int(mujoco.mjtEq.mjEQ_CONNECT),
        int(mujoco.mjtEq.mjEQ_WELD),
    }
    for equality_id in range(int(model.neq)):
        if int(model.eq_type[equality_id]) not in body_equality_types:
            continue
        obj1 = int(model.eq_obj1id[equality_id])
        obj2 = int(model.eq_obj2id[equality_id])
        if obj1 in racket_bodies or obj2 in racket_bodies:
            racket_equality_constraints += 1

    racket_joint_count = sum(
        int(model.body_jntnum[body_id]) for body_id in racket_bodies
    )

    stringbed_name = f"overall_{contract.stringbed_site_name}"
    stringbed_id = named_id(mujoco.mjtObj.mjOBJ_SITE, stringbed_name)
    racket_position = np.asarray(model.body_pos[racket_body_id], dtype=float)
    racket_quaternion = np.asarray(model.body_quat[racket_body_id], dtype=float)
    expected_position = np.asarray(contract.relative_position_m, dtype=float)
    expected_quaternion = np.asarray(contract.relative_quaternion_wxyz, dtype=float)

    def unit_quaternion(value: np.ndarray, *, field: str) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError(f"{field} must be a finite nonzero quaternion")
        return value / norm

    quaternion_dot = float(
        np.clip(
            abs(
                np.dot(
                    unit_quaternion(racket_quaternion, field="model racket quaternion"),
                    unit_quaternion(expected_quaternion, field="contract racket quaternion"),
                )
            ),
            0.0,
            1.0,
        )
    )
    rotation_error_rad = float(2.0 * np.arccos(quaternion_dot))
    position_error_m = float(np.linalg.norm(racket_position - expected_position))

    body_mass_error_kg = abs(float(model.body_mass[racket_body_id]) - contract.racket_mass_kg)
    body_com_error_m = float(
        np.linalg.norm(
            np.asarray(model.body_ipos[racket_body_id], dtype=float)
            - np.asarray(contract.racket_center_of_mass_m, dtype=float)
        )
    )
    body_inertia_error = float(
        np.max(
            np.abs(
                np.asarray(model.body_inertia[racket_body_id], dtype=float)
                - np.asarray(contract.racket_diagonal_inertia_kg_m2, dtype=float)
            )
        )
    )
    stringbed_position_error_m = float(
        np.linalg.norm(
            np.asarray(model.site_pos[stringbed_id], dtype=float)
            - np.asarray(contract.stringbed_position_m, dtype=float)
        )
    )
    stringbed_quaternion = np.asarray(model.site_quat[stringbed_id], dtype=float)
    expected_stringbed_quaternion = np.asarray(
        contract.stringbed_quaternion_wxyz, dtype=float
    )
    stringbed_quaternion_dot = float(
        np.clip(
            abs(
                np.dot(
                    unit_quaternion(
                        stringbed_quaternion,
                        field="model stringbed quaternion",
                    ),
                    unit_quaternion(
                        expected_stringbed_quaternion,
                        field="contract stringbed quaternion",
                    ),
                )
            ),
            0.0,
            1.0,
        )
    )
    stringbed_rotation_error_rad = float(2.0 * np.arccos(stringbed_quaternion_dot))

    contact_exclude_present = any(
        {node.attrib.get("body1"), node.attrib.get("body2")}
        == {human_root_body_name, racket_body_name}
        for node in root.findall(".//contact/exclude")
    )
    hand_racket_contact_enabled = bool(
        human_racket_mask_pairs or human_racket_explicit_pairs
    )
    racket_shuttle_contact_enabled = bool(
        racket_shuttle_mask_pairs or racket_shuttle_explicit_pairs
    )
    native_proxy_shuttle_contact_enabled = bool(
        proxy_shuttle_mask_pairs or proxy_shuttle_explicit_pairs
    )
    native_frame_shuttle_contact_enabled = bool(
        frame_shuttle_mask_pairs or frame_shuttle_explicit_pairs
    )
    proxy_ground_contact_enabled = bool(
        proxy_ground_mask_pairs or proxy_ground_explicit_pairs
    )
    try:
        contract_path_for_report = str(contract.source_path.relative_to(repository_root))
    except ValueError:
        contract_path_for_report = str(contract.source_path)
    report = {
        "schema_version": "stage3_exact_child_attachment_v2",
        "attachment_mode": contract.attachment_mode,
        "contract_id": contract.contract_id,
        "contract_fingerprint": contract.fingerprint,
        "contract_path": contract_path_for_report,
        "embedded_contract_metadata": custom_text,
        "embedded_contract_path": embedded_contract_path,
        "parent_body_name": contract.parent_body,
        "parent_body_matches": int(model.body_parentid[racket_body_id]) == parent_body_id,
        "racket_joint_count": int(racket_joint_count),
        "racket_equality_constraint_count": int(racket_equality_constraints),
        "relative_position_m": racket_position.tolist(),
        "relative_quaternion_wxyz": racket_quaternion.tolist(),
        "relative_position_error_m": position_error_m,
        "relative_rotation_error_rad": rotation_error_rad,
        "racket_mass_kg": float(model.body_mass[racket_body_id]),
        "racket_center_of_mass_m": np.asarray(
            model.body_ipos[racket_body_id], dtype=float
        ).tolist(),
        "racket_diagonal_inertia_kg_m2": np.asarray(
            model.body_inertia[racket_body_id], dtype=float
        ).tolist(),
        "racket_mass_error_kg": body_mass_error_kg,
        "racket_center_of_mass_error_m": body_com_error_m,
        "racket_inertia_max_abs_error_kg_m2": body_inertia_error,
        "stringbed_site_name": stringbed_name,
        "stringbed_position_error_m": stringbed_position_error_m,
        "stringbed_rotation_error_rad": stringbed_rotation_error_rad,
        "human_root_body_name": human_root_body_name,
        "racket_body_name": racket_body_name,
        "contact_exclude_present": bool(contact_exclude_present),
        "human_racket_mask_compatible_geom_pairs": int(human_racket_mask_pairs),
        "human_racket_explicit_contact_pairs": int(human_racket_explicit_pairs),
        "hand_racket_contact_enabled": hand_racket_contact_enabled,
        "racket_shuttle_mask_compatible_geom_pairs": int(racket_shuttle_mask_pairs),
        "racket_shuttle_explicit_contact_pairs": int(racket_shuttle_explicit_pairs),
        "racket_shuttle_contact_enabled": racket_shuttle_contact_enabled,
        "stringbed_contact_model": contract.stringbed_contact_model,
        "stringbed_proxy_geom_name": proxy_geom_name,
        "stringbed_proxy_shuttle_mask_compatible_geom_pairs": int(
            proxy_shuttle_mask_pairs
        ),
        "stringbed_proxy_shuttle_explicit_contact_pairs": int(
            proxy_shuttle_explicit_pairs
        ),
        "native_stringbed_proxy_shuttle_contact_enabled": native_proxy_shuttle_contact_enabled,
        "racket_frame_shuttle_mask_compatible_geom_pairs": int(frame_shuttle_mask_pairs),
        "racket_frame_shuttle_explicit_contact_pairs": int(
            frame_shuttle_explicit_pairs
        ),
        "native_racket_frame_shuttle_contact_enabled": native_frame_shuttle_contact_enabled,
        "stringbed_proxy_ground_contact_enabled": proxy_ground_contact_enabled,
    }
    tolerances = {
        "relative_position_error_m": 1.0e-8,
        "relative_rotation_error_rad": 1.0e-6,
        "racket_mass_error_kg": 1.0e-10,
        "racket_center_of_mass_error_m": 1.0e-8,
        "racket_inertia_max_abs_error_kg_m2": 1.0e-10,
        "stringbed_position_error_m": 1.0e-8,
        "stringbed_rotation_error_rad": 1.0e-6,
    }
    checks = {
        "direct_parent": bool(report["parent_body_matches"]),
        "jointless_racket_subtree": int(report["racket_joint_count"]) == 0,
        "no_racket_equality_constraint": int(report["racket_equality_constraint_count"]) == 0,
        "relative_position": position_error_m <= tolerances["relative_position_error_m"],
        "relative_rotation": rotation_error_rad <= tolerances["relative_rotation_error_rad"],
        "racket_mass": body_mass_error_kg <= tolerances["racket_mass_error_kg"],
        "racket_center_of_mass": body_com_error_m
        <= tolerances["racket_center_of_mass_error_m"],
        "racket_inertia": body_inertia_error
        <= tolerances["racket_inertia_max_abs_error_kg_m2"],
        "stringbed_position": stringbed_position_error_m
        <= tolerances["stringbed_position_error_m"],
        "stringbed_rotation": stringbed_rotation_error_rad
        <= tolerances["stringbed_rotation_error_rad"],
        "no_human_racket_contact": not hand_racket_contact_enabled,
        "racket_shuttle_contact_preserved": racket_shuttle_contact_enabled,
        "single_custom_stringbed_model": (
            custom_text.get("overall_stringbed_contact_model")
            == contract.stringbed_contact_model
        ),
        "no_native_stringbed_proxy_shuttle_contact": (
            not native_proxy_shuttle_contact_enabled
            and custom_text.get("overall_native_stringbed_proxy_shuttle_contact")
            == "false"
        ),
        "native_racket_frame_shuttle_contact_preserved": (
            native_frame_shuttle_contact_enabled
            and custom_text.get("overall_native_racket_frame_shuttle_contact")
            == "true"
        ),
        "stringbed_proxy_ground_contact_preserved": proxy_ground_contact_enabled,
        "embedded_contract_schema": (
            custom_text.get("overall_racket_attachment_contract_schema")
            == contract.schema
        ),
        "embedded_contract_id": (
            custom_text.get("overall_racket_attachment_contract_id")
            == contract.contract_id
        ),
        "embedded_contract_fingerprint": (
            custom_text.get("overall_racket_attachment_contract_fingerprint")
            == contract.fingerprint
        ),
        "embedded_contract_path": embedded_contract_path == contract_path_for_report,
        "embedded_attachment_mode": (
            custom_text.get("overall_racket_attachment_mode")
            == contract.attachment_mode
        ),
        "embedded_finger_mode": custom_text.get("overall_finger_mode") == "removed",
        "embedded_stringbed_proxy_geom": (
            custom_text.get("overall_stringbed_proxy_geom_name") == proxy_geom_name
        ),
    }
    report["contract_tolerances"] = tolerances
    report["contract_checks"] = checks
    report["contract_passed"] = all(checks.values())
    report["attachment_hash"] = _stable_hash(report)
    return report


def _validate_names(label: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    duplicates = sorted({name for name in result if result.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} contains duplicate actuator names: {duplicates}")
    return result


@dataclass(frozen=True)
class Stage3ActionRouter:
    """Exhaustive actuator ownership for rigid-tool and legacy fixtures.

    Production uses ``expected_sizes=(354, 0, 0)``.  The historical
    full-finger scene uses ``(354, 31, 31)`` and must be requested by loading
    that topology explicitly; partially present hands are always rejected.
    """

    all_actuator_names: Sequence[str]
    body_actuator_names: Sequence[str]
    right_grip_actuator_names: Sequence[str]
    left_neutral_actuator_names: Sequence[str]
    expected_sizes: tuple[int, int, int] = (354, 31, 31)

    def __post_init__(self) -> None:
        all_names = _validate_names("all_actuator_names", self.all_actuator_names)
        body_names = _validate_names("body_actuator_names", self.body_actuator_names)
        right_names = _validate_names(
            "right_grip_actuator_names", self.right_grip_actuator_names
        )
        left_names = _validate_names(
            "left_neutral_actuator_names", self.left_neutral_actuator_names
        )
        owners = (set(body_names), set(right_names), set(left_names))
        overlap = sorted((owners[0] & owners[1]) | (owners[0] & owners[2]) | (owners[1] & owners[2]))
        if overlap:
            raise ValueError(f"actuators assigned to multiple Stage-3 owners: {overlap}")
        owned = owners[0] | owners[1] | owners[2]
        unknown = sorted(owned - set(all_names))
        unowned = sorted(set(all_names) - owned)
        if unknown:
            raise ValueError(f"owned actuator names absent from model: {unknown}")
        if unowned:
            raise ValueError(f"actuators have no Stage-3 owner: {unowned}")
        expected = tuple(int(value) for value in self.expected_sizes)
        actual = (len(body_names), len(right_names), len(left_names))
        if actual != expected:
            raise ValueError(
                "Stage-3 actuator partition size mismatch: "
                f"expected body/right/left={expected}, got {actual}"
            )
        if len(all_names) != sum(expected):
            raise ValueError(
                f"full actuator size must be {sum(expected)}, got {len(all_names)}"
            )

        index_by_name = {name: index for index, name in enumerate(all_names)}
        object.__setattr__(self, "all_actuator_names", all_names)
        object.__setattr__(self, "body_actuator_names", body_names)
        object.__setattr__(self, "right_grip_actuator_names", right_names)
        object.__setattr__(self, "left_neutral_actuator_names", left_names)
        object.__setattr__(self, "expected_sizes", expected)
        object.__setattr__(
            self,
            "body_indices",
            np.asarray([index_by_name[name] for name in body_names], dtype=np.int32),
        )
        object.__setattr__(
            self,
            "right_grip_indices",
            np.asarray([index_by_name[name] for name in right_names], dtype=np.int32),
        )
        object.__setattr__(
            self,
            "left_neutral_indices",
            np.asarray([index_by_name[name] for name in left_names], dtype=np.int32),
        )

    @classmethod
    def from_model(cls, model: Any) -> "Stage3ActionRouter":
        """Create a strict 354-D rigid or explicit 416-D legacy router."""
        import mujoco

        from musclemimic.utils.finger_isolation import (
            LEFT_FINGER_ACTUATOR_NAMES,
            RIGHT_FINGER_ACTUATOR_NAMES,
            FingerActuatorPartition,
        )

        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        ]
        if any(name is None for name in names):
            raise ValueError("every Stage-3 actuator must have a name")
        right_set = frozenset(RIGHT_FINGER_ACTUATOR_NAMES)
        left_set = frozenset(LEFT_FINGER_ACTUATOR_NAMES)
        right_count = sum(name in right_set for name in names)
        left_count = sum(name in left_set for name in names)
        if (right_count, left_count) == (0, 0):
            expected_sizes = (354, 0, 0)
        elif (right_count, left_count) == (31, 31):
            expected_sizes = (354, 31, 31)
        else:
            raise ValueError(
                "Stage-3 requires either a completely fingerless model or the "
                f"explicit legacy full-finger model; got right/left={right_count}/{left_count}"
            )
        partition = FingerActuatorPartition.from_actuator_names(
            names,
            expected_sizes=expected_sizes,
        )
        return cls(
            all_actuator_names=partition.all_actuator_names,
            body_actuator_names=partition.body_actuator_names,
            right_grip_actuator_names=partition.right_grip_actuator_names,
            left_neutral_actuator_names=partition.left_neutral_actuator_names,
            expected_sizes=partition.expected_sizes,
        )

    @property
    def full_size(self) -> int:
        return len(self.all_actuator_names)

    @property
    def body_size(self) -> int:
        return len(self.body_actuator_names)

    @property
    def right_grip_size(self) -> int:
        return len(self.right_grip_actuator_names)

    @property
    def left_neutral_size(self) -> int:
        return len(self.left_neutral_actuator_names)

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "schema_version": "stage3_action_router_v2",
                "fixture_mode": self.fixture_mode,
                "all": self.all_actuator_names,
                "body": self.body_actuator_names,
                "right_grip": self.right_grip_actuator_names,
                "left_neutral": self.left_neutral_actuator_names,
            }
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "stage3_action_router_v2",
            "fixture_mode": self.fixture_mode,
            "all_actuator_names": list(self.all_actuator_names),
            "body_actuator_names": list(self.body_actuator_names),
            "right_grip_actuator_names": list(self.right_grip_actuator_names),
            "left_neutral_actuator_names": list(self.left_neutral_actuator_names),
            "partition_sizes": list(self.expected_sizes),
            "schema_hash": self.schema_hash,
        }

    @property
    def fixture_mode(self) -> str:
        if (self.right_grip_size, self.left_neutral_size) == (0, 0):
            return "rigid_tool_fingerless"
        return "legacy_free_fingers"

    def assert_runtime_mask(self, action_mask: Any) -> None:
        all_names = tuple(getattr(action_mask, "all_actuator_names"))
        body = tuple(getattr(action_mask, "body_actuator_names"))
        correction = tuple(getattr(action_mask, "correction_actuator_names"))
        neutral = tuple(getattr(action_mask, "neutral_actuator_names"))
        if body != self.body_actuator_names:
            raise ValueError("latent checkpoint body actuator partition does not match Stage-3 model")
        if self.fixture_mode == "rigid_tool_fingerless":
            # Latent datasets produced from the old full model may still carry
            # correction/neutral metadata.  Those channels were never decoded
            # by the 354-D body runtime and do not exist in the rigid fixture.
            # Bind the physical ABI to the body order and reject any body name
            # hidden in the hand partitions.
            if set(body) & (set(correction) | set(neutral)):
                raise ValueError("latent checkpoint hand partitions overlap the body schema")
            return
        if all_names != self.all_actuator_names:
            raise ValueError(
                "latent checkpoint full actuator order does not match the Stage-3 model"
            )
        if correction != self.right_grip_actuator_names:
            raise ValueError("latent checkpoint correction partition does not match right grip")
        if neutral != self.left_neutral_actuator_names:
            raise ValueError("latent checkpoint neutral partition does not match left hand")

    def merge_numpy(
        self,
        *,
        body_action: np.ndarray,
        right_grip_action: np.ndarray,
        left_neutral_action: np.ndarray,
    ) -> np.ndarray:
        body = _numpy_action("body_action", body_action, self.body_size)
        right = _numpy_action("right_grip_action", right_grip_action, self.right_grip_size)
        left = _numpy_action("left_neutral_action", left_neutral_action, self.left_neutral_size)
        batch_shape = np.broadcast_shapes(body.shape[:-1], right.shape[:-1], left.shape[:-1])
        dtype = np.result_type(body, right, left)
        full = np.zeros((*batch_shape, self.full_size), dtype=dtype)
        full[..., self.body_indices] = np.broadcast_to(body, (*batch_shape, self.body_size))
        full[..., self.right_grip_indices] = np.broadcast_to(
            right, (*batch_shape, self.right_grip_size)
        )
        full[..., self.left_neutral_indices] = np.broadcast_to(
            left, (*batch_shape, self.left_neutral_size)
        )
        return full

    def merge_jax(
        self,
        *,
        body_action: jax.Array,
        right_grip_action: jax.Array,
        left_neutral_action: jax.Array,
    ) -> jax.Array:
        body = _jax_action("body_action", body_action, self.body_size)
        right = _jax_action("right_grip_action", right_grip_action, self.right_grip_size)
        left = _jax_action("left_neutral_action", left_neutral_action, self.left_neutral_size)
        batch_shape = jnp.broadcast_shapes(body.shape[:-1], right.shape[:-1], left.shape[:-1])
        dtype = jnp.result_type(body, right, left)
        full = jnp.zeros((*batch_shape, self.full_size), dtype=dtype)
        full = full.at[..., jnp.asarray(self.body_indices)].set(
            jnp.broadcast_to(body, (*batch_shape, self.body_size))
        )
        full = full.at[..., jnp.asarray(self.right_grip_indices)].set(
            jnp.broadcast_to(right, (*batch_shape, self.right_grip_size))
        )
        return full.at[..., jnp.asarray(self.left_neutral_indices)].set(
            jnp.broadcast_to(left, (*batch_shape, self.left_neutral_size))
        )


DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES: tuple[str, ...] = (
    "SUP",
    "BRA",
    "BRD",
    "ECRL",
    "ECRB",
    "ECU",
    "FCR",
    "FCU",
    "PT",
    "PQ",
)


@dataclass(frozen=True)
class BoundedResidualMask:
    """Optional small right wrist/forearm correction inside the body channel.

    This is intentionally not a second full-muscle policy.  It owns an exact
    name list inside the decoder's 354-D body vector and is capped at 0.10.
    Finger actuator names are rejected even if supplied accidentally.
    """

    body_actuator_names: Sequence[str]
    residual_actuator_names: Sequence[str] = DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES
    alpha: float = 0.05

    def __post_init__(self) -> None:
        from musclemimic.utils.finger_isolation import finger_actuator_side

        body = _validate_names("body_actuator_names", self.body_actuator_names)
        residual = _validate_names(
            "residual_actuator_names", self.residual_actuator_names
        )
        missing = [name for name in residual if name not in body]
        if missing:
            raise ValueError(f"bounded residual names are absent from the body decoder: {missing}")
        fingers = [name for name in residual if finger_actuator_side(name) is not None]
        if fingers:
            raise ValueError(f"bounded residual must never own finger actuators: {fingers}")
        if not 0.0 <= float(self.alpha) <= 0.10:
            raise ValueError("bounded residual alpha must lie in [0, 0.10]")
        object.__setattr__(self, "body_actuator_names", body)
        object.__setattr__(self, "residual_actuator_names", residual)
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(
            self,
            "body_indices",
            np.asarray([body.index(name) for name in residual], dtype=np.int32),
        )

    @property
    def residual_size(self) -> int:
        return len(self.residual_actuator_names)

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "schema_version": "bounded_body_residual_v1",
                "body_actuator_names": self.body_actuator_names,
                "residual_actuator_names": self.residual_actuator_names,
                "alpha": self.alpha,
            }
        )

    def apply_numpy(self, body_action: Any, raw_residual: Any) -> np.ndarray:
        body = _numpy_action("body_action", body_action, len(self.body_actuator_names))
        residual = _numpy_action("raw bounded residual", raw_residual, self.residual_size)
        if body.shape[:-1] != residual.shape[:-1]:
            raise ValueError("body_action and bounded residual batch dimensions must match")
        result = np.array(body, copy=True)
        result[..., self.body_indices] += self.alpha * np.tanh(residual)
        return np.clip(result, -1.0, 1.0)

    def apply_jax(self, body_action: Any, raw_residual: Any) -> jax.Array:
        body = _jax_action("body_action", body_action, len(self.body_actuator_names))
        residual = _jax_action("raw bounded residual", raw_residual, self.residual_size)
        if body.shape[:-1] != residual.shape[:-1]:
            raise ValueError("body_action and bounded residual batch dimensions must match")
        indices = jnp.asarray(self.body_indices)
        corrected = body.at[..., indices].add(self.alpha * jnp.tanh(residual))
        return jnp.clip(corrected, -1.0, 1.0)


class ConstantGripProvider:
    """Deterministic right-hand grip, shared by NumPy and JAX execution."""

    def __init__(self, action: Sequence[float]) -> None:
        values = np.asarray(action, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("constant grip action must be a finite one-dimensional vector")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("constant grip action must lie in normalized range [-1, 1]")
        self._action = values.copy()

    @property
    def action_size(self) -> int:
        return int(self._action.size)

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "provider": "constant_grip_v1",
                "action": self._action.astype(float).tolist(),
            }
        )

    def action_numpy(self, lab_state: np.ndarray) -> np.ndarray:
        state = np.asarray(lab_state)
        return np.broadcast_to(self._action, (*state.shape[:-1], self.action_size)).copy()

    def action_jax(self, lab_state: jax.Array) -> jax.Array:
        state = jnp.asarray(lab_state)
        return jnp.broadcast_to(jnp.asarray(self._action), (*state.shape[:-1], self.action_size))


class FrozenGripProvider:
    """Adapter for an independently frozen grip policy."""

    def __init__(
        self,
        *,
        action_size: int,
        numpy_fn: Callable[[np.ndarray], np.ndarray],
        jax_fn: Callable[[jax.Array], jax.Array],
        schema_hash: str,
    ) -> None:
        if int(action_size) <= 0:
            raise ValueError("action_size must be positive")
        if not schema_hash:
            raise ValueError("frozen grip provider requires a schema hash")
        self.action_size = int(action_size)
        self._numpy_fn = numpy_fn
        self._jax_fn = jax_fn
        self.schema_hash = str(schema_hash)

    def action_numpy(self, lab_state: np.ndarray) -> np.ndarray:
        return _numpy_action("frozen right grip", self._numpy_fn(lab_state), self.action_size)

    def action_jax(self, lab_state: jax.Array) -> jax.Array:
        return _jax_action("frozen right grip", self._jax_fn(lab_state), self.action_size)


@dataclass(frozen=True)
class Stage3LABOutput:
    full_action: Any
    body_action: Any
    right_grip_action: Any
    left_neutral_action: Any
    latent: Any
    raw_latent: Any
    prior_mu: Any
    prior_sigma: Any
    lambda_lab: Any
    raw_bounded_residual: Any | None = None


class Stage3LABController:
    """Frozen prior/decoder plus the sole latent tanh barrier."""

    def __init__(
        self,
        *,
        runtime: LatentRuntimeProtocol,
        router: Stage3ActionRouter,
        right_grip_provider: GripProviderProtocol | None = None,
        lambda_lab: float = 0.25,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        left_neutral_value: float = 0.0,
        bounded_residual_mask: BoundedResidualMask | None = None,
    ) -> None:
        if float(lambda_lab) < 0.0:
            raise ValueError("lambda_lab must be non-negative")
        if float(sigma_min) <= 0.0 or float(sigma_max) < float(sigma_min):
            raise ValueError("LAB sigma bounds must satisfy 0 < sigma_min <= sigma_max")
        if int(runtime.action_dim) != router.body_size:
            raise ValueError(
                f"latent decoder action_dim={runtime.action_dim} != body partition {router.body_size}"
            )
        runtime_ctrlrange = getattr(runtime, "body_ctrlrange", None)
        if runtime_ctrlrange is not None:
            validate_unit_muscle_ctrlrange(
                router.body_actuator_names,
                runtime_ctrlrange,
            )
        for method_name in ("prior_raw_numpy", "prior_raw_jax"):
            if not callable(getattr(runtime, method_name, None)):
                raise ValueError(
                    "Stage-3 latent runtime ABI requires raw prior scale logits via "
                    f"{method_name}; an already transformed sigma would be softplus-applied twice"
                )
        if router.fixture_mode == "rigid_tool_fingerless":
            if right_grip_provider is not None:
                raise ValueError("fingerless rigid-tool Stage-3 must not install a hand provider")
        elif (
            right_grip_provider is None
            or int(right_grip_provider.action_size) != router.right_grip_size
        ):
            raise ValueError("legacy right grip provider size does not match the hand partition")
        if not np.isfinite(left_neutral_value):
            raise ValueError("left_neutral_value must be finite")
        action_mask = getattr(runtime, "action_mask", None)
        if action_mask is not None:
            router.assert_runtime_mask(action_mask)
        if (
            bounded_residual_mask is not None
            and tuple(bounded_residual_mask.body_actuator_names)
            != router.body_actuator_names
        ):
            raise ValueError("bounded residual body schema does not match the Stage-3 router")
        self.runtime = runtime
        self.router = router
        self.right_grip_provider = right_grip_provider
        self.lambda_lab = float(lambda_lab)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.left_neutral_value = float(left_neutral_value)
        self.bounded_residual_mask = bounded_residual_mask

    @property
    def task_action_size(self) -> int:
        return self.latent_action_size + self.residual_action_size

    @property
    def latent_action_size(self) -> int:
        return int(self.runtime.latent_dim)

    @property
    def residual_action_size(self) -> int:
        return (
            0
            if self.bounded_residual_mask is None
            else self.bounded_residual_mask.residual_size
        )

    @property
    def lab_state_size(self) -> int:
        return int(self.runtime.state_dim)

    @property
    def control_manifest(self) -> dict[str, Any]:
        runtime_hash = str(getattr(self.runtime, "schema_hash", "unknown"))
        runtime_control_manifest = getattr(self.runtime, "control_manifest", None)
        runtime_physical_schema = (
            runtime_control_manifest.get("physical_signal_schema_version")
            if isinstance(runtime_control_manifest, Mapping)
            else None
        )
        grip_hash = (
            None
            if self.right_grip_provider is None
            else str(getattr(self.right_grip_provider, "schema_hash", "unknown"))
        )
        payload = {
            "schema_version": "stage3_lab_control_v1",
            "runtime_schema_hash": runtime_hash,
            "runtime_checkpoint_fingerprint": getattr(
                self.runtime, "checkpoint_fingerprint", None
            ),
            "latent_checkpoint_fingerprint": getattr(
                self.runtime, "checkpoint_fingerprint", None
            ),
            "decoder_type": getattr(self.runtime, "decoder_type", None),
            "frozen_body_decoder_fingerprint": getattr(
                getattr(self.runtime, "frozen_body_decoder", None),
                "artifact_fingerprint",
                None,
            ),
            "body_synergy_contract_fingerprint": getattr(
                self.runtime, "body_synergy_contract_fingerprint", None
            ),
            "body_synergy_portable_core_fingerprint": getattr(
                self.runtime, "body_synergy_portable_core_fingerprint", None
            ),
            "teacher_ctrlrange_schema_hash": getattr(
                self.runtime, "ctrlrange_schema_hash", None
            ),
            "physical_signal_schema_version": runtime_physical_schema,
            "latent_checkpoint_dir": getattr(self.runtime, "checkpoint_dir", None),
            "router_schema_hash": self.router.schema_hash,
            "fixture_mode": self.router.fixture_mode,
            "grip_provider_schema_hash": grip_hash,
            "state_dim": self.lab_state_size,
            "task_action_dim": self.task_action_size,
            "latent_action_dim": self.latent_action_size,
            "bounded_residual_dim": self.residual_action_size,
            "bounded_residual_schema_hash": (
                None
                if self.bounded_residual_mask is None
                else self.bounded_residual_mask.schema_hash
            ),
            "body_action_dim": self.router.body_size,
            "right_grip_dim": self.router.right_grip_size,
            "left_neutral_dim": self.router.left_neutral_size,
            "full_action_dim": self.router.full_size,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "left_neutral_value": self.left_neutral_value,
        }
        payload["control_hash"] = _stable_hash(payload)
        return payload

    def decode_numpy(
        self,
        *,
        lab_state: np.ndarray,
        raw_latent: np.ndarray,
        lambda_lab: float | None = None,
        raw_bounded_residual: np.ndarray | None = None,
    ) -> Stage3LABOutput:
        state = _numpy_action("lab_state", lab_state, self.lab_state_size)
        raw = _numpy_action("raw_latent", raw_latent, self.latent_action_size)
        if state.shape[:-1] != raw.shape[:-1]:
            raise ValueError("lab_state and raw_latent batch dimensions must match")
        mu, raw_sigma = self.runtime.prior_raw_numpy(state)
        mu = _numpy_action("prior_mu", mu, self.latent_action_size)
        raw_sigma = _numpy_action("prior_raw_sigma", raw_sigma, self.latent_action_size)
        sigma = np.clip(_softplus_numpy(raw_sigma), self.sigma_min, self.sigma_max)
        scale = self.lambda_lab if lambda_lab is None else float(lambda_lab)
        if scale < 0.0:
            raise ValueError("lambda_lab must be non-negative")
        latent = mu + scale * sigma * np.tanh(raw)
        body = _numpy_action(
            "decoded body_action", self.runtime.decoder_numpy(state, latent), self.router.body_size
        )
        if self.bounded_residual_mask is None:
            if raw_bounded_residual is not None:
                raise ValueError("bounded residual was provided but the controller has no mask")
        else:
            if raw_bounded_residual is None:
                raise ValueError("enabled bounded residual mask requires its task action")
            body = self.bounded_residual_mask.apply_numpy(body, raw_bounded_residual)
        right = (
            np.empty((*state.shape[:-1], 0), dtype=body.dtype)
            if self.right_grip_provider is None
            else self.right_grip_provider.action_numpy(state)
        )
        left = np.full((*state.shape[:-1], self.router.left_neutral_size), self.left_neutral_value)
        full = self.router.merge_numpy(
            body_action=body,
            right_grip_action=right,
            left_neutral_action=left,
        )
        return Stage3LABOutput(
            full, body, right, left, latent, raw, mu, sigma, scale, raw_bounded_residual
        )

    def decode_task_numpy(
        self,
        *,
        lab_state: np.ndarray,
        task_action: np.ndarray,
        lambda_lab: float | None = None,
    ) -> Stage3LABOutput:
        task = _numpy_action("task_action", task_action, self.task_action_size)
        return self.decode_numpy(
            lab_state=lab_state,
            raw_latent=task[..., : self.latent_action_size],
            raw_bounded_residual=(
                None
                if self.residual_action_size == 0
                else task[..., self.latent_action_size :]
            ),
            lambda_lab=lambda_lab,
        )

    def decode_task_with_latent_override_numpy(
        self,
        *,
        lab_state: np.ndarray,
        task_action: np.ndarray,
        effective_latent: np.ndarray,
    ) -> Stage3LABOutput:
        """Decode one task action while intervening on the effective latent.

        This is an evaluation-only causal hook.  The high-level task action is
        still evaluated so its raw latent, optional bounded distal residual,
        grip command and prior statistics remain bound to the trained Stage-3
        policy.  Only the effective latent passed to the frozen body decoder is
        replaced.  Normal training and :meth:`decode_task_numpy` never enter
        this path.
        """

        state = _numpy_action("lab_state", lab_state, self.lab_state_size)
        task = _numpy_action("task_action", task_action, self.task_action_size)
        latent = _numpy_action(
            "effective_latent override",
            effective_latent,
            self.latent_action_size,
        )
        if state.shape[:-1] != task.shape[:-1] or state.shape[:-1] != latent.shape[:-1]:
            raise ValueError(
                "LAB state, task action and effective-latent override batch dimensions must match"
            )

        raw = task[..., : self.latent_action_size]
        mu, raw_sigma = self.runtime.prior_raw_numpy(state)
        mu = _numpy_action("prior_mu", mu, self.latent_action_size)
        raw_sigma = _numpy_action("prior_raw_sigma", raw_sigma, self.latent_action_size)
        sigma = np.clip(_softplus_numpy(raw_sigma), self.sigma_min, self.sigma_max)
        body = _numpy_action(
            "decoded body_action",
            self.runtime.decoder_numpy(state, latent),
            self.router.body_size,
        )
        residual = (
            None
            if self.residual_action_size == 0
            else task[..., self.latent_action_size :]
        )
        if self.bounded_residual_mask is None:
            if residual is not None:
                raise ValueError("task action contains a residual but the controller has no mask")
        else:
            if residual is None:
                raise ValueError("enabled bounded residual mask requires its task action")
            body = self.bounded_residual_mask.apply_numpy(body, residual)
        right = (
            np.empty((*state.shape[:-1], 0), dtype=body.dtype)
            if self.right_grip_provider is None
            else self.right_grip_provider.action_numpy(state)
        )
        left = np.full(
            (*state.shape[:-1], self.router.left_neutral_size),
            self.left_neutral_value,
        )
        full = self.router.merge_numpy(
            body_action=body,
            right_grip_action=right,
            left_neutral_action=left,
        )
        return Stage3LABOutput(
            full,
            body,
            right,
            left,
            latent,
            raw,
            mu,
            sigma,
            self.lambda_lab,
            residual,
        )

    def decode_jax(
        self,
        *,
        lab_state: jax.Array,
        raw_latent: jax.Array,
        lambda_lab: float | jax.Array | None = None,
        raw_bounded_residual: jax.Array | None = None,
    ) -> Stage3LABOutput:
        state = _jax_action("lab_state", lab_state, self.lab_state_size)
        raw = _jax_action("raw_latent", raw_latent, self.latent_action_size)
        if state.shape[:-1] != raw.shape[:-1]:
            raise ValueError("lab_state and raw_latent batch dimensions must match")
        mu, raw_sigma = self.runtime.prior_raw_jax(state)
        mu = _jax_action("prior_mu", mu, self.latent_action_size)
        raw_sigma = _jax_action("prior_raw_sigma", raw_sigma, self.latent_action_size)
        sigma = jnp.clip(jax.nn.softplus(raw_sigma), self.sigma_min, self.sigma_max)
        scale = jnp.asarray(self.lambda_lab if lambda_lab is None else lambda_lab, dtype=state.dtype)
        latent = mu + scale * sigma * jnp.tanh(raw)
        body = _jax_action(
            "decoded body_action", self.runtime.decoder_jax(state, latent), self.router.body_size
        )
        if self.bounded_residual_mask is None:
            if raw_bounded_residual is not None:
                raise ValueError("bounded residual was provided but the controller has no mask")
        else:
            if raw_bounded_residual is None:
                raise ValueError("enabled bounded residual mask requires its task action")
            body = self.bounded_residual_mask.apply_jax(body, raw_bounded_residual)
        right = (
            jnp.empty((*state.shape[:-1], 0), dtype=body.dtype)
            if self.right_grip_provider is None
            else self.right_grip_provider.action_jax(state)
        )
        left = jnp.full(
            (*state.shape[:-1], self.router.left_neutral_size),
            self.left_neutral_value,
            dtype=body.dtype,
        )
        full = self.router.merge_jax(
            body_action=body,
            right_grip_action=right,
            left_neutral_action=left,
        )
        return Stage3LABOutput(
            full, body, right, left, latent, raw, mu, sigma, scale, raw_bounded_residual
        )

    def decode_task_jax(
        self,
        *,
        lab_state: jax.Array,
        task_action: jax.Array,
        lambda_lab: float | jax.Array | None = None,
    ) -> Stage3LABOutput:
        task = _jax_action("task_action", task_action, self.task_action_size)
        return self.decode_jax(
            lab_state=lab_state,
            raw_latent=task[..., : self.latent_action_size],
            raw_bounded_residual=(
                None
                if self.residual_action_size == 0
                else task[..., self.latent_action_size :]
            ),
            lambda_lab=lambda_lab,
        )


class Stage3LabStateBuilder:
    """Build the exact non-finger ``state + phase`` used in distillation.

    The source :class:`BodyObsSchema` must describe the fingerless Stage-2
    teacher.  This builder maps those names into the full Stage-3 model and
    appends exactly one synthesized phase value.  Shuttle, racket and finger
    state never enter the latent prior.
    """

    def __init__(self, *, model: Any, body_schema: Any, expected_state_dim: int) -> None:
        import mujoco

        from environment.overall_environment.src.body_obs_adapter import (
            BodyObsAdapter,
            BodyObsSchema,
        )

        base_size = (
            int(body_schema.kinematic_size)
            + int(body_schema.muscle_size)
            + int(body_schema.touch_size)
        )
        state_schema = BodyObsSchema(
            total_size=base_size + 1,
            kinematic_size=int(body_schema.kinematic_size),
            muscle_size=int(body_schema.muscle_size),
            touch_size=int(body_schema.touch_size),
            goal_size=1,
            action_size=int(body_schema.action_size),
            root_joint_name=str(body_schema.root_joint_name),
            joint_names=tuple(body_schema.joint_names),
            actuator_names=tuple(body_schema.actuator_names),
            touch_sensor_names=tuple(body_schema.touch_sensor_names),
            observation_names=tuple(body_schema.observation_names),
            student_filtered=True,
        )
        if state_schema.total_size != int(expected_state_dim):
            raise ValueError(
                "Stage-3 LAB state dimension does not match latent checkpoint: "
                f"builder={state_schema.total_size}, runtime={expected_state_dim}"
            )
        self.schema = state_schema
        self.expected_state_dim = int(expected_state_dim)
        self._numpy_adapter = BodyObsAdapter(
            expected_obs_size=state_schema.total_size,
            schema=state_schema,
        )

        def named_id(obj: Any, name: str) -> int:
            index = mujoco.mj_name2id(model, obj, name)
            if index < 0:
                raise ValueError(f"Stage-3 model is missing LAB schema name {name!r}")
            return int(index)

        root_id = named_id(mujoco.mjtObj.mjOBJ_JOINT, state_schema.root_joint_name)
        self.root_qadr = int(model.jnt_qposadr[root_id])
        self.root_dadr = int(model.jnt_dofadr[root_id])
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        for name in state_schema.joint_names:
            joint_id = named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = int(model.jnt_qposadr[joint_id])
            dadr = int(model.jnt_dofadr[joint_id])
            joint_type = int(model.jnt_type[joint_id])
            qwidth = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (
                4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            )
            dwidth = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (
                3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            )
            qpos_indices.extend(range(qadr, qadr + qwidth))
            qvel_indices.extend(range(dadr, dadr + dwidth))
        self.qpos_indices = jnp.asarray(qpos_indices, dtype=jnp.int32)
        self.qvel_indices = jnp.asarray(qvel_indices, dtype=jnp.int32)
        muscle_contract = resolve_muscle_channel_contract(
            model,
            state_schema.actuator_names,
        )
        self.actuator_indices = jnp.asarray(muscle_contract.actuator_ids, dtype=jnp.int32)
        self.activation_addresses = jnp.asarray(
            muscle_contract.actuator_actadr,
            dtype=jnp.int32,
        )
        self.sensor_addresses = jnp.asarray(
            [
                int(model.sensor_adr[named_id(mujoco.mjtObj.mjOBJ_SENSOR, name)])
                for name in state_schema.touch_sensor_names
            ],
            dtype=jnp.int32,
        )

    @classmethod
    def from_runtime(cls, *, model: Any, runtime: Any) -> "Stage3LabStateBuilder":
        """Resolve a body schema carried by a runtime or its teacher checkpoint."""
        from environment.overall_environment.src.body_obs_adapter import (
            BodyObsSchema,
        )
        from musclemimic.distill.body_obs_schema import validate_body_obs_schema

        schema_payload = getattr(runtime, "body_obs_schema", None)
        body_schema_keys = {
            "total_size",
            "kinematic_size",
            "muscle_size",
            "touch_size",
            "goal_size",
            "action_size",
            "root_joint_name",
            "joint_names",
            "actuator_names",
            "touch_sensor_names",
            "observation_names",
        }
        if isinstance(schema_payload, dict) and body_schema_keys.issubset(schema_payload):
            validate_body_obs_schema(schema_payload, state_dim=int(runtime.state_dim))
            if int(schema_payload.get("other_size", 0)) != 0:
                raise ValueError("Stage-3 LAB cannot synthesize unknown/condition body channels")
            if int(schema_payload.get("goal_size", -1)) != 1:
                raise ValueError("Stage-3 LAB requires exactly one synthesized phase channel")
            channels = list(schema_payload.get("channels") or [])
            if not channels or channels[-1].get("category") != "goal":
                raise ValueError("Stage-3 LAB requires motion phase as the final state channel")
            categories = [str(channel.get("category")) for channel in channels]
            expected_order = {"kinematic": 0, "muscle": 1, "touch": 2, "goal": 3}
            if categories != sorted(categories, key=expected_order.__getitem__):
                raise ValueError("body observation channels are not in reconstructable policy order")
            runtime_names = tuple(getattr(runtime, "body_actuator_names", ()))
            if runtime_names and tuple(schema_payload["actuator_names"]) != runtime_names:
                raise ValueError("body observation and decoder actuator schemas differ")
            schema_payload = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in schema_payload.items()
                if key in body_schema_keys | {"student_filtered"}
            }
            body_schema = BodyObsSchema(**schema_payload)
        elif schema_payload is not None and not isinstance(schema_payload, dict):
            body_schema = schema_payload
        else:
            raise ValueError(
                "production latent runtime must carry a self-contained body_obs_schema"
            )
        if body_schema is None:
            raise ValueError("could not resolve the Stage-2 body observation schema")
        return cls(
            model=model,
            body_schema=body_schema,
            expected_state_dim=int(runtime.state_dim),
        )

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "schema_version": "stage3_lab_state_v1",
                "state_dim": self.expected_state_dim,
                "root_joint_name": self.schema.root_joint_name,
                "joint_names": self.schema.joint_names,
                "actuator_names": self.schema.actuator_names,
                "touch_sensor_names": self.schema.touch_sensor_names,
                "goal": "swing_phase",
            }
        )

    def build_numpy(self, *, model: Any, data: Any, phase: float) -> np.ndarray:
        return self._numpy_adapter.build_from_mujoco(
            model,
            data,
            goal_obs=np.asarray([np.clip(float(phase), 0.0, 1.0)], dtype=float),
        )

    def build_jax(self, *, data: Any, phase: jax.Array) -> jax.Array:
        qpos = data.qpos
        qvel = data.qvel
        if qpos.ndim != 2:
            raise ValueError("Stage3LabStateBuilder.build_jax expects batched MJX data")
        actuator_ids = self.actuator_indices
        activation_addresses = self.activation_addresses
        muscle = jnp.stack(
            [
                data.actuator_length[:, actuator_ids],
                data.actuator_velocity[:, actuator_ids],
                data.actuator_force[:, actuator_ids],
                data.ctrl[:, actuator_ids],
                data.act[:, activation_addresses],
            ],
            axis=2,
        ).reshape(qpos.shape[0], -1)
        touch = (
            data.sensordata[:, self.sensor_addresses]
            if int(self.sensor_addresses.shape[0])
            else jnp.zeros((qpos.shape[0], 0), dtype=qpos.dtype)
        )
        result = jnp.concatenate(
            [
                qpos[:, self.root_qadr + 2 : self.root_qadr + 7],
                qpos[:, self.qpos_indices],
                qvel[:, self.root_dadr : self.root_dadr + 6],
                qvel[:, self.qvel_indices],
                muscle,
                touch,
                jnp.clip(jnp.asarray(phase), 0.0, 1.0).reshape((-1, 1)),
            ],
            axis=-1,
        )
        if result.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"built JAX LAB state has {result.shape[-1]} values, "
                f"expected {self.expected_state_dim}"
            )
        return result

@dataclass(frozen=True)
class Stage3CurriculumValues:
    lambda_lab: float
    feed_fraction: float
    active_feed_count: int


@dataclass(frozen=True)
class Stage3Curriculum:
    """Ordered fixed-feed -> jitter -> full-bank -> LAB-radius curriculum.

    Feed diversity is fully expanded before the latent radius can grow.  This
    prevents two exploration axes from changing at once and makes the absolute
    environment-step schedule exactly reproducible after resume.
    """

    lambda_start: float = 0.25
    lambda_end: float = 0.5
    fixed_feed_steps: int = 2_000_000
    jitter_feed_count: int = 16
    jitter_expand_steps: int = 4_000_000
    full_bank_expand_steps: int = 8_000_000
    lambda_expand_steps: int = 4_000_000
    gate_min_no_fall_rate: float = 0.95
    fixed_min_hit_rate: float = 0.50
    jitter_min_hit_rate: float = 0.70
    jitter_min_crossed_net_rate: float = 0.50
    full_bank_min_hit_rate: float = 0.85
    full_bank_min_crossed_net_rate: float = 0.75

    def __post_init__(self) -> None:
        if min(float(self.lambda_start), float(self.lambda_end)) < 0.0:
            raise ValueError("curriculum LAB lambda values must be non-negative")
        if int(self.jitter_feed_count) <= 0:
            raise ValueError("jitter_feed_count must be positive")
        if min(
            int(self.fixed_feed_steps),
            int(self.jitter_expand_steps),
            int(self.full_bank_expand_steps),
            int(self.lambda_expand_steps),
        ) < 0:
            raise ValueError("curriculum step counts must be non-negative")
        for name in (
            "gate_min_no_fall_rate",
            "fixed_min_hit_rate",
            "jitter_min_hit_rate",
            "jitter_min_crossed_net_rate",
            "full_bank_min_hit_rate",
            "full_bank_min_crossed_net_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    @property
    def fixed_end(self) -> int:
        return int(self.fixed_feed_steps)

    @property
    def jitter_end(self) -> int:
        return self.fixed_end + int(self.jitter_expand_steps)

    @property
    def full_bank_end(self) -> int:
        return self.jitter_end + int(self.full_bank_expand_steps)

    @property
    def curriculum_end(self) -> int:
        return self.full_bank_end + int(self.lambda_expand_steps)

    def phase(self, effective_steps: int) -> str:
        step = max(0, int(effective_steps))
        if step < self.fixed_end:
            return "fixed_feed"
        if step < self.jitter_end:
            return "intercept_jitter"
        if step < self.full_bank_end:
            return "full_bank_expansion"
        if step < self.curriculum_end:
            return "lambda_expansion"
        return "complete"

    def advance(
        self,
        *,
        effective_steps: int,
        delta_steps: int,
        metrics: Mapping[str, float],
    ) -> tuple[int, dict[str, Any]]:
        """Advance curriculum time, holding each boundary until its gate passes."""
        current = max(0, int(effective_steps))
        proposed = current + max(0, int(delta_steps))
        no_fall_rate = 1.0 - float(metrics.get("fall_rate", float("inf")))
        hit_rate = float(metrics.get("hit_rate", float("-inf")))
        crossed_rate = float(metrics.get("crossed_net_rate", float("-inf")))
        episodes = float(metrics.get("episodes_finished", 0.0))
        gates = (
            (
                self.fixed_end,
                "fixed_feed",
                hit_rate >= float(self.fixed_min_hit_rate),
                float(self.fixed_min_hit_rate),
                None,
            ),
            (
                self.jitter_end,
                "intercept_jitter",
                hit_rate >= float(self.jitter_min_hit_rate)
                and crossed_rate >= float(self.jitter_min_crossed_net_rate),
                float(self.jitter_min_hit_rate),
                float(self.jitter_min_crossed_net_rate),
            ),
            (
                self.full_bank_end,
                "full_bank_expansion",
                hit_rate >= float(self.full_bank_min_hit_rate)
                and crossed_rate >= float(self.full_bank_min_crossed_net_rate),
                float(self.full_bank_min_hit_rate),
                float(self.full_bank_min_crossed_net_rate),
            ),
        )
        gate_report: dict[str, Any] = {
            "checked": False,
            "passed": True,
            "phase": self.phase(current),
            "episodes_finished": episodes,
            "no_fall_rate": no_fall_rate,
            "hit_rate": hit_rate,
            "crossed_net_rate": crossed_rate,
        }
        for boundary, phase_name, task_passed, min_hit, min_crossed in gates:
            if current < boundary <= proposed:
                passed = bool(
                    episodes > 0.0
                    and no_fall_rate >= float(self.gate_min_no_fall_rate)
                    and task_passed
                )
                gate_report.update(
                    {
                        "checked": True,
                        "passed": passed,
                        "phase": phase_name,
                        "boundary_steps": int(boundary),
                        "min_no_fall_rate": float(self.gate_min_no_fall_rate),
                        "min_hit_rate": min_hit,
                        "min_crossed_net_rate": min_crossed,
                    }
                )
                if not passed:
                    proposed = max(current, int(boundary) - 1)
                    break
        return proposed, gate_report

    def values(self, *, env_steps: int, feed_bank_size: int) -> Stage3CurriculumValues:
        if int(feed_bank_size) <= 0:
            raise ValueError("feed_bank_size must be positive")
        step = max(0, int(env_steps))
        fixed_end = self.fixed_end
        jitter_end = self.jitter_end
        full_bank_end = self.full_bank_end
        jitter_target = min(int(feed_bank_size), int(self.jitter_feed_count))

        if step < fixed_end:
            active = 1
        elif step < jitter_end:
            progress = _progress(step - fixed_end, self.jitter_expand_steps)
            active = int(np.ceil(_lerp(1, jitter_target, progress)))
        elif step < full_bank_end:
            progress = _progress(step - jitter_end, self.full_bank_expand_steps)
            active = int(np.ceil(_lerp(jitter_target, int(feed_bank_size), progress)))
        else:
            active = int(feed_bank_size)

        lambda_progress = _progress(step - full_bank_end, self.lambda_expand_steps)
        lambda_lab = _lerp(self.lambda_start, self.lambda_end, lambda_progress)
        active = max(1, min(int(feed_bank_size), int(active)))
        feed_fraction = float(active) / float(feed_bank_size)
        return Stage3CurriculumValues(lambda_lab, feed_fraction, active)


def _progress(step: int, duration: int) -> float:
    if int(duration) <= 0:
        return 1.0
    return float(np.clip(float(step) / float(duration), 0.0, 1.0))


def _lerp(start: float, end: float, progress: float) -> float:
    return float(start) + (float(end) - float(start)) * float(progress)


def _numpy_action(label: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != int(size):
        raise ValueError(f"{label} must have final dimension {size}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _jax_action(label: str, value: Any, size: int) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[-1] != int(size):
        raise ValueError(f"{label} must have final dimension {size}, got {array.shape}")
    return array


def _softplus_numpy(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)
