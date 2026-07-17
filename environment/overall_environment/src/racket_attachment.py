"""Versioned rigid-racket attachment contracts shared by Stages 2 and 3.

The contract is deliberately data-only.  It pins every quantity that changes
the hand-to-stringbed rigid transform or the racket's load on the human, and
uses a canonical JSON SHA-256 fingerprint so silent edits fail closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_SCHEMA = "musclemimic.racket_attachment.v1"
CUSTOM_STRINGBED_CONTACT_MODEL = "custom_force_event_rebound_v1"
DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH = (
    REPO_ROOT / "configs" / "racket_attachment" / "forehand_clear_rigid_v2.json"
)


def canonical_json_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return ``sha256:<hex>`` for canonical UTF-8 JSON ``payload``."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def canonical_contract_fingerprint(document: Mapping[str, Any]) -> str:
    """Fingerprint a contract document, excluding its declared fingerprint."""

    payload = dict(document)
    payload.pop("fingerprint", None)
    return canonical_json_fingerprint(payload)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field} keys mismatch: missing={missing}, extra={extra}")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_float(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be > 0")
    return result


def _require_vector(
    value: object,
    width: int,
    *,
    field: str,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{field} must be a length-{width} JSON array")
    return tuple(
        _require_float(item, field=f"{field}[{index}]", positive=positive)
        for index, item in enumerate(value)
    )


def _require_pose(value: object, *, field: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pose = _require_exact_keys(
        value,
        {"position_m", "quaternion_wxyz"},
        field=field,
    )
    position = _require_vector(pose["position_m"], 3, field=f"{field}.position_m")
    quaternion = _require_vector(
        pose["quaternion_wxyz"],
        4,
        field=f"{field}.quaternion_wxyz",
    )
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{field}.quaternion_wxyz must be unit length, got {norm}")
    return position, quaternion


@dataclass(frozen=True)
class RacketAttachmentContract:
    """Validated exact-child attachment contract."""

    schema: str
    contract_id: str
    attachment_mode: str
    parent_body: str
    relative_position_m: tuple[float, float, float]
    relative_quaternion_wxyz: tuple[float, float, float, float]
    racket_asset_path: str
    racket_asset_sha256: str
    racket_source_body: str
    racket_mass_kg: float
    racket_center_of_mass_m: tuple[float, float, float]
    racket_diagonal_inertia_kg_m2: tuple[float, float, float]
    stringbed_site_name: str
    stringbed_position_m: tuple[float, float, float]
    stringbed_quaternion_wxyz: tuple[float, float, float, float]
    native_hand_racket_contact: bool
    racket_collision_bit: int
    stringbed_contact_model: str
    native_stringbed_proxy_shuttle_contact: bool
    native_racket_frame_shuttle_contact: bool
    stringbed_proxy_geom_name: str
    stringbed_ground_collision_bit: int
    fingerprint: str
    source_path: Path

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical, fingerprint-free JSON payload."""

        return {
            "schema": self.schema,
            "contract_id": self.contract_id,
            "attachment_mode": self.attachment_mode,
            "parent_body": self.parent_body,
            "relative_pose": {
                "position_m": list(self.relative_position_m),
                "quaternion_wxyz": list(self.relative_quaternion_wxyz),
            },
            "racket_asset": {
                "path": self.racket_asset_path,
                "sha256": self.racket_asset_sha256,
            },
            "racket_body": {
                "source_name": self.racket_source_body,
                "mass_kg": self.racket_mass_kg,
                "center_of_mass_m": list(self.racket_center_of_mass_m),
                "diagonal_inertia_kg_m2": list(self.racket_diagonal_inertia_kg_m2),
            },
            "stringbed": {
                "site_name": self.stringbed_site_name,
                "relative_pose": {
                    "position_m": list(self.stringbed_position_m),
                    "quaternion_wxyz": list(self.stringbed_quaternion_wxyz),
                },
            },
            "collision": {
                "native_hand_racket_contact": self.native_hand_racket_contact,
                "racket_collision_bit": self.racket_collision_bit,
                "stringbed_contact_model": self.stringbed_contact_model,
                "native_stringbed_proxy_shuttle_contact": self.native_stringbed_proxy_shuttle_contact,
                "native_racket_frame_shuttle_contact": self.native_racket_frame_shuttle_contact,
                "stringbed_proxy_geom_name": self.stringbed_proxy_geom_name,
                "stringbed_ground_collision_bit": self.stringbed_ground_collision_bit,
            },
        }

    @property
    def asset_path(self) -> Path:
        path = Path(self.racket_asset_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("racket_asset.path must be a repository-relative path without '..'")
        resolved = (REPO_ROOT / path).resolve()
        if not resolved.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError(f"racket asset escapes repository: {resolved}")
        return resolved

    def verify_asset(self) -> Path:
        """Verify the pinned racket asset bytes and return the resolved path."""

        path = self.asset_path
        if not path.is_file():
            raise FileNotFoundError(f"racket asset not found: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual, self.racket_asset_sha256):
            raise ValueError(
                "racket asset SHA-256 mismatch: "
                f"contract={self.racket_asset_sha256}, actual={actual}, path={path}"
            )
        return path


def load_racket_attachment_contract(
    path: str | Path | None = None,
    *,
    verify_asset: bool = True,
) -> RacketAttachmentContract:
    """Load and strictly validate a versioned exact-child contract."""

    source_path = (
        DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH if path is None else Path(path)
    ).resolve()
    try:
        document = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid racket attachment JSON {source_path}: {exc}") from exc

    root = _require_exact_keys(
        document,
        {
            "schema",
            "contract_id",
            "attachment_mode",
            "parent_body",
            "relative_pose",
            "racket_asset",
            "racket_body",
            "stringbed",
            "collision",
            "fingerprint",
        },
        field="contract",
    )
    declared_fingerprint = _require_string(root["fingerprint"], field="fingerprint")
    if not declared_fingerprint.startswith("sha256:") or len(declared_fingerprint) != 71:
        raise ValueError("fingerprint must have form 'sha256:<64 lowercase hex digits>'")
    suffix = declared_fingerprint.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in suffix):
        raise ValueError("fingerprint must contain lowercase hexadecimal digits")
    actual_fingerprint = canonical_contract_fingerprint(root)
    if not hmac.compare_digest(declared_fingerprint, actual_fingerprint):
        raise ValueError(
            "racket attachment fingerprint mismatch: "
            f"declared={declared_fingerprint}, actual={actual_fingerprint}"
        )

    schema = _require_string(root["schema"], field="schema")
    if schema != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported racket attachment schema {schema!r}")
    attachment_mode = _require_string(root["attachment_mode"], field="attachment_mode")
    if attachment_mode != "exact_child":
        raise ValueError(f"unsupported production attachment mode {attachment_mode!r}")

    relative_position, relative_quaternion = _require_pose(
        root["relative_pose"],
        field="relative_pose",
    )
    asset = _require_exact_keys(root["racket_asset"], {"path", "sha256"}, field="racket_asset")
    asset_path = _require_string(asset["path"], field="racket_asset.path")
    asset_sha256 = _require_string(asset["sha256"], field="racket_asset.sha256")
    if len(asset_sha256) != 64 or any(c not in "0123456789abcdef" for c in asset_sha256):
        raise ValueError("racket_asset.sha256 must be 64 lowercase hexadecimal digits")

    body = _require_exact_keys(
        root["racket_body"],
        {"source_name", "mass_kg", "center_of_mass_m", "diagonal_inertia_kg_m2"},
        field="racket_body",
    )
    center_of_mass = _require_vector(
        body["center_of_mass_m"],
        3,
        field="racket_body.center_of_mass_m",
    )
    diagonal_inertia = _require_vector(
        body["diagonal_inertia_kg_m2"],
        3,
        field="racket_body.diagonal_inertia_kg_m2",
        positive=True,
    )

    stringbed = _require_exact_keys(
        root["stringbed"],
        {"site_name", "relative_pose"},
        field="stringbed",
    )
    stringbed_position, stringbed_quaternion = _require_pose(
        stringbed["relative_pose"],
        field="stringbed.relative_pose",
    )

    collision = _require_exact_keys(
        root["collision"],
        {
            "native_hand_racket_contact",
            "racket_collision_bit",
            "stringbed_contact_model",
            "native_stringbed_proxy_shuttle_contact",
            "native_racket_frame_shuttle_contact",
            "stringbed_proxy_geom_name",
            "stringbed_ground_collision_bit",
        },
        field="collision",
    )
    native_contact = collision["native_hand_racket_contact"]
    if not isinstance(native_contact, bool):
        raise ValueError("collision.native_hand_racket_contact must be boolean")
    if native_contact:
        raise ValueError("exact_child production contract requires native hand-racket contact=false")
    collision_bit = collision["racket_collision_bit"]
    if isinstance(collision_bit, bool) or not isinstance(collision_bit, int):
        raise ValueError("collision.racket_collision_bit must be an integer")
    if collision_bit <= 0 or collision_bit & (collision_bit - 1):
        raise ValueError("collision.racket_collision_bit must be one positive collision bit")
    stringbed_contact_model = _require_string(
        collision["stringbed_contact_model"],
        field="collision.stringbed_contact_model",
    )
    if stringbed_contact_model != CUSTOM_STRINGBED_CONTACT_MODEL:
        raise ValueError(
            "production racket contract requires "
            f"collision.stringbed_contact_model={CUSTOM_STRINGBED_CONTACT_MODEL!r}"
        )
    native_proxy_contact = collision["native_stringbed_proxy_shuttle_contact"]
    if not isinstance(native_proxy_contact, bool):
        raise ValueError(
            "collision.native_stringbed_proxy_shuttle_contact must be boolean"
        )
    if native_proxy_contact:
        raise ValueError(
            "custom stringbed contact requires native stringbed-proxy/shuttle contact=false"
        )
    native_frame_contact = collision["native_racket_frame_shuttle_contact"]
    if not isinstance(native_frame_contact, bool):
        raise ValueError("collision.native_racket_frame_shuttle_contact must be boolean")
    if not native_frame_contact:
        raise ValueError("production racket contract requires native frame/shuttle contact=true")
    proxy_geom_name = _require_string(
        collision["stringbed_proxy_geom_name"],
        field="collision.stringbed_proxy_geom_name",
    )
    ground_collision_bit = collision["stringbed_ground_collision_bit"]
    if isinstance(ground_collision_bit, bool) or not isinstance(ground_collision_bit, int):
        raise ValueError("collision.stringbed_ground_collision_bit must be an integer")
    if ground_collision_bit <= 0 or ground_collision_bit & (ground_collision_bit - 1):
        raise ValueError(
            "collision.stringbed_ground_collision_bit must be one positive collision bit"
        )
    if ground_collision_bit == collision_bit:
        raise ValueError(
            "stringbed ground collision bit must differ from the racket/frame collision bit"
        )

    contract = RacketAttachmentContract(
        schema=schema,
        contract_id=_require_string(root["contract_id"], field="contract_id"),
        attachment_mode=attachment_mode,
        parent_body=_require_string(root["parent_body"], field="parent_body"),
        relative_position_m=relative_position,  # type: ignore[arg-type]
        relative_quaternion_wxyz=relative_quaternion,  # type: ignore[arg-type]
        racket_asset_path=asset_path,
        racket_asset_sha256=asset_sha256,
        racket_source_body=_require_string(body["source_name"], field="racket_body.source_name"),
        racket_mass_kg=_require_float(body["mass_kg"], field="racket_body.mass_kg", positive=True),
        racket_center_of_mass_m=center_of_mass,  # type: ignore[arg-type]
        racket_diagonal_inertia_kg_m2=diagonal_inertia,  # type: ignore[arg-type]
        stringbed_site_name=_require_string(stringbed["site_name"], field="stringbed.site_name"),
        stringbed_position_m=stringbed_position,  # type: ignore[arg-type]
        stringbed_quaternion_wxyz=stringbed_quaternion,  # type: ignore[arg-type]
        native_hand_racket_contact=native_contact,
        racket_collision_bit=collision_bit,
        stringbed_contact_model=stringbed_contact_model,
        native_stringbed_proxy_shuttle_contact=native_proxy_contact,
        native_racket_frame_shuttle_contact=native_frame_contact,
        stringbed_proxy_geom_name=proxy_geom_name,
        stringbed_ground_collision_bit=ground_collision_bit,
        fingerprint=declared_fingerprint,
        source_path=source_path,
    )
    if canonical_json_fingerprint(contract.to_payload()) != contract.fingerprint:
        raise ValueError("parsed racket attachment contract does not round-trip canonically")
    if verify_asset:
        contract.verify_asset()
    return contract


def validate_racket_spec_against_contract(
    spec: object,
    contract: RacketAttachmentContract,
    *,
    atol: float = 1e-10,
) -> None:
    """Validate the pinned inertial and stringbed fields on a racket ``MjSpec``.

    The function intentionally uses the small public ``MjSpec`` surface via
    duck typing, keeping contract parsing independent of MuJoCo import time.
    """

    body_lookup = getattr(spec, "body", None)
    site_lookup = getattr(spec, "site", None)
    geom_lookup = getattr(spec, "geom", None)
    if not callable(body_lookup) or not callable(site_lookup) or not callable(geom_lookup):
        raise TypeError("spec must provide callable body(name), site(name), and geom(name) lookups")
    body = body_lookup(contract.racket_source_body)
    if body is None:
        raise ValueError(
            f"racket asset is missing root body {contract.racket_source_body!r}"
        )
    site = site_lookup(contract.stringbed_site_name)
    if site is None:
        raise ValueError(
            f"racket asset is missing stringbed site {contract.stringbed_site_name!r}"
        )
    proxy_geom = geom_lookup(contract.stringbed_proxy_geom_name)
    if proxy_geom is None:
        raise ValueError(
            "racket asset is missing stringbed ground-contact proxy "
            f"{contract.stringbed_proxy_geom_name!r}"
        )

    def check_scalar(actual: object, expected: float, field: str) -> None:
        value = float(actual)
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=atol):
            raise ValueError(f"racket asset {field} mismatch: {value} != {expected}")

    def check_vector(actual: object, expected: tuple[float, ...], field: str) -> None:
        values = tuple(float(value) for value in actual)  # type: ignore[arg-type]
        if len(values) != len(expected) or any(
            not math.isclose(value, target, rel_tol=0.0, abs_tol=atol)
            for value, target in zip(values, expected, strict=False)
        ):
            raise ValueError(f"racket asset {field} mismatch: {values} != {expected}")

    check_scalar(body.mass, contract.racket_mass_kg, "mass")
    check_vector(body.ipos, contract.racket_center_of_mass_m, "center of mass")
    check_vector(body.inertia, contract.racket_diagonal_inertia_kg_m2, "diagonal inertia")
    check_vector(site.pos, contract.stringbed_position_m, "stringbed position")
    check_vector(site.quat, contract.stringbed_quaternion_wxyz, "stringbed quaternion")
