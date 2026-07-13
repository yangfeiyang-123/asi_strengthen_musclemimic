"""Name-based finger isolation contracts shared by training and deployment.

This module is intentionally independent of the Stage-3 environment and policy
implementations.  It provides the contracts those components need:

* an exhaustive body/right-grip/left-neutral actuator partition;
* a named observation schema that can remove finger-owned fields;
* deterministic schema hashes and fail-fast compatibility checks; and
* paired clean/perturbed robustness metrics.

No partition in this module relies on positional actuator slices.  Model order
is preserved, but ownership is derived exclusively from names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np


FingerSide = Literal["right", "left"]


# MyoFullBody model order is recorded here for readable manifests.  Partition
# construction still follows the order supplied by the runtime model.
RIGHT_FINGER_ACTUATOR_NAMES: tuple[str, ...] = (
    "FDS5",
    "FDS4",
    "FDS3",
    "FDS2",
    "FDP5",
    "FDP4",
    "FDP3",
    "FDP2",
    "EDC5",
    "EDC4",
    "EDC3",
    "EDC2",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "LU_RB2",
    "UI_UB2",
    "RI3",
    "LU_RB3",
    "UI_UB3",
    "RI4",
    "LU_RB4",
    "UI_UB4",
    "RI5",
    "LU_RB5",
    "UI_UB5",
)
LEFT_FINGER_ACTUATOR_NAMES: tuple[str, ...] = tuple(
    f"{name}_left" for name in RIGHT_FINGER_ACTUATOR_NAMES
)

_RIGHT_FINGER_ACTUATOR_SET = frozenset(RIGHT_FINGER_ACTUATOR_NAMES)
_LEFT_FINGER_ACTUATOR_SET = frozenset(LEFT_FINGER_ACTUATOR_NAMES)
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


class SchemaMismatchError(ValueError):
    """Raised when a runtime name/dimension schema differs from a checkpoint contract."""


def finger_actuator_side(name: str | None) -> FingerSide | None:
    """Return the hand that owns an exact MyoFullBody finger actuator name."""
    if name in _RIGHT_FINGER_ACTUATOR_SET:
        return "right"
    if name in _LEFT_FINGER_ACTUATOR_SET:
        return "left"
    return None


def finger_joint_side(name: str | None) -> FingerSide | None:
    """Return the hand that owns a MyoFullBody finger joint name."""
    if not name or not name.startswith(_FINGER_JOINT_PREFIXES):
        return None
    if name.endswith("_r"):
        return "right"
    if name.endswith("_l"):
        return "left"
    return None


@dataclass(frozen=True)
class FingerActuatorPartition:
    """Exhaustive, name-based ownership for a full-finger MyoFullBody model.

    ``left_neutral_action`` is a required input to :meth:`merge`; the third
    owner can therefore never be silently omitted or confused with an
    unassigned actuator.
    """

    all_actuator_names: tuple[str, ...]
    body_actuator_names: tuple[str, ...]
    right_grip_actuator_names: tuple[str, ...]
    left_neutral_actuator_names: tuple[str, ...]
    body_indices: np.ndarray
    right_grip_indices: np.ndarray
    left_neutral_indices: np.ndarray
    expected_sizes: tuple[int, int, int] = (354, 31, 31)

    def __post_init__(self) -> None:
        all_names = tuple(str(name) for name in self.all_actuator_names)
        body_names = tuple(str(name) for name in self.body_actuator_names)
        right_names = tuple(str(name) for name in self.right_grip_actuator_names)
        left_names = tuple(str(name) for name in self.left_neutral_actuator_names)
        _reject_duplicates("all_actuator_names", all_names)
        _reject_duplicates("body_actuator_names", body_names)
        _reject_duplicates("right_grip_actuator_names", right_names)
        _reject_duplicates("left_neutral_actuator_names", left_names)

        body_set = set(body_names)
        right_set = set(right_names)
        left_set = set(left_names)
        overlaps = sorted((body_set & right_set) | (body_set & left_set) | (right_set & left_set))
        if overlaps:
            raise ValueError(f"actuators have multiple owners: {overlaps}")
        all_set = set(all_names)
        owned_set = body_set | right_set | left_set
        missing = sorted(owned_set - all_set)
        unowned = sorted(all_set - owned_set)
        if missing:
            raise ValueError(f"owned actuators missing from all_actuator_names: {missing}")
        if unowned:
            raise ValueError(f"actuators have no explicit body/right_grip/left_neutral owner: {unowned}")

        expected_body, expected_right, expected_left = self.expected_sizes
        for owner, names, expected in (
            ("body", body_names, expected_body),
            ("right_grip", right_names, expected_right),
            ("left_neutral", left_names, expected_left),
        ):
            if len(names) != expected:
                raise ValueError(f"{owner} actuator partition must contain {expected} names, got {len(names)}")
        if len(all_names) != expected_body + expected_right + expected_left:
            raise ValueError(
                "full actuator partition size mismatch: "
                f"expected {expected_body + expected_right + expected_left}, got {len(all_names)}"
            )

        index_by_name = {name: index for index, name in enumerate(all_names)}
        body_indices = _validate_indices("body_indices", self.body_indices, body_names, index_by_name)
        right_indices = _validate_indices(
            "right_grip_indices", self.right_grip_indices, right_names, index_by_name
        )
        left_indices = _validate_indices(
            "left_neutral_indices", self.left_neutral_indices, left_names, index_by_name
        )
        object.__setattr__(self, "all_actuator_names", all_names)
        object.__setattr__(self, "body_actuator_names", body_names)
        object.__setattr__(self, "right_grip_actuator_names", right_names)
        object.__setattr__(self, "left_neutral_actuator_names", left_names)
        object.__setattr__(self, "body_indices", body_indices)
        object.__setattr__(self, "right_grip_indices", right_indices)
        object.__setattr__(self, "left_neutral_indices", left_indices)

    @classmethod
    def from_actuator_names(
        cls,
        actuator_names: Sequence[str],
        *,
        expected_sizes: tuple[int, int, int] = (354, 31, 31),
    ) -> "FingerActuatorPartition":
        all_names = tuple(str(name) for name in actuator_names)
        _reject_duplicates("all_actuator_names", all_names)
        right_names = tuple(name for name in all_names if finger_actuator_side(name) == "right")
        left_names = tuple(name for name in all_names if finger_actuator_side(name) == "left")
        finger_names = set(right_names) | set(left_names)
        body_names = tuple(name for name in all_names if name not in finger_names)
        index_by_name = {name: index for index, name in enumerate(all_names)}
        return cls(
            all_actuator_names=all_names,
            body_actuator_names=body_names,
            right_grip_actuator_names=right_names,
            left_neutral_actuator_names=left_names,
            body_indices=np.asarray([index_by_name[name] for name in body_names], dtype=np.int32),
            right_grip_indices=np.asarray([index_by_name[name] for name in right_names], dtype=np.int32),
            left_neutral_indices=np.asarray([index_by_name[name] for name in left_names], dtype=np.int32),
            expected_sizes=tuple(int(size) for size in expected_sizes),
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
                "kind": "finger_actuator_partition_v1",
                "all": self.all_actuator_names,
                "body": self.body_actuator_names,
                "right_grip": self.right_grip_actuator_names,
                "left_neutral": self.left_neutral_actuator_names,
            }
        )

    def source_labels(self) -> list[str]:
        labels = [""] * self.full_size
        for indices, label in (
            (self.body_indices, "body"),
            (self.right_grip_indices, "right_grip"),
            (self.left_neutral_indices, "left_neutral"),
        ):
            for index in indices:
                labels[int(index)] = label
        if any(not label for label in labels):
            raise RuntimeError("partition invariant violated: an actuator has no source label")
        return labels

    def neutral_left_action(self, value: float = 0.0, *, dtype: Any = float) -> np.ndarray:
        if not np.isfinite(value):
            raise ValueError("left neutral value must be finite")
        return np.full(self.left_neutral_size, value, dtype=dtype)

    def merge(
        self,
        *,
        body_action: np.ndarray,
        right_grip_action: np.ndarray,
        left_neutral_action: np.ndarray,
    ) -> np.ndarray:
        body = _as_vector("body_action", body_action, self.body_size)
        right = _as_vector("right_grip_action", right_grip_action, self.right_grip_size)
        left = _as_vector("left_neutral_action", left_neutral_action, self.left_neutral_size)
        full = np.empty(self.full_size, dtype=np.result_type(body, right, left))
        full[self.body_indices] = body
        full[self.right_grip_indices] = right
        full[self.left_neutral_indices] = left
        return full

    def assert_compatible(self, runtime_actuator_names: Sequence[str]) -> None:
        runtime = FingerActuatorPartition.from_actuator_names(
            runtime_actuator_names,
            expected_sizes=self.expected_sizes,
        )
        if runtime.schema_hash != self.schema_hash:
            raise SchemaMismatchError(
                "actuator schema hash mismatch: "
                f"expected={self.schema_hash} runtime={runtime.schema_hash}; "
                "actuator dimensions, names, order, or ownership changed"
            )


@dataclass(frozen=True)
class ObservationField:
    """One contiguous named field in a flattened observation vector."""

    feature_name: str
    width: int = 1
    joint_name: str | None = None
    actuator_name: str | None = None

    def __post_init__(self) -> None:
        if not str(self.feature_name):
            raise ValueError("feature_name must be non-empty")
        if int(self.width) <= 0:
            raise ValueError(f"observation field {self.feature_name!r} width must be positive")
        if self.joint_name is not None and self.actuator_name is not None:
            raise ValueError(
                f"observation field {self.feature_name!r} cannot reference both a joint and an actuator"
            )
        object.__setattr__(self, "feature_name", str(self.feature_name))
        object.__setattr__(self, "width", int(self.width))
        if self.joint_name is not None:
            object.__setattr__(self, "joint_name", str(self.joint_name))
        if self.actuator_name is not None:
            object.__setattr__(self, "actuator_name", str(self.actuator_name))


@dataclass(frozen=True)
class NamedObservationSchema:
    """Dimensioned observation fields with stable joint/actuator provenance."""

    fields: tuple[ObservationField, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("observation schema must contain at least one field")
        if not all(isinstance(field, ObservationField) for field in fields):
            raise TypeError("all observation schema fields must be ObservationField instances")
        object.__setattr__(self, "fields", fields)

    @property
    def total_size(self) -> int:
        return sum(field.width for field in self.fields)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(field.feature_name for field in self.fields)

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "kind": "named_observation_schema_v1",
                "fields": [
                    {
                        "feature_name": field.feature_name,
                        "width": field.width,
                        "joint_name": field.joint_name,
                        "actuator_name": field.actuator_name,
                    }
                    for field in self.fields
                ],
                "total_size": self.total_size,
            }
        )

    def without_fingers(
        self,
        *,
        sides: Iterable[FingerSide] = ("right", "left"),
        finger_feature_names: Iterable[str] = (),
    ) -> "ObservationFilter":
        removed_sides = _normalise_sides(sides)
        explicit_features = {str(name) for name in finger_feature_names}
        kept_fields: list[ObservationField] = []
        removed_fields: list[ObservationField] = []
        kept_indices: list[int] = []
        cursor = 0
        for field in self.fields:
            field_side = finger_joint_side(field.joint_name) or finger_actuator_side(field.actuator_name)
            should_remove = (
                (field_side is not None and field_side in removed_sides)
                or field.feature_name in explicit_features
            )
            indices = range(cursor, cursor + field.width)
            if should_remove:
                removed_fields.append(field)
            else:
                kept_fields.append(field)
                kept_indices.extend(indices)
            cursor += field.width
        if not kept_fields:
            raise ValueError("finger filtering removed every observation field")
        return ObservationFilter(
            source_schema=self,
            target_schema=NamedObservationSchema(tuple(kept_fields)),
            kept_indices=np.asarray(kept_indices, dtype=np.int32),
            removed_fields=tuple(removed_fields),
        )

    def assert_compatible(self, runtime_schema: "NamedObservationSchema") -> None:
        if self.schema_hash != runtime_schema.schema_hash:
            raise SchemaMismatchError(
                "observation schema hash mismatch: "
                f"expected={self.schema_hash} runtime={runtime_schema.schema_hash}; "
                "feature dimensions, names, order, or provenance changed"
            )


@dataclass(frozen=True)
class ObservationFilter:
    source_schema: NamedObservationSchema
    target_schema: NamedObservationSchema
    kept_indices: np.ndarray
    removed_fields: tuple[ObservationField, ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.kept_indices, dtype=np.int32)
        if indices.shape != (self.target_schema.total_size,):
            raise ValueError(
                "kept_indices dimension does not match target observation schema: "
                f"{indices.shape} vs ({self.target_schema.total_size},)"
            )
        if np.any(indices < 0) or np.any(indices >= self.source_schema.total_size):
            raise ValueError("kept_indices contains an out-of-bounds source observation index")
        if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
            raise ValueError("kept_indices must be unique and preserve source observation order")
        object.__setattr__(self, "kept_indices", indices)
        object.__setattr__(self, "removed_fields", tuple(self.removed_fields))

    @property
    def removed_feature_names(self) -> tuple[str, ...]:
        return tuple(field.feature_name for field in self.removed_fields)

    @property
    def schema_hash(self) -> str:
        return _stable_hash(
            {
                "kind": "observation_filter_v1",
                "source": self.source_schema.schema_hash,
                "target": self.target_schema.schema_hash,
                "kept_indices": self.kept_indices.tolist(),
            }
        )

    def apply(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim == 0 or array.shape[-1] != self.source_schema.total_size:
            raise ValueError(
                "observation last dimension must match source schema size "
                f"{self.source_schema.total_size}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("observation contains non-finite values")
        return np.take(array, self.kept_indices, axis=-1)

    def assert_source_schema(self, runtime_schema: NamedObservationSchema) -> None:
        self.source_schema.assert_compatible(runtime_schema)


@dataclass(frozen=True)
class PairedMetricRule:
    metric_name: str
    lower_is_better: bool
    max_relative_degradation: float | None = None
    max_absolute_degradation: float | None = None

    def __post_init__(self) -> None:
        if not str(self.metric_name):
            raise ValueError("metric_name must be non-empty")
        if self.max_relative_degradation is None and self.max_absolute_degradation is None:
            raise ValueError(f"paired metric {self.metric_name!r} must define at least one threshold")
        for name, value in (
            ("max_relative_degradation", self.max_relative_degradation),
            ("max_absolute_degradation", self.max_absolute_degradation),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be a finite non-negative value")


@dataclass(frozen=True)
class PairedMetricResult:
    clean_mean: float
    perturbed_mean: float
    absolute_degradation: float
    relative_degradation: float
    max_pair_degradation: float
    passed: bool


@dataclass(frozen=True)
class PairedRobustnessReport:
    pair_count: int
    seed_hash: str | None
    metrics: Mapping[str, PairedMetricResult]
    passed: bool


def compare_paired_metrics(
    clean_metrics: Mapping[str, Sequence[float] | np.ndarray],
    perturbed_metrics: Mapping[str, Sequence[float] | np.ndarray],
    rules: Sequence[PairedMetricRule],
    *,
    clean_seeds: Sequence[Any] | None = None,
    perturbed_seeds: Sequence[Any] | None = None,
    relative_epsilon: float = 1e-12,
) -> PairedRobustnessReport:
    """Compare matched clean/perturbed rollouts with direction-aware gates.

    Each metric array must contain one scalar per paired rollout in identical
    order.  When seeds are supplied, both ordered seed sequences are required
    and must match exactly; their hash can be persisted in evaluation reports.
    """
    if relative_epsilon <= 0.0 or not np.isfinite(relative_epsilon):
        raise ValueError("relative_epsilon must be finite and positive")
    rule_list = tuple(rules)
    if not rule_list:
        raise ValueError("at least one paired metric rule is required")
    _reject_duplicates("paired metric rules", tuple(rule.metric_name for rule in rule_list))

    clean_arrays: dict[str, np.ndarray] = {}
    perturbed_arrays: dict[str, np.ndarray] = {}
    pair_count: int | None = None
    for rule in rule_list:
        name = rule.metric_name
        if name not in clean_metrics or name not in perturbed_metrics:
            raise KeyError(f"paired metric {name!r} is missing from clean or perturbed metrics")
        clean = _as_metric_array(f"clean[{name}]", clean_metrics[name])
        perturbed = _as_metric_array(f"perturbed[{name}]", perturbed_metrics[name])
        if clean.shape != perturbed.shape:
            raise ValueError(f"paired metric {name!r} shape mismatch: {clean.shape} vs {perturbed.shape}")
        if pair_count is None:
            pair_count = int(clean.shape[0])
        elif clean.shape[0] != pair_count:
            raise ValueError(
                f"paired metric {name!r} has {clean.shape[0]} pairs; expected {pair_count}"
            )
        clean_arrays[name] = clean
        perturbed_arrays[name] = perturbed
    assert pair_count is not None

    seed_hash = _validate_paired_seeds(clean_seeds, perturbed_seeds, pair_count)
    results: dict[str, PairedMetricResult] = {}
    for rule in rule_list:
        clean = clean_arrays[rule.metric_name]
        perturbed = perturbed_arrays[rule.metric_name]
        signed_pair_degradation = perturbed - clean if rule.lower_is_better else clean - perturbed
        absolute_degradation = max(0.0, float(np.mean(signed_pair_degradation)))
        denominator = max(abs(float(np.mean(clean))), relative_epsilon)
        relative_degradation = absolute_degradation / denominator
        passed = True
        if rule.max_relative_degradation is not None:
            passed = passed and relative_degradation <= rule.max_relative_degradation
        if rule.max_absolute_degradation is not None:
            passed = passed and absolute_degradation <= rule.max_absolute_degradation
        results[rule.metric_name] = PairedMetricResult(
            clean_mean=float(np.mean(clean)),
            perturbed_mean=float(np.mean(perturbed)),
            absolute_degradation=absolute_degradation,
            relative_degradation=relative_degradation,
            max_pair_degradation=max(0.0, float(np.max(signed_pair_degradation))),
            passed=bool(passed),
        )
    return PairedRobustnessReport(
        pair_count=pair_count,
        seed_hash=seed_hash,
        metrics=results,
        passed=all(result.passed for result in results.values()),
    )


def _validate_indices(
    label: str,
    indices: np.ndarray,
    names: tuple[str, ...],
    index_by_name: Mapping[str, int],
) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int32)
    expected = np.asarray([index_by_name[name] for name in names], dtype=np.int32)
    if array.shape != expected.shape or not np.array_equal(array, expected):
        raise ValueError(f"{label} must match its actuator names in all_actuator_names order")
    return array


def _reject_duplicates(label: str, values: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {duplicates}")


def _as_vector(label: str, value: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (expected_size,):
        raise ValueError(f"{label} must have shape ({expected_size},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite numeric values")
    return array


def _normalise_sides(sides: Iterable[FingerSide]) -> frozenset[FingerSide]:
    if isinstance(sides, str):
        side_values = (sides,)
    else:
        side_values = tuple(sides)
    invalid = sorted(set(side_values) - {"right", "left"})
    if invalid:
        raise ValueError(f"finger sides must be 'right' and/or 'left', got {invalid}")
    if not side_values:
        raise ValueError("at least one finger side must be selected")
    return frozenset(side_values)  # type: ignore[arg-type]


def _as_metric_array(label: str, value: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional paired array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _validate_paired_seeds(
    clean_seeds: Sequence[Any] | None,
    perturbed_seeds: Sequence[Any] | None,
    pair_count: int,
) -> str | None:
    if clean_seeds is None and perturbed_seeds is None:
        return None
    if clean_seeds is None or perturbed_seeds is None:
        raise ValueError("clean and perturbed runs must provide the same ordered seeds")
    clean = tuple(clean_seeds)
    perturbed = tuple(perturbed_seeds)
    if clean != perturbed:
        raise ValueError("clean and perturbed runs must use the same ordered seeds")
    if len(clean) != pair_count:
        raise ValueError(f"paired seed count {len(clean)} does not match metric pair count {pair_count}")
    return _stable_hash({"kind": "paired_rollout_seeds_v1", "seeds": clean})


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
