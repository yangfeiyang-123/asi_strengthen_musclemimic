"""Self-contained body observation ABI for latent deployment."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from musclemimic.distill.action_schema import ordered_schema_hash


BODY_OBS_SCHEMA_VERSION = "body_obs_v1"


def build_body_obs_schema(
    *,
    env: Any,
    spec: Any,
    actuator_names: Iterable[str],
    channels: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact state+phase payload needed without a teacher checkpoint."""
    names = [str(name) for name in actuator_names]
    if len(channels) != int(spec.student_obs_dim):
        raise ValueError(
            f"body observation channels={len(channels)} != student_obs_dim={spec.student_obs_dim}"
        )
    base = _base_env(env)
    root_joint_name = str(getattr(base, "root_free_joint_xml_name", "root"))

    joint_names = _ordered_unique(
        str(channel["joint_name"])
        for channel in channels
        if channel.get("joint_name") not in {None, root_joint_name}
    )
    touch_names = _ordered_unique(
        str(channel["xml_name"])
        for channel in channels
        if str(channel.get("entry", "")).startswith("touch_") and channel.get("xml_name") is not None
    )
    observation_names = _ordered_unique(str(channel["entry"]) for channel in channels)

    phase_index = spec.phase_student_index
    category_counts = {"kinematic": 0, "muscle": 0, "touch": 0, "goal": 0, "other": 0}
    semantic_channels: list[dict[str, Any]] = []
    for channel in channels:
        entry = str(channel.get("entry", "unknown"))
        student_index = int(channel["student_index"])
        if phase_index is not None and student_index == int(phase_index):
            category = "goal"
        elif entry in {"q_free_joint", "q_all_pos", "dq_free_joint", "dq_all_vel"}:
            category = "kinematic"
        elif entry.startswith("muscle_"):
            category = "muscle"
        elif entry.startswith("touch_"):
            category = "touch"
        else:
            category = "other"
        category_counts[category] += 1
        semantic_channels.append(
            {
                key: channel[key]
                for key in (
                    "name",
                    "entry",
                    "entry_type",
                    "entry_offset",
                    "student_index",
                    "joint_name",
                    "actuator_name",
                    "xml_name",
                )
                if key in channel
            }
            | {"category": category}
        )

    semantic = {
        "schema_version": BODY_OBS_SCHEMA_VERSION,
        "total_size": int(spec.student_obs_dim),
        "kinematic_size": int(category_counts["kinematic"]),
        "muscle_size": int(category_counts["muscle"]),
        "touch_size": int(category_counts["touch"]),
        "goal_size": int(category_counts["goal"]),
        "other_size": int(category_counts["other"]),
        "action_size": len(names),
        "root_joint_name": root_joint_name,
        "joint_names": joint_names,
        "actuator_names": names,
        "touch_sensor_names": touch_names,
        "observation_names": observation_names,
        "student_filtered": True,
        "channels": semantic_channels,
    }
    semantic_hash = ordered_schema_hash(kind="body_observation", payload=semantic)
    return semantic | {
        "semantic_hash": semantic_hash,
        # Provenance is deliberately outside semantic_hash. Moving a checkpoint
        # or recollecting identical tensors must not change the policy ABI.
        "provenance": dict(provenance or {}),
    }


def validate_body_obs_schema(schema: dict[str, Any], *, state_dim: int | None = None) -> str:
    supplied = schema.get("semantic_hash")
    if not supplied:
        raise ValueError("body observation schema is missing semantic_hash")
    semantic = {key: value for key, value in schema.items() if key not in {"semantic_hash", "provenance"}}
    actual = ordered_schema_hash(kind="body_observation", payload=semantic)
    if str(supplied) != actual:
        raise ValueError(f"body observation semantic hash mismatch: supplied={supplied} computed={actual}")
    if state_dim is not None and int(schema.get("total_size", -1)) != int(state_dim):
        raise ValueError(
            f"body observation total_size={schema.get('total_size')} != state_dim={int(state_dim)}"
        )
    channels = list(schema.get("channels") or [])
    if len(channels) != int(schema.get("total_size", -1)):
        raise ValueError("body observation channel count differs from total_size")
    if [int(channel.get("student_index", -1)) for channel in channels] != list(range(len(channels))):
        raise ValueError("body observation channels are not in exact student policy order")
    return actual


def _base_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))
