"""Simulation-side physiology and kinetic-chain report primitives.

The functions are data-only and CPU-only.  They accept rollout arrays after
collection, so evaluation never has to mutate an environment or checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_EXCITATION_SEMANTICS,
    MUSCLE_EXCITATION_SOURCE,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    physical_ctrl_to_effective_muscle_excitation,
    physical_signal_metadata,
    validate_activation_valid_mask,
    validate_muscle_channel_contract,
    validate_physical_signal_semantics,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
    validate_unit_muscle_excitation,
)
from musclemimic.evaluation.emg_eval import (
    cocontraction_index,
    validate_simulation_policy_evidence,
)
from musclemimic.physiology.anatomical_groups import (
    AnatomicalTaxonomy,
    build_intra_muscle_spec,
    load_anatomical_taxonomy,
)
from musclemimic.physiology.continuity_groups import (
    FascicleContinuityGraph,
    build_fascicle_continuity_spec,
    load_fascicle_continuity_graph,
)
from musclemimic.physiology.intra_muscle import (
    FascicleContinuitySpec,
    IntraMuscleSpec,
    exact_exo_imr,
    robust_fascicle_continuity,
    robust_intra_muscle_consistency,
)
from musclemimic.physiology.synergy_binding import (
    SYNERGY_SCHEMA_HASH_FIELDS,
    assert_taxonomy_matches_ordered_muscles,
    taxonomy_ordered_muscle_schema_hash,
)

PHYSIOLOGY_METRICS_SCHEMA_VERSION = "simulation_physiology_v2"
PHYSIOLOGY_REPORT_SCHEMA_VERSION = "simulation_physiology_v3"
PHYSIOLOGY_CONFIG_SCHEMA_VERSION = "simulation_physiology_config_v1"
PHYSIOLOGY_LINEAGE_SCHEMA_VERSION = "simulation_physiology_lineage_v2"
PHYSIOLOGY_SIGNAL_CONTRACT_SCHEMA_VERSION = "simulation_physiology_physical_signal_v2"
INTRA_MUSCLE_DIAGNOSTICS_SCHEMA_VERSION = "simulation_intra_muscle_diagnostics_v1"
FASCICLE_CONTINUITY_DIAGNOSTICS_SCHEMA_VERSION = "simulation_fascicle_continuity_diagnostics_v1"


def muscle_timing_metrics(
    values: np.ndarray,
    *,
    muscle_names: Sequence[str],
    impact_frames: np.ndarray,
    sampling_rate_hz: float,
    onset_fraction: float = 0.2,
) -> dict[str, Any]:
    """Report peak, onset and integral relative to measured impact."""

    signal = _trial_time_channel(values, "muscle signal", nonnegative=True)
    names = _names(muscle_names, signal.shape[2], "muscle_names")
    impacts = _impact_frames(impact_frames, signal)
    fs = _positive_scalar(sampling_rate_hz, "sampling_rate_hz")
    if not 0.0 < float(onset_fraction) < 1.0:
        raise ValueError("onset_fraction must lie in (0,1)")
    peak_index = np.argmax(signal, axis=1)
    peak_value = np.max(signal, axis=1)
    onset_index = np.zeros_like(peak_index)
    for trial in range(signal.shape[0]):
        for channel in range(signal.shape[2]):
            threshold = float(onset_fraction) * peak_value[trial, channel]
            candidates = np.flatnonzero(signal[trial, :, channel] >= threshold)
            onset_index[trial, channel] = candidates[0] if candidates.size else signal.shape[1] - 1
    peak_time = (peak_index - impacts[:, None]) / fs
    onset_time = (onset_index - impacts[:, None]) / fs
    integrated = np.trapezoid(signal, dx=1.0 / fs, axis=1)
    return {
        name: {
            "peak_value": _summary(peak_value[:, index]),
            "peak_time_from_impact_s": _summary(peak_time[:, index]),
            "onset_time_from_impact_s": _summary(onset_time[:, index]),
            "integrated_signal": _summary(integrated[:, index]),
        }
        for index, name in enumerate(names)
    }


def phase_signal_summary(
    values: np.ndarray,
    phase_id: np.ndarray,
    *,
    channel_names: Sequence[str],
) -> dict[str, Any]:
    signal = _trial_time_channel(values, "phase signal", nonnegative=False)
    names = _names(channel_names, signal.shape[2], "channel_names")
    phase = np.asarray(phase_id)
    if phase.ndim == 1 and signal.shape[0] == 1:
        phase = phase[None, :]
    if phase.shape != signal.shape[:2] or not np.issubdtype(phase.dtype, np.integer):
        raise ValueError("phase_id must be integer [trial,time]")
    result: dict[str, Any] = {}
    for phase_value in sorted(np.unique(phase).astype(int).tolist()):
        mask = phase == phase_value
        selected = signal[mask]
        result[str(phase_value)] = {name: _summary(selected[:, channel]) for channel, name in enumerate(names)}
    return result


def synergy_residual_metrics(
    physical_excitation: np.ndarray,
    synergy_reconstruction: np.ndarray,
    *,
    residual: np.ndarray | None = None,
    allowed_residual_mask: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Measure reconstruction residual and detect residual bypass outside its mask."""

    target = _trial_time_channel(physical_excitation, "physical_excitation", nonnegative=True)
    reconstruction = _trial_time_channel(
        synergy_reconstruction,
        "synergy_reconstruction",
        nonnegative=True,
    )
    if target.shape != reconstruction.shape:
        raise ValueError("physical excitation and synergy reconstruction shapes differ")
    residual_value = (
        target - reconstruction
        if residual is None
        else _trial_time_channel(
            residual,
            "residual",
            nonnegative=False,
        )
    )
    if residual_value.shape != target.shape:
        raise ValueError("residual shape differs from physical excitation")
    residual_energy = np.sum(np.square(residual_value), axis=(0, 1))
    target_energy = np.sum(np.square(target), axis=(0, 1))
    result: dict[str, Any] = {
        "residual_energy_ratio": float(np.sum(residual_energy) / max(float(np.sum(target_energy)), float(epsilon))),
        "per_channel_residual_energy_ratio": (residual_energy / np.maximum(target_energy, float(epsilon))).tolist(),
    }
    if allowed_residual_mask is not None:
        mask = np.asarray(allowed_residual_mask, dtype=bool)
        if mask.shape != (target.shape[2],) or not np.any(mask):
            raise ValueError("allowed_residual_mask must select at least one channel")
        total = float(np.sum(residual_energy))
        outside = float(np.sum(residual_energy[~mask]))
        result["outside_allowed_mask_energy_ratio"] = outside / max(total, float(epsilon))
        result["allowed_channel_count"] = int(np.sum(mask))
    if phase_id is not None:
        phase = np.asarray(phase_id)
        if phase.ndim == 1 and target.shape[0] == 1:
            phase = phase[None, :]
        if phase.shape != target.shape[:2] or not np.issubdtype(phase.dtype, np.integer):
            raise ValueError("phase_id must be integer [trial,time]")
        result["per_phase_residual_energy_ratio"] = {}
        for value in sorted(np.unique(phase).astype(int).tolist()):
            selected = phase == value
            phase_residual = float(np.sum(np.square(residual_value[selected])))
            phase_target = float(np.sum(np.square(target[selected])))
            result["per_phase_residual_energy_ratio"][str(value)] = phase_residual / max(
                phase_target,
                float(epsilon),
            )
    return result


def joint_load_metrics(
    joint_torque: np.ndarray,
    joint_angular_velocity: np.ndarray,
    *,
    joint_names: Sequence[str],
    sampling_rate_hz: float,
) -> dict[str, Any]:
    torque = _trial_time_channel(joint_torque, "joint_torque", nonnegative=False)
    velocity = _trial_time_channel(
        joint_angular_velocity,
        "joint_angular_velocity",
        nonnegative=False,
    )
    if torque.shape != velocity.shape:
        raise ValueError("joint torque and angular velocity shapes differ")
    names = _names(joint_names, torque.shape[2], "joint_names")
    fs = _positive_scalar(sampling_rate_hz, "sampling_rate_hz")
    power = torque * velocity
    return {
        name: {
            "peak_abs_torque": _summary(np.max(np.abs(torque[:, :, index]), axis=1)),
            "peak_abs_angular_velocity": _summary(np.max(np.abs(velocity[:, :, index]), axis=1)),
            "peak_abs_power": _summary(np.max(np.abs(power[:, :, index]), axis=1)),
            "integrated_abs_power": _summary(np.trapezoid(np.abs(power[:, :, index]), dx=1.0 / fs, axis=1)),
        }
        for index, name in enumerate(names)
    }


def kinetic_chain_metrics(
    joint_angular_velocity: np.ndarray,
    *,
    joint_names: Sequence[str],
    ordered_segments: Sequence[Mapping[str, Any]],
    impact_frames: np.ndarray,
    sampling_rate_hz: float,
    post_impact_horizon_s: float = 0.3,
) -> dict[str, Any]:
    """Check proximal-to-distal peak order and post-impact deceleration."""

    velocity = _trial_time_channel(
        joint_angular_velocity,
        "joint_angular_velocity",
        nonnegative=False,
    )
    names = _names(joint_names, velocity.shape[2], "joint_names")
    impacts = _impact_frames(impact_frames, velocity)
    fs = _positive_scalar(sampling_rate_hz, "sampling_rate_hz")
    if not ordered_segments:
        raise ValueError("ordered_segments must define a proximal-to-distal chain")
    segment_names: list[str] = []
    segment_signals: list[np.ndarray] = []
    for entry in ordered_segments:
        segment = str(entry.get("name", ""))
        joints = [str(name) for name in entry.get("joints", ())]
        if not segment or not joints or any(name not in names for name in joints):
            raise ValueError(f"invalid kinetic-chain segment: {entry!r}")
        indices = [names.index(name) for name in joints]
        segment_names.append(segment)
        segment_signals.append(np.mean(np.abs(velocity[:, :, indices]), axis=2))
    stacked = np.stack(segment_signals, axis=2)
    peak_index = np.argmax(stacked, axis=1)
    peak_time = (peak_index - impacts[:, None]) / fs
    order_ok = np.all(np.diff(peak_time, axis=1) >= 0.0, axis=1)
    horizon = round(float(post_impact_horizon_s) * fs)
    if horizon <= 0:
        raise ValueError("post_impact_horizon_s is too short")
    deceleration: dict[str, Any] = {}
    for segment, signal in zip(segment_names, segment_signals, strict=True):
        trial_values = []
        for trial, impact in enumerate(impacts.tolist()):
            end = impact + horizon
            if end >= signal.shape[1]:
                raise ValueError(f"trial {trial} lacks the requested post-impact horizon")
            window = signal[trial, impact : end + 1]
            peak_offset = int(np.argmax(window))
            peak = float(window[peak_offset])
            duration = max((len(window) - 1 - peak_offset) / fs, 1.0 / fs)
            trial_values.append((peak - float(window[-1])) / duration)
        deceleration[segment] = _summary(np.asarray(trial_values))
    return {
        "ordered_segments": segment_names,
        "peak_time_from_impact_s": {
            segment: _summary(peak_time[:, index]) for index, segment in enumerate(segment_names)
        },
        "proximal_to_distal_order_agreement": float(np.mean(order_ok)),
        "post_impact_deceleration_per_s": deceleration,
    }


def build_physiology_report(
    arrays: Mapping[str, np.ndarray],
    *,
    co_contraction_pairs: Sequence[Sequence[str]] = (),
    ordered_segments: Sequence[Mapping[str, Any]] = (),
    allowed_residual_mask: np.ndarray | None = None,
    anatomical_taxonomy: AnatomicalTaxonomy | None = None,
    fascicle_continuity_graph: FascicleContinuityGraph | None = None,
) -> dict[str, Any]:
    required = {
        "muscle_excitation",
        "muscle_activation",
        "actuator_names",
        "sampling_rate_hz",
        "impact_frame",
    }
    if missing := sorted(required - set(arrays)):
        raise ValueError(f"physiology input is missing fields: {missing}")
    names = _string_names(arrays["actuator_names"], "actuator_names")
    signal_contract = validate_physiology_signal_contract(arrays)
    fs = _positive_scalar(np.asarray(arrays["sampling_rate_hz"]).reshape(-1)[0], "sampling_rate_hz")
    activation = _trial_time_channel(
        validate_unit_muscle_activation(arrays["muscle_activation"]),
        "muscle_activation",
        nonnegative=True,
    )
    excitation = _trial_time_channel(
        validate_unit_muscle_excitation(arrays["muscle_excitation"]),
        "muscle_excitation",
        nonnegative=True,
    )
    if activation.shape != excitation.shape or activation.shape[2] != len(names):
        raise ValueError("activation/excitation dimensions do not match actuator_names")
    report: dict[str, Any] = {
        "schema_version": PHYSIOLOGY_METRICS_SCHEMA_VERSION,
        "num_trials": int(activation.shape[0]),
        "muscle_count": int(activation.shape[2]),
        "physical_signal_contract": signal_contract,
        "physical_signal_semantics_fingerprint": signal_contract["signal_semantics_fingerprint"],
        "excitation_timing": muscle_timing_metrics(
            excitation,
            muscle_names=names,
            impact_frames=arrays["impact_frame"],
            sampling_rate_hz=fs,
        ),
        "activation_timing": muscle_timing_metrics(
            activation,
            muscle_names=names,
            impact_frames=arrays["impact_frame"],
            sampling_rate_hz=fs,
        ),
        "co_contraction": cocontraction_index(
            activation,
            channel_names=names,
            pairs=co_contraction_pairs,
        ),
    }
    if "phase_id" in arrays:
        report["phase_activation"] = phase_signal_summary(
            activation,
            arrays["phase_id"],
            channel_names=names,
        )
    if anatomical_taxonomy is not None:
        report["intra_muscle_diagnostics"] = intra_muscle_diagnostics(
            activation,
            excitation,
            taxonomy=anatomical_taxonomy,
            physical_signal_contract=signal_contract,
            phase_id=arrays.get("phase_id"),
        )
    if fascicle_continuity_graph is not None:
        if anatomical_taxonomy is None:
            raise ValueError("fascicle continuity evaluation requires an anatomical taxonomy")
        report["fascicle_continuity"] = fascicle_continuity_diagnostics(
            activation,
            excitation,
            taxonomy=anatomical_taxonomy,
            graph=fascicle_continuity_graph,
            physical_signal_contract=signal_contract,
            phase_id=arrays.get("phase_id"),
        )
    if "synergy_reconstruction" in arrays:
        residual_report = synergy_residual_metrics(
            excitation,
            arrays["synergy_reconstruction"],
            residual=arrays.get("synergy_residual"),
            allowed_residual_mask=allowed_residual_mask,
            phase_id=arrays.get("phase_id"),
        )
        if anatomical_taxonomy is not None:
            # Both halves of the muscle stack are about to be reported side by
            # side against the same channel axis, so require them to agree on
            # what that axis means instead of assuming it.
            residual_report["taxonomy_binding"] = _bind_synergy_signals_to_taxonomy(
                arrays,
                taxonomy=anatomical_taxonomy,
                muscle_names=names,
            )
        report["synergy_residual"] = residual_report
    joint_fields = {"joint_torque", "joint_angular_velocity", "joint_names"}
    if joint_fields <= set(arrays):
        joint_names = _string_names(arrays["joint_names"], "joint_names")
        report["joint_load"] = joint_load_metrics(
            arrays["joint_torque"],
            arrays["joint_angular_velocity"],
            joint_names=joint_names,
            sampling_rate_hz=fs,
        )
        if ordered_segments:
            report["kinetic_chain"] = kinetic_chain_metrics(
                arrays["joint_angular_velocity"],
                joint_names=joint_names,
                ordered_segments=ordered_segments,
                impact_frames=arrays["impact_frame"],
                sampling_rate_hz=fs,
            )
    return report


def intra_muscle_diagnostics(
    activation: np.ndarray,
    excitation: np.ndarray,
    *,
    taxonomy: AnatomicalTaxonomy,
    physical_signal_contract: Mapping[str, Any],
    phase_id: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report hard/soft group dispersion without changing any reward.

    The taxonomy is bound to the exact ordered signal channels here.  Full
    compiled-model validation remains an environment/export preflight concern;
    this offline report verifies the persisted actuator names, ids, activation
    addresses, scalar-state counts, unit ctrlranges, package version and
    taxonomy fingerprint.
    """

    activation_array = _trial_time_channel(
        activation,
        "muscle_activation",
        nonnegative=True,
    )
    excitation_array = _trial_time_channel(
        excitation,
        "muscle_excitation",
        nonnegative=True,
    )
    if activation_array.shape != excitation_array.shape:
        raise ValueError("intra-muscle activation/excitation shapes differ")
    binding = _bind_taxonomy_to_physical_signals(
        taxonomy,
        physical_signal_contract,
    )
    phases = None
    if phase_id is not None:
        phases = np.asarray(phase_id)
        if phases.ndim == 1 and activation_array.shape[0] == 1:
            phases = phases[None, :]
        if phases.shape != activation_array.shape[:2]:
            raise ValueError("intra-muscle phase_id must match [trial,time]")
        if not np.issubdtype(phases.dtype, np.integer):
            raise ValueError("intra-muscle phase_id must contain integer labels")

    relationships: dict[str, Any] = {}
    measured_group_counts: dict[str, int] = {}
    for collection in ("hard_line_groups", "soft_compartment_groups"):
        spec = build_intra_muscle_spec(
            taxonomy,
            collection=collection,
            training_enabled_only=False,
        )
        measured_group_counts[collection] = len(spec.group_ids)
        relationships[collection] = {
            "relationship": spec.relationship,
            "group_ids": list(spec.group_ids),
            "training_behavior": (
                "verified_hard_groups_are_diagnostics_only_in_this_report"
                if collection == "hard_line_groups"
                else "soft_compartments_are_diagnostics_only_never_hard_equality"
            ),
            "activation": _intra_signal_diagnostics(
                activation_array,
                spec,
                phase_id=phases,
            ),
            "effective_excitation": _intra_signal_diagnostics(
                excitation_array,
                spec,
                phase_id=phases,
            ),
        }
    total_measured = sum(measured_group_counts.values())
    return {
        "schema_version": INTRA_MUSCLE_DIAGNOSTICS_SCHEMA_VERSION,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "default_behavior": "diagnostics_only_no_reward",
        "signal_priority": {
            "primary": "muscle_activation",
            "secondary": "effective_muscle_excitation",
        },
        "offline_taxonomy_binding": binding,
        # Every loss below is zero when no group was measured, which reads exactly
        # like a perfectly consistent model.  Coverage is what tells the two apart,
        # so it is reported at the top level rather than only inside each
        # relationship's aggregate block.
        "coverage": {
            "measured_group_counts": measured_group_counts,
            "total_measured_group_count": total_measured,
            "intra_muscle_measured": total_measured > 0,
            "zero_loss_interpretation": (
                "no_group_measured_zero_loss_is_not_evidence_of_consistency"
                if total_measured == 0
                else "loss_reflects_measured_group_dispersion"
            ),
        },
        # Published so a synergy artifact's muscle_schema_sha256 /
        # ordered_muscle_schema_sha256 can be compared against this report.
        "ordered_muscle_schema_sha256": taxonomy_ordered_muscle_schema_hash(taxonomy),
        "relationships": relationships,
    }


def fascicle_continuity_diagnostics(
    activation: np.ndarray,
    excitation: np.ndarray,
    *,
    taxonomy: AnatomicalTaxonomy,
    graph: FascicleContinuityGraph,
    physical_signal_contract: Mapping[str, Any],
    phase_id: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report local adjacency continuity separately from mean-based IMR."""

    activation_array = _trial_time_channel(
        activation,
        "muscle_activation",
        nonnegative=True,
    )
    excitation_array = _trial_time_channel(
        excitation,
        "muscle_excitation",
        nonnegative=True,
    )
    if activation_array.shape != excitation_array.shape:
        raise ValueError("fascicle continuity activation/excitation shapes differ")
    offline_binding = _bind_taxonomy_to_physical_signals(
        taxonomy,
        physical_signal_contract,
    )
    spec = build_fascicle_continuity_spec(graph, taxonomy)
    phases = None
    if phase_id is not None:
        phases = np.asarray(phase_id)
        if phases.ndim == 1 and activation_array.shape[0] == 1:
            phases = phases[None, :]
        if phases.shape != activation_array.shape[:2] or not np.issubdtype(
            phases.dtype,
            np.integer,
        ):
            raise ValueError("fascicle continuity phase_id must be integer [trial,time]")
    measured_edges = int(np.sum(np.asarray(spec.edge_mask, dtype=np.int64)))
    measured_chains = len(spec.chain_ids)
    training_promotion = None
    if isinstance(graph.generation, Mapping):
        raw_promotion = graph.generation.get("training_promotion")
        if isinstance(raw_promotion, Mapping):
            training_promotion = copy.deepcopy(dict(raw_promotion))
    return {
        "schema_version": FASCICLE_CONTINUITY_DIAGNOSTICS_SCHEMA_VERSION,
        "graph_id": graph.graph_id,
        "graph_fingerprint": graph.graph_fingerprint,
        "taxonomy_binding": copy.deepcopy(graph.taxonomy_binding),
        "offline_taxonomy_binding": offline_binding,
        "default_behavior": graph.default_behavior,
        "training_promotion": training_promotion,
        "signal_priority": {
            "primary": "muscle_activation",
            "secondary": "effective_muscle_excitation",
        },
        "coverage": {
            "declared_chain_count": len(graph.chains),
            "measured_chain_count": measured_chains,
            "training_enabled_chain_count": graph.training_enabled_chain_count,
            "measured_edge_count": measured_edges,
            "continuity_measured": measured_chains > 0 and measured_edges > 0,
            "zero_loss_interpretation": (
                "loss_reflects_measured_adjacency_dispersion"
                if measured_edges > 0
                else "no_edge_measured_zero_loss_is_not_evidence_of_continuity"
            ),
        },
        "activation": _continuity_signal_diagnostics(
            activation_array,
            spec,
            phase_id=phases,
        ),
        "excitation": _continuity_signal_diagnostics(
            excitation_array,
            spec,
            phase_id=phases,
        ),
    }


def _bind_synergy_signals_to_taxonomy(
    arrays: Mapping[str, Any],
    *,
    taxonomy: AnatomicalTaxonomy,
    muscle_names: Sequence[str],
) -> dict[str, Any]:
    """Require the synergy channel axis to be the taxonomy's channel axis.

    The reconstruction itself carries no names, so the ordered actuator names are
    verified against the taxonomy and any ordered-muscle hash the input declares
    is required to match.  When the input declares none, that is recorded rather
    than glossed over: the channel order was checked, the artifact lineage was not.
    """

    record = assert_taxonomy_matches_ordered_muscles(
        taxonomy,
        muscle_names,
        context="synergy_reconstruction_channels",
    )
    declared = {
        field: _identity_scalar(np.asarray(arrays[field]), field)
        for field in SYNERGY_SCHEMA_HASH_FIELDS
        if field in arrays
    }
    expected = record["ordered_muscle_schema_sha256"]
    for field, value in sorted(declared.items()):
        if value != expected:
            raise ValueError(
                f"synergy_reconstruction {field}={value!r} does not match the "
                f"taxonomy ordered muscle schema hash {expected!r}"
            )
    record["verified_synergy_hash_fields"] = sorted(declared)
    if not declared:
        record["unverified_synergy_lineage"] = (
            "input declares no synergy ordered-muscle hash; only the channel order was bound to the taxonomy"
        )
    return record


def _bind_taxonomy_to_physical_signals(
    taxonomy: AnatomicalTaxonomy,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    channel = contract.get("muscle_channel_contract")
    if not isinstance(channel, Mapping):
        raise ValueError("intra-muscle diagnostics require the persisted muscle channel contract")
    names = tuple(str(value) for value in channel.get("actuator_names", ()))
    if names != taxonomy.actuator_names:
        raise ValueError("anatomical taxonomy actuator names/order differ from physiology signals")
    if actuator_schema_hash(names) != taxonomy.model_binding["actuator_schema_hash"]:
        raise ValueError("anatomical taxonomy actuator schema hash differs from physiology signals")
    installed_version = importlib.metadata.version(taxonomy.model_binding["package"])
    if installed_version != taxonomy.model_binding["version"]:
        raise ValueError("installed model package version differs from anatomical taxonomy")
    expected_vectors = {
        "actuator_ids": [row["actuator_id"] for row in taxonomy.ordered_actuators],
        "actuator_actnum": [row["actnum"] for row in taxonomy.ordered_actuators],
        "actuator_actadr": [row["actadr"] for row in taxonomy.ordered_actuators],
    }
    for field, expected in expected_vectors.items():
        actual = [int(value) for value in channel.get(field, ())]
        if actual != expected:
            raise ValueError(f"anatomical taxonomy {field} differ from physiology signals")
    ctrlrange = np.asarray(
        [row["ctrlrange"] for row in taxonomy.ordered_actuators],
        dtype=np.float64,
    )
    validate_unit_muscle_ctrlrange(names, ctrlrange)
    return {
        "verification_scope": ("exact_ordered_persisted_channel_contract_and_installed_asset_version"),
        "actuator_count": len(names),
        "actuator_schema_hash": taxonomy.model_binding["actuator_schema_hash"],
        "model_package": taxonomy.model_binding["package"],
        "model_package_version": installed_version,
        "taxonomy_runtime_model_hash": taxonomy.model_binding["runtime_model_hash"],
        "compiled_model_hash_revalidation": ("required_at_environment_export_preflight_not_claimed_by_offline_npz"),
    }


def _intra_signal_diagnostics(
    signal: np.ndarray,
    spec: IntraMuscleSpec,
    *,
    phase_id: np.ndarray | None,
) -> dict[str, Any]:
    flat = np.asarray(signal, dtype=np.float32).reshape(-1, signal.shape[-1])
    aggregate = _intra_flat_diagnostics(flat, spec)
    per_phase: dict[str, Any] = {}
    if phase_id is not None:
        flat_phase = np.asarray(phase_id).reshape(-1)
        for phase in sorted(np.unique(flat_phase).tolist()):
            selected = flat[flat_phase == phase]
            if selected.size:
                per_phase[str(int(phase))] = _intra_flat_diagnostics(
                    selected,
                    spec,
                )
    return {
        "aggregate": aggregate,
        "per_phase": per_phase,
    }


def _intra_flat_diagnostics(
    flat_signal: np.ndarray,
    spec: IntraMuscleSpec,
) -> dict[str, Any]:
    values = jnp.asarray(flat_signal, dtype=jnp.float32)
    exact = jax.vmap(lambda row: exact_exo_imr(row, spec))(values)
    robust = jax.vmap(lambda row: robust_intra_muscle_consistency(row, spec))(values)
    exact_group_loss = np.asarray(exact.group_loss)
    robust_group_loss = np.asarray(robust.group_loss)
    robust_group_violation = np.asarray(robust.group_violation_fraction)
    group_activity = np.asarray(robust.group_activity_gate)
    group_indices = np.asarray(spec.group_indices, dtype=np.int64)
    member_mask = np.asarray(spec.member_mask, dtype=bool)
    member_weights = np.asarray(spec.member_weights, dtype=np.float64)
    grouped_values = np.take(
        np.asarray(flat_signal, dtype=np.float64),
        group_indices,
        axis=1,
    )
    effective_weights = member_weights * member_mask
    weight_sum = np.maximum(np.sum(effective_weights, axis=1), 1e-12)
    group_mean = (
        np.sum(
            grouped_values * effective_weights[None, :, :],
            axis=2,
        )
        / weight_sum[None, :]
    )
    group_deviation = np.abs(grouped_values - group_mean[:, :, None])
    per_group = {
        group_id: {
            "exact_exo_group_loss_mean": float(np.mean(exact_group_loss[:, index])),
            "robust_group_loss_mean": float(np.mean(robust_group_loss[:, index])),
            "robust_violation_fraction_mean": float(np.mean(robust_group_violation[:, index])),
            "activity_gate_mean": float(np.mean(group_activity[:, index])),
            "mean_abs_deviation": float(np.mean(group_deviation[:, index, member_mask[index]])),
            "rms_deviation": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            group_deviation[
                                :,
                                index,
                                member_mask[index],
                            ]
                        )
                    )
                )
            ),
            "p95_abs_deviation": float(
                np.percentile(
                    group_deviation[:, index, member_mask[index]],
                    95.0,
                )
            ),
            "max_abs_deviation": float(np.max(group_deviation[:, index, member_mask[index]])),
        }
        for index, group_id in enumerate(spec.group_ids)
    }
    return {
        "sample_count": int(flat_signal.shape[0]),
        "group_count": len(spec.group_ids),
        "exact_exo": {
            "definition": "hard_deadband_0.1_unnormalized_sum",
            "loss_mean": float(np.mean(np.asarray(exact.loss))),
            "violation_fraction_mean": float(np.mean(np.asarray(exact.violation_fraction))),
            "mean_abs_deviation": float(np.mean(np.asarray(exact.mean_abs_deviation))),
            "max_abs_deviation": float(np.max(np.asarray(exact.max_abs_deviation))),
        },
        "robust_project": {
            "definition": ("deadband_excess_huber_activity_gate_group_normalized"),
            "loss_mean": float(np.mean(np.asarray(robust.loss))),
            "active_group_fraction_mean": float(np.mean(np.asarray(robust.active_group_fraction))),
            "violation_fraction_mean": float(np.mean(np.asarray(robust.violation_fraction))),
            "mean_abs_deviation": float(np.mean(np.asarray(robust.mean_abs_deviation))),
            "max_abs_deviation": float(np.max(np.asarray(robust.max_abs_deviation))),
        },
        "per_group": per_group,
    }


def _continuity_signal_diagnostics(
    signal: np.ndarray,
    spec: FascicleContinuitySpec,
    *,
    phase_id: np.ndarray | None,
) -> dict[str, Any]:
    flat = np.asarray(signal, dtype=np.float32).reshape(-1, signal.shape[-1])
    complete = _continuity_flat_diagnostics(flat, spec)
    per_phase: dict[str, Any] = {}
    if phase_id is not None:
        flat_phase = np.asarray(phase_id).reshape(-1)
        for phase in sorted(np.unique(flat_phase).tolist()):
            selected = flat[flat_phase == phase]
            if selected.size:
                per_phase[str(int(phase))] = _continuity_flat_diagnostics(selected, spec)
    return {
        "aggregate": complete["aggregate"],
        "per_chain": complete["per_chain"],
        "per_phase": per_phase,
    }


def _continuity_flat_diagnostics(
    flat_signal: np.ndarray,
    spec: FascicleContinuitySpec,
) -> dict[str, Any]:
    values = jnp.asarray(flat_signal, dtype=jnp.float32)
    metrics = jax.vmap(lambda row: robust_fascicle_continuity(row, spec))(values)
    chain_count = len(spec.chain_ids)
    if chain_count == 0:
        zero = _zero_quantile_summary()
        return {
            "aggregate": {
                "sample_count": int(flat_signal.shape[0]),
                "chain_count": 0,
                "edge_count": 0,
                "loss": zero,
                "active_chain_fraction": zero,
                "violation_fraction": zero,
                "mean_abs_edge_difference": zero,
                "max_abs_edge_difference": zero,
                "edge_absolute_difference": zero,
                "chain_loss": zero,
            },
            "per_chain": {},
        }

    edge_indices = np.asarray(spec.edge_indices, dtype=np.int64)
    edge_mask = np.asarray(spec.edge_mask, dtype=bool)
    edge_values = np.take(np.asarray(flat_signal, dtype=np.float64), edge_indices, axis=1)
    edge_difference = np.abs(edge_values[..., 0] - edge_values[..., 1])
    per_chain = {}
    chain_loss = np.asarray(metrics.chain_loss)
    chain_violation = np.asarray(metrics.chain_violation_fraction)
    chain_mean = np.asarray(metrics.chain_mean_activation)
    chain_gate = np.asarray(metrics.chain_activity_gate)
    for index, chain_id in enumerate(spec.chain_ids):
        valid_difference = edge_difference[:, index, edge_mask[index]]
        per_chain[chain_id] = {
            "edge_count": int(np.sum(edge_mask[index])),
            "loss": _quantile_summary(chain_loss[:, index]),
            "violation_fraction": _quantile_summary(chain_violation[:, index]),
            "mean_activation": _quantile_summary(chain_mean[:, index]),
            "activity_gate": _quantile_summary(chain_gate[:, index]),
            "edge_absolute_difference": _quantile_summary(valid_difference),
        }
    return {
        "aggregate": {
            "sample_count": int(flat_signal.shape[0]),
            "chain_count": chain_count,
            "edge_count": int(np.sum(edge_mask)),
            "loss": _quantile_summary(np.asarray(metrics.loss)),
            "active_chain_fraction": _quantile_summary(np.asarray(metrics.active_chain_fraction)),
            "violation_fraction": _quantile_summary(np.asarray(metrics.violation_fraction)),
            "mean_abs_edge_difference": _quantile_summary(np.asarray(metrics.mean_abs_edge_difference)),
            "max_abs_edge_difference": _quantile_summary(np.asarray(metrics.max_abs_edge_difference)),
            "edge_absolute_difference": _quantile_summary(edge_difference[:, edge_mask]),
            "chain_loss": _quantile_summary(chain_loss),
        },
        "per_chain": per_chain,
    }


def validate_physiology_signal_contract(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Fail closed unless physiology uses canonical physical muscle signals.

    Every reported actuator is used by the timing and co-contraction summaries,
    so the public physiology report requires a scalar MuJoCo activation state
    for every name-aligned channel.  Partial masks must be filtered into a new,
    explicitly named NPZ before evaluation instead of silently supporting a
    whole-body physiology claim.
    """

    required = {
        "muscle_excitation",
        "muscle_activation",
        "teacher_ctrl_physical",
        "actuator_names",
        "actuator_ctrlrange",
        "physical_signal_schema_version",
        "muscle_excitation_source",
        "muscle_excitation_semantics",
        "muscle_excitation_transform",
        "muscle_excitation_formula",
        "muscle_excitation_roundoff_policy",
        "muscle_activation_source",
        "muscle_activation_semantics",
        "muscle_activation_roundoff_policy",
        "activation_valid_mask",
        "muscle_channel_contract_schema_version",
        "actuator_ids",
        "actuator_dyntype",
        "actuator_actnum",
        "actuator_actadr",
        "model_na",
    }
    if missing := sorted(required - set(arrays)):
        raise ValueError(f"physiology physical signal contract is missing fields: {missing}")

    names = _string_names(arrays["actuator_names"], "actuator_names")
    excitation = _trial_time_channel(
        validate_unit_muscle_excitation(arrays["muscle_excitation"]),
        "muscle_excitation",
        nonnegative=True,
    )
    activation = _trial_time_channel(
        validate_unit_muscle_activation(arrays["muscle_activation"]),
        "muscle_activation",
        nonnegative=True,
    )
    if excitation.shape != activation.shape or excitation.shape[2] != len(names):
        raise ValueError("physical excitation/activation dimensions do not match actuator_names")
    raw_ctrl = _trial_time_channel(
        arrays["teacher_ctrl_physical"],
        "teacher_ctrl_physical",
        nonnegative=False,
    )
    if raw_ctrl.shape != excitation.shape:
        raise ValueError("raw physical ctrl dimensions do not match muscle excitation")
    ctrlrange = validate_unit_muscle_ctrlrange(
        names,
        arrays["actuator_ctrlrange"],
    )
    del ctrlrange
    channel_contract = validate_muscle_channel_contract(
        {
            "schema_version": _identity_scalar(
                arrays["muscle_channel_contract_schema_version"],
                "muscle_channel_contract_schema_version",
            ),
            "actuator_names": names,
            "actuator_ids": np.asarray(arrays["actuator_ids"]).tolist(),
            "actuator_dyntype": [str(value) for value in np.asarray(arrays["actuator_dyntype"]).tolist()],
            "actuator_actnum": np.asarray(arrays["actuator_actnum"]).tolist(),
            "actuator_actadr": np.asarray(arrays["actuator_actadr"]).tolist(),
            "model_na": int(np.asarray(arrays["model_na"]).item()),
        },
        expected_names=names,
    )
    recomputed_excitation = physical_ctrl_to_effective_muscle_excitation(
        raw_ctrl,
        channel_contract=channel_contract,
    )
    if not np.allclose(
        excitation,
        recomputed_excitation,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("physiology muscle_excitation differs from clip(raw data.ctrl,0,1)")

    schema = _identity_scalar(
        arrays["physical_signal_schema_version"],
        "physical_signal_schema_version",
    )
    excitation_source = _identity_scalar(
        arrays["muscle_excitation_source"],
        "muscle_excitation_source",
    )
    excitation_semantics = _identity_scalar(
        arrays["muscle_excitation_semantics"],
        "muscle_excitation_semantics",
    )
    excitation_transform = _identity_scalar(
        arrays["muscle_excitation_transform"],
        "muscle_excitation_transform",
    )
    excitation_formula = _identity_scalar(
        arrays["muscle_excitation_formula"],
        "muscle_excitation_formula",
    )
    excitation_roundoff = _identity_scalar(
        arrays["muscle_excitation_roundoff_policy"],
        "muscle_excitation_roundoff_policy",
    )
    activation_source = _identity_scalar(
        arrays["muscle_activation_source"],
        "muscle_activation_source",
    )
    activation_semantics = _identity_scalar(
        arrays["muscle_activation_semantics"],
        "muscle_activation_semantics",
    )
    activation_roundoff = _identity_scalar(
        arrays["muscle_activation_roundoff_policy"],
        "muscle_activation_roundoff_policy",
    )

    semantics = {
        "schema_version": schema,
        "muscle_excitation": {
            "source": excitation_source,
            "semantics": excitation_semantics,
            "transform": excitation_transform,
            "formula": excitation_formula,
            "roundoff_policy": excitation_roundoff,
            "nonnegative": True,
        },
        "muscle_activation": {
            "source": activation_source,
            "semantics": activation_semantics,
            "nonnegative": True,
            "upper_bound": 1.0,
            "roundoff_policy": activation_roundoff,
        },
    }
    # Reuse the shared distillation/NMF contract rather than maintaining a
    # physiology-specific interpretation of the same persisted signals.
    validate_physical_signal_semantics(semantics)
    canonical = physical_signal_metadata()
    for signal_name, keys in {
        "muscle_excitation": (
            "source",
            "semantics",
            "transform",
            "formula",
            "roundoff_policy",
            "nonnegative",
        ),
        "muscle_activation": (
            "source",
            "semantics",
            "nonnegative",
            "upper_bound",
            "roundoff_policy",
        ),
    }.items():
        if any(semantics[signal_name][key] != canonical[signal_name][key] for key in keys):
            raise ValueError(f"physiology {signal_name} semantics are not canonical")

    valid = validate_activation_valid_mask(
        arrays["activation_valid_mask"],
        expected_width=len(names),
    )
    if not np.all(valid):
        invalid = [names[index] for index in np.flatnonzero(~valid).tolist()]
        raise ValueError(f"physiology input includes actuators without a scalar MuJoCo activation state: {invalid}")

    contract: dict[str, Any] = {
        "schema_version": PHYSIOLOGY_SIGNAL_CONTRACT_SCHEMA_VERSION,
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "muscle_excitation": semantics["muscle_excitation"],
        "muscle_activation": semantics["muscle_activation"],
        "actuator_names": names,
        "muscle_channel_contract": channel_contract.to_metadata(),
        "activation_valid_mask": valid.tolist(),
        "activation_channel_policy": "require_all_reported_actuators_activation_valid",
        "unit_interval_verified": True,
    }
    contract["signal_semantics_fingerprint"] = _canonical_json_sha256(contract)
    return contract


def validate_physiology_lineage(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_policy_checkpoint_fingerprint: str,
    expected_policy_promotion_fingerprint: str,
    expected_formal_synergy_basis_fingerprint: str,
    expected_event_reference_fingerprint: str,
    expected_session_uid: str,
    expected_policy_decoder_type: str,
) -> dict[str, Any]:
    """Bind simulation signals to one selected policy, event set and session."""

    required = {
        "policy_decoder_type",
        "policy_checkpoint_fingerprint",
        "policy_promotion_fingerprint",
        "formal_synergy_basis_fingerprint",
        "analysis_synergy_basis_fingerprint",
        "event_reference_fingerprint",
        "session_uid",
    }
    if missing := sorted(required - set(arrays)):
        raise ValueError(f"physiology lineage is missing fields: {missing}")
    signal_contract = validate_physiology_signal_contract(arrays)
    policy = validate_simulation_policy_evidence(
        arrays,
        expected_policy_checkpoint_fingerprint=expected_policy_checkpoint_fingerprint,
        expected_policy_promotion_fingerprint=expected_policy_promotion_fingerprint,
        expected_formal_synergy_basis_fingerprint=expected_formal_synergy_basis_fingerprint,
    )
    decoder = _identity_scalar(arrays["policy_decoder_type"], "policy_decoder_type")
    if decoder != str(expected_policy_decoder_type):
        raise ValueError("physiology decoder identity differs from selected policy")
    event_reference = _sha256_scalar(arrays["event_reference_fingerprint"], "event_reference_fingerprint")
    if event_reference != _require_sha256_text(
        expected_event_reference_fingerprint,
        "expected_event_reference_fingerprint",
    ):
        raise ValueError("physiology event reference differs from selected evaluation set")
    session_uid = _single_session_identity(arrays["session_uid"], "session_uid")
    expected_session = str(expected_session_uid).strip()
    if not expected_session or session_uid != expected_session:
        raise ValueError("physiology session identity differs from selected collection")
    lineage = {
        "schema_version": PHYSIOLOGY_LINEAGE_SCHEMA_VERSION,
        "binding_verified": 1.0,
        "policy_evidence": policy,
        "event_reference_fingerprint": event_reference,
        "session_uid": session_uid,
        "policy_decoder_type": decoder,
        "physical_signal_contract": signal_contract,
        "physical_signal_semantics_fingerprint": signal_contract["signal_semantics_fingerprint"],
    }
    lineage["lineage_fingerprint"] = _canonical_json_sha256(lineage)
    return lineage


def _resolve_expected_lineage_bindings(
    *,
    policy_evidence_json: str | Path | None,
    signal_identity_json: str | Path | None,
    expected_policy_checkpoint_fingerprint: str | None,
    expected_policy_promotion_fingerprint: str | None,
    expected_formal_synergy_basis_fingerprint: str | None,
    expected_event_reference_fingerprint: str | None,
    expected_session_uid: str | None,
    expected_policy_decoder_type: str | None,
) -> dict[str, str]:
    explicit = {
        "policy_checkpoint_fingerprint": expected_policy_checkpoint_fingerprint,
        "policy_promotion_fingerprint": expected_policy_promotion_fingerprint,
        "formal_synergy_basis_fingerprint": expected_formal_synergy_basis_fingerprint,
        "event_reference_fingerprint": expected_event_reference_fingerprint,
        "policy_decoder_type": expected_policy_decoder_type,
    }
    if policy_evidence_json:
        from musclemimic.evaluation.stage3_signal_export import (
            load_paired_policy_evidence,
        )

        evidence = load_paired_policy_evidence(policy_evidence_json)
        sealed = {
            "policy_checkpoint_fingerprint": evidence.policy_checkpoint_fingerprint,
            "policy_promotion_fingerprint": evidence.policy_promotion_fingerprint,
            "formal_synergy_basis_fingerprint": evidence.formal_synergy_basis_fingerprint,
            "event_reference_fingerprint": evidence.event_reference_fingerprint,
            "policy_decoder_type": evidence.decoder_type,
        }
        for key, value in explicit.items():
            if value is not None and str(value) != sealed[key]:
                raise ValueError(f"explicit physiology lineage differs from paired evidence: {key}")
        resolved = sealed
    else:
        missing = [key for key, value in explicit.items() if not value]
        if missing:
            raise SystemExit(
                "missing policy evidence: supply --policy-evidence-json or all explicit expected lineage fields "
                f"({', '.join(missing)})"
            )
        resolved = {key: str(value) for key, value in explicit.items()}

    session = str(expected_session_uid or "").strip()
    if signal_identity_json:
        from musclemimic.evaluation.stage3_signal_export import (
            load_trial_identity_manifest,
        )

        identity = load_trial_identity_manifest(signal_identity_json)
        identity_sessions = {trial.session_uid for trial in identity.trials_by_feed.values()}
        if len(identity_sessions) != 1:
            raise ValueError("Stage-3 identity manifest must bind one held-out session")
        sealed_session = next(iter(identity_sessions))
        if session and session != sealed_session:
            raise ValueError("explicit physiology session differs from the signal identity manifest")
        session = sealed_session
    if not session:
        raise SystemExit("missing session evidence: supply --signal-identity-json or --expected-session-uid")
    resolved["session_uid"] = session
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz")
    parser.add_argument("--evaluation-config-json")
    parser.add_argument("--output-json")
    parser.add_argument("--expected-policy-checkpoint-fingerprint")
    parser.add_argument("--expected-policy-promotion-fingerprint")
    parser.add_argument("--expected-formal-synergy-basis-fingerprint")
    parser.add_argument("--expected-event-reference-fingerprint")
    parser.add_argument("--expected-session-uid")
    parser.add_argument("--expected-policy-decoder-type")
    parser.add_argument(
        "--policy-evidence-json",
        help="sealed Stage-3 paired comparison used to derive selected policy/basis/event bindings",
    )
    parser.add_argument(
        "--signal-identity-json",
        help="held-out trial identity manifest used to derive the externally sealed session identity",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": PHYSIOLOGY_REPORT_SCHEMA_VERSION,
                    "required_fields": [
                        "muscle_excitation",
                        "muscle_activation",
                        "actuator_names",
                        "physical_signal_schema_version",
                        "muscle_excitation_source",
                        "muscle_excitation_semantics",
                        "muscle_excitation_transform",
                        "muscle_excitation_formula",
                        "muscle_excitation_roundoff_policy",
                        "teacher_ctrl_physical",
                        "actuator_ctrlrange",
                        "muscle_channel_contract_schema_version",
                        "actuator_ids",
                        "actuator_dyntype",
                        "actuator_actnum",
                        "actuator_actadr",
                        "model_na",
                        "muscle_activation_source",
                        "muscle_activation_semantics",
                        "muscle_activation_roundoff_policy",
                        "activation_valid_mask",
                        "sampling_rate_hz",
                        "impact_frame",
                        "policy_decoder_type",
                        "policy_checkpoint_fingerprint",
                        "policy_promotion_fingerprint",
                        "formal_synergy_basis_fingerprint",
                        "analysis_synergy_basis_fingerprint",
                        "event_reference_fingerprint",
                        "session_uid",
                    ],
                    "optional_fields": [
                        "phase_id",
                        "synergy_reconstruction",
                        "synergy_residual",
                        "ordered_muscle_schema_sha256",
                        "muscle_schema_sha256",
                        "joint_torque",
                        "joint_angular_velocity",
                        "joint_names",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    expected = _resolve_expected_lineage_bindings(
        policy_evidence_json=args.policy_evidence_json,
        signal_identity_json=args.signal_identity_json,
        expected_policy_checkpoint_fingerprint=args.expected_policy_checkpoint_fingerprint,
        expected_policy_promotion_fingerprint=args.expected_policy_promotion_fingerprint,
        expected_formal_synergy_basis_fingerprint=args.expected_formal_synergy_basis_fingerprint,
        expected_event_reference_fingerprint=args.expected_event_reference_fingerprint,
        expected_session_uid=args.expected_session_uid,
        expected_policy_decoder_type=args.expected_policy_decoder_type,
    )
    required_arguments = {
        "--input-npz": args.input_npz,
        "--evaluation-config-json": args.evaluation_config_json,
        "--output-json": args.output_json,
    }
    if missing := [flag for flag, value in required_arguments.items() if not value]:
        raise SystemExit(f"missing required arguments outside --dry-run: {', '.join(missing)}")
    input_path = Path(args.input_npz)
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    config = load_json_strict(args.evaluation_config_json)
    if config.get("schema_version") != PHYSIOLOGY_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"physiology config schema_version must be {PHYSIOLOGY_CONFIG_SCHEMA_VERSION!r}")
    allowed_mask = None
    if "allowed_residual_actuators" in config:
        actuator_names = _string_names(arrays["actuator_names"], "actuator_names")
        allowed = {str(name) for name in config["allowed_residual_actuators"]}
        missing = sorted(allowed - set(actuator_names))
        if missing:
            raise ValueError(f"allowed residual actuators are absent: {missing}")
        allowed_mask = np.asarray([name in allowed for name in actuator_names], dtype=bool)
    taxonomy = None
    taxonomy_path = config.get("anatomical_taxonomy_path")
    if taxonomy_path:
        taxonomy = load_anatomical_taxonomy(taxonomy_path)
    continuity_graph = None
    continuity_path = config.get("fascicle_continuity_path")
    if continuity_path:
        if taxonomy is None:
            raise ValueError("fascicle_continuity_path requires anatomical_taxonomy_path")
        continuity_graph = load_fascicle_continuity_graph(
            continuity_path,
            taxonomy=taxonomy,
        )
    report = build_physiology_report(
        arrays,
        co_contraction_pairs=config.get("co_contraction_pairs", ()),
        ordered_segments=config.get("ordered_segments", ()),
        allowed_residual_mask=allowed_mask,
        anatomical_taxonomy=taxonomy,
        fascicle_continuity_graph=continuity_graph,
    )
    report["metrics_schema_version"] = report["schema_version"]
    report["schema_version"] = PHYSIOLOGY_REPORT_SCHEMA_VERSION
    report["lineage"] = validate_physiology_lineage(
        arrays,
        expected_policy_checkpoint_fingerprint=expected["policy_checkpoint_fingerprint"],
        expected_policy_promotion_fingerprint=expected["policy_promotion_fingerprint"],
        expected_formal_synergy_basis_fingerprint=expected["formal_synergy_basis_fingerprint"],
        expected_event_reference_fingerprint=expected["event_reference_fingerprint"],
        expected_session_uid=expected["session_uid"],
        expected_policy_decoder_type=expected["policy_decoder_type"],
    )
    if report["physical_signal_semantics_fingerprint"] != report["lineage"]["physical_signal_semantics_fingerprint"]:
        raise RuntimeError("physiology metrics and lineage signal semantics differ")
    report["input_npz_sha256"] = _sha256(input_path)
    report["evaluation_config_sha256"] = _sha256(Path(args.evaluation_config_json))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(output)
    return 0


def _trial_time_channel(values: np.ndarray, field: str, *, nonnegative: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or min(array.shape) <= 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite [trial,time,channel]")
    if nonnegative and np.min(array) < -1e-8:
        raise ValueError(f"{field} must be non-negative")
    return np.maximum(array, 0.0) if nonnegative else array


def _names(values: Sequence[str], width: int, field: str) -> list[str]:
    names = [str(value) for value in values]
    if len(names) != int(width) or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(f"{field} must be unique and match signal width")
    return names


def _string_names(values: np.ndarray, field: str) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a one-dimensional string array")
    return _names(array.astype(str).tolist(), len(array), field)


def _impact_frames(values: np.ndarray, signal: np.ndarray) -> np.ndarray:
    impact = np.asarray(values)
    if impact.shape != (signal.shape[0],) or not np.issubdtype(impact.dtype, np.integer):
        raise ValueError("impact_frame must contain one integer per trial")
    if np.any(impact < 0) or np.any(impact >= signal.shape[1]):
        raise ValueError("impact_frame lies outside the signal time axis")
    return impact.astype(np.int64)


def _positive_scalar(value: float, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary input must be non-empty and finite")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("quantile summary input must be non-empty and finite")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def _zero_quantile_summary() -> dict[str, Any]:
    return {
        "mean": 0.0,
        "std": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max": 0.0,
        "n": 0,
    }


def _identity_scalar(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be one string scalar")
    result = str(array.reshape(-1)[0]).strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _single_session_identity(value: np.ndarray, field: str) -> str:
    """Accept a scalar or a trial-aligned vector naming one collection session.

    EMG pairing requires ``session_uid [trial]`` while physiology lineage is
    collection-level.  Allowing an exactly constant vector lets one exported
    simulation NPZ satisfy both contracts without weakening the one-session
    physiology claim.
    """

    array = np.asarray(value)
    if array.ndim > 1 or array.size == 0 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a string scalar or one-session trial vector")
    values = [str(item).strip() for item in array.reshape(-1).tolist()]
    if any(not item for item in values) or len(set(values)) != 1:
        raise ValueError(f"{field} must name exactly one non-empty collection session")
    return values[0]


def _require_sha256_text(value: str, field: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return result


def _sha256_scalar(value: np.ndarray, field: str) -> str:
    return _require_sha256_text(_identity_scalar(value, field), field)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
