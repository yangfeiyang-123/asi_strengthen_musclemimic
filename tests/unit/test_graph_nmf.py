"""Fail-closed and numerical contracts for optional graph-regularized NMF."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import continuity_graph_fingerprint
from musclemimic.synergy.basis_artifact import (
    _validate_graph_normalization,
    save_synergy_basis,
)
from musclemimic.synergy.fit import SynergyFitConfig, build_parser
from musclemimic.synergy.frozen_decoder import build_frozen_body_decoder_execution_binding
from musclemimic.synergy.graph_nmf import (
    GraphRegularizationBinding,
    load_verified_graph_regularization,
    validate_graph_regularization_manifest,
)
from musclemimic.synergy.nmf import fit_nmf
from musclemimic.synergy.rank_selection import candidate_basis_fingerprint

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
GRAPH_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"


def _verified_graph_path(tmp_path: Path) -> tuple[Path, str]:
    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    payload["chains"][0]["review_status"] = "verified"
    payload["chains"][0]["training_enabled"] = True
    payload["chains"][0]["provenance"] = [
        {
            "kind": "independent_anatomical_and_baseline_distribution_review",
            "reference": "unit-test evidence fixture",
        }
    ]
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    path = tmp_path / "verified_graph.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, str(payload["graph_fingerprint"])


def _verified_binding(tmp_path: Path, *, coefficient: float = 0.25) -> GraphRegularizationBinding:
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph_path, graph_fingerprint = _verified_graph_path(tmp_path)
    return load_verified_graph_regularization(
        taxonomy_path=TAXONOMY_PATH,
        continuity_path=graph_path,
        expected_taxonomy_fingerprint=taxonomy.fingerprint,
        expected_continuity_fingerprint=graph_fingerprint,
        muscle_names=taxonomy.actuator_names,
        lambda_value=coefficient,
    )


def test_lambda_zero_is_bitwise_identical_to_historical_nmf_path():
    rng = np.random.default_rng(7)
    values = rng.uniform(0.0, 1.0, size=(48, 9))
    historical = fit_nmf(values, rank=3, seed=11, max_iter=80, tol=0.0)
    explicit_zero = fit_nmf(
        values,
        rank=3,
        seed=11,
        max_iter=80,
        tol=0.0,
        graph_lambda=0.0,
    )

    assert np.array_equal(explicit_zero.basis, historical.basis)
    assert np.array_equal(explicit_zero.coefficients, historical.coefficients)
    assert np.array_equal(explicit_zero.reconstruction, historical.reconstruction)
    assert explicit_zero.loss == historical.loss
    assert explicit_zero.objective == historical.loss
    assert explicit_zero.graph_penalty == 0.0


def test_graph_update_matches_the_prescribed_multiplicative_equation():
    values = np.asarray(
        [
            [0.8, 0.7, 0.1],
            [0.4, 0.5, 0.9],
            [0.9, 0.8, 0.2],
            [0.2, 0.3, 0.8],
        ],
        dtype=np.float64,
    )
    adjacency = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
    )
    coefficient = 0.7
    epsilon = 1e-10
    rng = np.random.default_rng(5)
    expected_c = rng.random((values.shape[0], 2)) + 0.05
    expected_b = rng.random((2, values.shape[1])) + 0.05
    expected_c *= (values @ expected_b.T) / (expected_c @ expected_b @ expected_b.T + epsilon)
    degree = np.sum(adjacency, axis=1)
    expected_b *= (expected_c.T @ values + coefficient * expected_b @ adjacency) / (
        expected_c.T @ expected_c @ expected_b + coefficient * expected_b * degree[None, :] + epsilon
    )
    norms = np.linalg.norm(expected_b, axis=1)
    expected_b /= norms[:, None]
    expected_c *= norms[None, :]

    actual = fit_nmf(
        values,
        rank=2,
        seed=5,
        max_iter=1,
        tol=0.0,
        epsilon=epsilon,
        graph_adjacency=adjacency,
        graph_lambda=coefficient,
    )

    assert np.allclose(actual.basis, expected_b.T, rtol=1e-13, atol=1e-13)
    assert np.allclose(actual.coefficients, expected_c, rtol=1e-13, atol=1e-13)


def test_positive_graph_lambda_smooths_adjacent_basis_rows_deterministically():
    rng = np.random.default_rng(4)
    coefficients = rng.uniform(0.05, 1.0, size=(300, 2))
    source_basis = np.asarray(
        [
            [1.0, 0.05],
            [0.9, 0.10],
            [0.05, 1.0],
            [0.10, 0.9],
        ]
    )
    values = coefficients @ source_basis.T + rng.uniform(0.0, 0.03, size=(300, 4))
    adjacency = np.zeros((4, 4), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0
    standard = fit_nmf(values, rank=2, seed=3, max_iter=500)
    graph = fit_nmf(
        values,
        rank=2,
        seed=3,
        max_iter=500,
        graph_adjacency=adjacency,
        graph_lambda=10.0,
    )
    repeated = fit_nmf(
        values,
        rank=2,
        seed=3,
        max_iter=500,
        graph_adjacency=adjacency,
        graph_lambda=10.0,
    )
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    standard_penalty = float(np.trace(standard.basis.T @ laplacian @ standard.basis))

    assert graph.graph_penalty < standard_penalty
    assert graph.objective > 0.0
    assert np.array_equal(graph.basis, repeated.basis)
    assert np.array_equal(graph.coefficients, repeated.coefficients)


@pytest.mark.parametrize(
    ("adjacency", "coefficient", "message"),
    [
        (np.eye(3), 1.0, "zero diagonal"),
        (
            np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            1.0,
            "symmetric",
        ),
        (np.zeros((3, 3)), 1.0, "at least one graph edge"),
        (np.zeros((3, 3)), 0.0, "must be omitted"),
    ],
)
def test_graph_nmf_rejects_invalid_adjacency_contract(adjacency, coefficient, message):
    with pytest.raises(ValueError, match=message):
        fit_nmf(
            np.ones((8, 3)),
            rank=2,
            graph_adjacency=adjacency,
            graph_lambda=coefficient,
        )


def test_checked_in_provisional_graph_cannot_enable_graph_nmf():
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph_payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="no verified training-enabled chains"):
        load_verified_graph_regularization(
            taxonomy_path=TAXONOMY_PATH,
            continuity_path=GRAPH_PATH,
            expected_taxonomy_fingerprint=taxonomy.fingerprint,
            expected_continuity_fingerprint=graph_payload["graph_fingerprint"],
            muscle_names=taxonomy.actuator_names,
            lambda_value=0.1,
        )


def test_verified_binding_subsets_and_manifest_tampering_are_content_bound(tmp_path):
    binding = _verified_binding(tmp_path)
    assert binding.enabled is True
    assert binding.edge_count == len(
        json.loads(_verified_graph_path(tmp_path)[0].read_text(encoding="utf-8"))["chains"][0]["edges"]
    )
    first_members = np.flatnonzero(np.sum(binding.adjacency, axis=1) > 0.0)
    subset = binding.subset(first_members, scope="unit_test_subset")
    assert subset.edge_count == binding.edge_count
    assert subset.manifest["scope"] == "unit_test_subset"
    isolated = binding.restrict_to([0], scope="unit_test_isolated")
    assert isolated.enabled is False
    assert isolated.lambda_value == 0.0

    tampered = copy.deepcopy(binding.manifest)
    tampered["edge_count"] += 1
    with pytest.raises(ValueError, match=r"exceeds|enabled|edge_count"):
        GraphRegularizationBinding(binding.adjacency, binding.muscle_names, tampered)


def test_fit_config_and_cli_require_unscaled_verified_graph_inputs(tmp_path):
    binding = _verified_binding(tmp_path, coefficient=0.5)
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph_path, graph_fingerprint = _verified_graph_path(tmp_path)
    config = SynergyFitConfig(
        normalization="none",
        graph_regularization_lambda=0.5,
        graph_taxonomy_path=str(TAXONOMY_PATH),
        graph_continuity_path=str(graph_path),
        graph_expected_taxonomy_fingerprint=taxonomy.fingerprint,
        graph_expected_continuity_fingerprint=graph_fingerprint,
    ).validated()
    assert config.graph_regularization_lambda == binding.lambda_value

    with pytest.raises(ValueError, match="requires normalization=none"):
        SynergyFitConfig(
            graph_regularization_lambda=0.5,
            graph_taxonomy_path=str(TAXONOMY_PATH),
            graph_continuity_path=str(graph_path),
            graph_expected_taxonomy_fingerprint=taxonomy.fingerprint,
            graph_expected_continuity_fingerprint=graph_fingerprint,
        ).validated()
    with pytest.raises(ValueError, match="requires taxonomy/continuity"):
        SynergyFitConfig(
            normalization="none",
            graph_regularization_lambda=0.5,
        ).validated()

    args = build_parser().parse_args(
        [
            "--train",
            "train",
            "--val",
            "val",
            "--output-dir",
            "out",
            "--normalization",
            "none",
            "--graph-regularization-lambda",
            "0.5",
        ]
    )
    assert args.graph_regularization_lambda == 0.5


def test_graph_lineage_changes_candidate_and_frozen_decoder_fingerprints(tmp_path):
    binding = _verified_binding(tmp_path)
    basis = np.ones((len(binding.muscle_names), 1), dtype=np.float64)
    standard_candidate = candidate_basis_fingerprint(
        basis,
        muscle_names=binding.muscle_names,
        signal_kind="muscle_excitation_v2",
        region="whole_body",
    )
    graph_candidate = candidate_basis_fingerprint(
        basis,
        muscle_names=binding.muscle_names,
        signal_kind="muscle_excitation_v2",
        region="whole_body",
        graph_regularization=binding.manifest,
    )
    assert graph_candidate != standard_candidate

    zeros = np.zeros((len(binding.muscle_names),), dtype=np.float32)
    execution = build_frozen_body_decoder_execution_binding(
        mode="fixed_synergy",
        actuator_names=binding.muscle_names,
        residual_alpha=0.0,
        basis=basis,
        excitation_bounds=np.tile(np.asarray([[0.0, 1.0]]), (len(binding.muscle_names), 1)),
        coefficient_maximum=np.ones((1,), dtype=np.float32),
        coefficient_center=zeros[:1],
        coefficient_temperature=np.ones((1,), dtype=np.float32),
        tonic_baseline=zeros,
        residual_basis=np.zeros((len(binding.muscle_names), 0), dtype=np.float32),
        basis_fingerprint="a" * 64,
        runtime_basis_fingerprint="b" * 64,
        coefficient_transform_fingerprint="c" * 64,
        coefficient_statistics_fingerprint="d" * 64,
        tonic_baseline_fingerprint="e" * 64,
        residual_basis_fingerprint=None,
        residual_fit_contract_fingerprint=None,
        residual_allowed_muscle_mask_fingerprint=None,
        graph_regularization=binding.manifest,
    )
    assert execution["contract_decoder_declarations"]["graph_regularization"] == (
        validate_graph_regularization_manifest(binding.manifest)
    )


def test_basis_artifact_rejects_graph_lineage_for_another_channel_order(tmp_path):
    binding = _verified_binding(tmp_path)
    reversed_names = tuple(reversed(binding.muscle_names))
    manifest = {
        "signal_kind": "muscle_excitation_v2",
        "region": "whole_body",
        "rank": 1,
        "normalization": {"kind": "none"},
        "source_dataset_fingerprint": "fixture-source",
        "teacher_checkpoint_fingerprint": "a" * 64,
        "fit_seed": 0,
        "transform": {"kind": "unit_excitation"},
        "split_provenance": {"train": ["fixture"]},
        "train_motion_uids": [1],
        "graph_regularization": binding.manifest,
    }

    with pytest.raises(ValueError, match="graph regularization order differs"):
        save_synergy_basis(
            tmp_path / "misordered_basis",
            basis=np.ones((len(reversed_names), 1), dtype=np.float64),
            muscle_names=reversed_names,
            manifest=manifest,
        )


def test_basis_artifact_rejects_scaled_graph_nmf_lineage(tmp_path):
    binding = _verified_binding(tmp_path)
    manifest = {
        "signal_kind": "muscle_excitation_v2",
        "region": "whole_body",
        "rank": 1,
        "normalization": {
            "normalization": "channel_max",
            "scales": [2.0] * len(binding.muscle_names),
        },
        "source_dataset_fingerprint": "fixture-source",
        "teacher_checkpoint_fingerprint": "a" * 64,
        "fit_seed": 0,
        "transform": {"kind": "unit_excitation"},
        "split_provenance": {"train": ["fixture"]},
        "train_motion_uids": [1],
        "graph_regularization": binding.manifest,
    }

    with pytest.raises(ValueError, match="normalization=none"):
        save_synergy_basis(
            tmp_path / "scaled_graph_basis",
            basis=np.ones((len(binding.muscle_names), 1), dtype=np.float64),
            muscle_names=binding.muscle_names,
            manifest=manifest,
        )


def test_composite_and_hybrid_graph_normalization_accept_only_recursive_none():
    leaf = {"normalization": "none", "scales": [1.0, 1.0]}
    regional = {
        "kind": "per_region_train_only",
        "regions": {"arm": leaf, "trunk": copy.deepcopy(leaf)},
    }
    hybrid = {
        "kind": "hybrid_preserves_source_component_units",
        "regional": regional,
        "global": copy.deepcopy(leaf),
    }

    _validate_graph_normalization(regional)
    _validate_graph_normalization(hybrid)
    hybrid["regional"]["regions"]["arm"]["scales"][0] = 2.0
    with pytest.raises(ValueError, match="normalization=none"):
        _validate_graph_normalization(hybrid)
