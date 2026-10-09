import hashlib
import json

import numpy as np
import pytest

from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
)
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.fit import (
    BasisNotEligibleForEarlyControl,
    SynergyFitConfig,
    _rank_rejection_reasons,
    fit_synergy_region,
)
from musclemimic.synergy.rank_selection import (
    DYNAMIC_COVERAGE_EVIDENCE_KIND,
    DYNAMIC_COVERAGE_SCHEMA_VERSION,
    candidate_basis_fingerprint,
    candidate_ranks_for_region,
    dynamic_coverage_artifact_fingerprint,
    dynamic_coverage_requirement,
    enforce_total_rank_budget,
    select_smallest_eligible_rank,
    validate_dynamic_coverage_gate,
    validate_dynamic_coverage_rank_inventory,
    validate_dynamic_coverage_requirement,
)
from musclemimic.synergy.schema import (
    ACTIVATION_SIGNAL_KIND,
    SignalTransform,
    SynergySignal,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _muscle_contract(names: tuple[str, ...]) -> dict:
    width = len(names)
    return {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": list(names),
        "actuator_ids": list(range(width)),
        "actuator_dyntype": ["muscle"] * width,
        "actuator_actnum": [1] * width,
        "actuator_actadr": list(range(width)),
        "model_na": width,
    }


def _dynamic_report(*, region: str, rank: int, candidate_fingerprint: str, passed: bool = True):
    mean_gap = 0.10 if passed else 0.20
    phase_gap = 0.20 if passed else 0.30
    checks = {
        "mean_dynamic_gap": mean_gap <= 0.15,
        "key_phase_dynamic_gap": phase_gap <= 0.25,
        "nonempty_rollout_evidence": True,
    }
    report = {
        "schema_version": DYNAMIC_COVERAGE_SCHEMA_VERSION,
        "evidence_kind": DYNAMIC_COVERAGE_EVIDENCE_KIND,
        "signal_kind": ACTIVATION_SIGNAL_KIND,
        "region": region,
        "rank": rank,
        "candidate_basis_fingerprint": candidate_fingerprint,
        "rollout_manifest_fingerprint": _sha("rollout"),
        "environment_fingerprint": _sha("environment"),
        "metrics": {
            "mean_dynamic_gap": mean_gap,
            "max_key_phase_dynamic_gap": phase_gap,
            "rollout_count": 8,
            "key_phase_count": 3,
            "horizon_steps": 32,
        },
        "thresholds": {
            "max_mean_dynamic_gap": 0.15,
            "max_key_phase_dynamic_gap": 0.25,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    report["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(report)
    return report


def test_region_candidate_ranks_override_only_named_regions():
    overrides = {"arm": (4, 2, 2), "trunk": (3,)}

    assert candidate_ranks_for_region((1, 2, 3), overrides, region="arm") == (2, 4)
    assert candidate_ranks_for_region((1, 2, 3), overrides, region="leg") == (1, 2, 3)

    config = SynergyFitConfig(region_ranks=overrides).validated()
    assert config.region_ranks == {"arm": (2, 4), "trunk": (3,)}


def test_rank_selection_uses_smallest_eligible_and_never_best_available_fallback():
    reports = {
        1: {"eligible": False, "validation": {"global_vaf": 0.99}},
        4: {"eligible": True, "validation": {"global_vaf": 0.95}},
        2: {"eligible": True, "validation": {"global_vaf": 0.91}},
    }
    assert select_smallest_eligible_rank(reports, region="arm") == 2

    reports[2]["eligible"] = False
    reports[4]["eligible"] = False
    with pytest.raises(BasisNotEligibleForEarlyControl, match="no rank passing every required gate"):
        select_smallest_eligible_rank(reports, region="arm")


def test_total_rank_budget_fails_instead_of_truncating_components():
    ranks = {"lower_body": 3, "trunk": 2, "right_arm": 4}
    assert enforce_total_rank_budget(ranks, total_rank_budget=9) == 9

    with pytest.raises(
        BasisNotEligibleForEarlyControl,
        match="exceeds explicit total_rank_budget=8.*truncation is forbidden",
    ):
        enforce_total_rank_budget(ranks, total_rank_budget=8)


def test_dynamic_coverage_gate_requires_bound_environment_rollout_evidence():
    requirement = dynamic_coverage_requirement(
        required=True,
        max_mean_dynamic_gap=0.15,
        max_key_phase_dynamic_gap=0.25,
        expected_environment_fingerprint=_sha("environment"),
        expected_rollout_manifest_fingerprint=_sha("rollout"),
    )
    assert validate_dynamic_coverage_requirement(requirement)["required"] is True
    invalid_requirement = dict(requirement)
    invalid_requirement["required"] = 1
    with pytest.raises(ValueError, match="must be boolean"):
        validate_dynamic_coverage_requirement(invalid_requirement)

    basis = np.asarray([[1.0, 0.0], [0.25, 0.75]], dtype=np.float64)
    fingerprint = candidate_basis_fingerprint(
        basis,
        muscle_names=("a", "b"),
        signal_kind=ACTIVATION_SIGNAL_KIND,
        region="arm",
    )
    report = _dynamic_report(region="arm", rank=2, candidate_fingerprint=fingerprint)
    assert validate_dynamic_coverage_gate(
        report,
        region="arm",
        rank=2,
        candidate_fingerprint=fingerprint,
        signal_kind=ACTIVATION_SIGNAL_KIND,
        max_mean_dynamic_gap=0.15,
        max_key_phase_dynamic_gap=0.25,
        expected_environment_fingerprint=_sha("environment"),
        expected_rollout_manifest_fingerprint=_sha("rollout"),
    )["passed"] is True

    static_proxy = dict(report)
    static_proxy["evidence_kind"] = "static_proxy_excitation_reconstruction"
    static_proxy["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(static_proxy)
    with pytest.raises(ValueError, match="static proxy is insufficient"):
        validate_dynamic_coverage_gate(
            static_proxy,
            region="arm",
            rank=2,
            candidate_fingerprint=fingerprint,
            signal_kind=ACTIVATION_SIGNAL_KIND,
            max_mean_dynamic_gap=0.15,
            max_key_phase_dynamic_gap=0.25,
            expected_environment_fingerprint=_sha("environment"),
            expected_rollout_manifest_fingerprint=_sha("rollout"),
        )

    missing_rollout = dict(report)
    missing_rollout["rollout_manifest_fingerprint"] = ""
    missing_rollout["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(
        missing_rollout
    )
    with pytest.raises(ValueError, match="rollout manifest fingerprint"):
        validate_dynamic_coverage_gate(
            missing_rollout,
            region="arm",
            rank=2,
            candidate_fingerprint=fingerprint,
            signal_kind=ACTIVATION_SIGNAL_KIND,
            max_mean_dynamic_gap=0.15,
            max_key_phase_dynamic_gap=0.25,
            expected_environment_fingerprint=_sha("environment"),
            expected_rollout_manifest_fingerprint=_sha("rollout"),
        )

    loose_thresholds = dict(report)
    loose_thresholds["thresholds"] = {
        "max_mean_dynamic_gap": 1.0,
        "max_key_phase_dynamic_gap": 1.0,
    }
    loose_thresholds["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(
        loose_thresholds
    )
    with pytest.raises(ValueError, match="configured promotion thresholds"):
        validate_dynamic_coverage_gate(
            loose_thresholds,
            region="arm",
            rank=2,
            candidate_fingerprint=fingerprint,
            signal_kind=ACTIVATION_SIGNAL_KIND,
            max_mean_dynamic_gap=0.15,
            max_key_phase_dynamic_gap=0.25,
            expected_environment_fingerprint=_sha("environment"),
            expected_rollout_manifest_fingerprint=_sha("rollout"),
        )


@pytest.mark.parametrize(
    ("field", "expected_argument", "message"),
    [
        (
            "environment_fingerprint",
            "expected_environment_fingerprint",
            "environment fingerprint mismatch",
        ),
        (
            "rollout_manifest_fingerprint",
            "expected_rollout_manifest_fingerprint",
            "rollout manifest fingerprint mismatch",
        ),
    ],
)
def test_dynamic_coverage_rejects_wrong_environment_or_rollout_lineage(
    field,
    expected_argument,
    message,
):
    fingerprint = candidate_basis_fingerprint(
        np.asarray([[1.0], [0.5]], dtype=np.float64),
        muscle_names=("a", "b"),
        signal_kind=ACTIVATION_SIGNAL_KIND,
        region="arm",
    )
    report = _dynamic_report(region="arm", rank=1, candidate_fingerprint=fingerprint)
    report[field] = _sha(f"wrong-{field}")
    report["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(report)
    expected = {
        "expected_environment_fingerprint": _sha("environment"),
        "expected_rollout_manifest_fingerprint": _sha("rollout"),
    }
    expected[expected_argument] = (
        _sha("environment")
        if field == "environment_fingerprint"
        else _sha("rollout")
    )
    with pytest.raises(ValueError, match=message):
        validate_dynamic_coverage_gate(
            report,
            region="arm",
            rank=1,
            candidate_fingerprint=fingerprint,
            signal_kind=ACTIVATION_SIGNAL_KIND,
            max_mean_dynamic_gap=0.15,
            max_key_phase_dynamic_gap=0.25,
            **expected,
        )

def test_fit_region_raises_when_every_rank_fails_offline_gates(tmp_path):
    names = ("m0", "m1", "m2")
    transform = SignalTransform(
        kind="identity_nonnegative_activation",
        raw_signal_kind=MUSCLE_ACTIVATION_SOURCE,
        formula="activation",
        actuator_names=names,
        roundoff_policy=UNIT_INTERVAL_ROUNDOFF_POLICY,
        physical_signal_schema_version=PHYSICAL_SIGNAL_SCHEMA_VERSION,
        muscle_channel_contract=_muscle_contract(names),
    )
    train_values = np.asarray(
        [
            [0.1, 0.8, 0.2],
            [0.7, 0.1, 0.5],
            [0.2, 0.4, 0.9],
            [0.8, 0.2, 0.1],
            [0.3, 0.9, 0.4],
            [0.9, 0.3, 0.8],
            [0.4, 0.6, 0.3],
            [0.6, 0.5, 0.7],
        ],
        dtype=np.float64,
    )
    train = SynergySignal(train_values, names, ACTIVATION_SIGNAL_KIND, transform)
    validation = SynergySignal(train_values[::-1].copy(), names, ACTIVATION_SIGNAL_KIND, transform)
    config = SynergyFitConfig(
        ranks=(1,),
        seeds=(0, 1),
        max_iter=30,
        split_half_repeats=1,
        bootstrap_repeats=1,
        cross_trial_max_trials=2,
        min_val_global_vaf=1.0,
        min_val_local_vaf_quantile=1.0,
        min_initialization_similarity=0.0,
        min_split_half_similarity=0.0,
        min_bootstrap_similarity=0.0,
        min_cross_trial_similarity=0.0,
    )

    with pytest.raises(BasisNotEligibleForEarlyControl, match="no rank passing every required gate"):
        fit_synergy_region(
            train,
            validation,
            train_phase_id=np.resize(np.arange(6, dtype=np.int32), len(train_values)),
            val_phase_id=np.resize(np.arange(6, dtype=np.int32), len(train_values)),
            output_path=tmp_path / "must_not_exist",
            region="whole_body",
            teacher_checkpoint_fingerprint="a" * 64,
            source_dataset_fingerprint="dataset",
            split_provenance={"train": {}, "validation": {}},
            config=config,
            train_motion_ids=None,
        )
    assert not (tmp_path / "must_not_exist" / "manifest.json").exists()


def test_required_dynamic_gate_without_evidence_leaves_no_eligible_rank():
    reports = {
        1: {
            "eligible": False,
            "rejection_reasons": ["required_dynamic_coverage_evidence_missing"],
        }
    }
    with pytest.raises(BasisNotEligibleForEarlyControl):
        select_smallest_eligible_rank(reports, region="whole_body")

    with pytest.raises(ValueError, match="unconfigured or unevaluated rank 2"):
        validate_dynamic_coverage_rank_inventory(
            {2: {}},
            candidate_ranks=(1,),
        )


def test_fit_region_requires_and_consumes_bound_dynamic_coverage(tmp_path):
    names = ("m0", "m1", "m2")
    transform = SignalTransform(
        kind="identity_nonnegative_activation",
        raw_signal_kind=MUSCLE_ACTIVATION_SOURCE,
        formula="activation",
        actuator_names=names,
        roundoff_policy=UNIT_INTERVAL_ROUNDOFF_POLICY,
        physical_signal_schema_version=PHYSICAL_SIGNAL_SCHEMA_VERSION,
        muscle_channel_contract=_muscle_contract(names),
    )
    coefficients = np.asarray(
        [0.10, 0.25, 0.40, 0.55, 0.70, 0.85] * 2,
        dtype=np.float64,
    )
    direction = np.asarray([0.20, 0.55, 0.90], dtype=np.float64)
    values = coefficients[:, None] * direction[None, :]
    train = SynergySignal(values, names, ACTIVATION_SIGNAL_KIND, transform)
    validation = SynergySignal(values[::-1].copy(), names, ACTIVATION_SIGNAL_KIND, transform)
    common = {
        "ranks": (1,),
        "seeds": (0, 1),
        "max_iter": 100,
        "split_half_repeats": 1,
        "bootstrap_repeats": 1,
        "cross_trial_max_trials": 2,
        "min_val_global_vaf": 0.90,
        "min_val_local_vaf_quantile": 0.90,
        "min_initialization_similarity": 0.90,
        "min_split_half_similarity": 0.90,
        "min_bootstrap_similarity": 0.90,
        "min_cross_trial_similarity": 0.90,
    }
    kwargs = {
        "train_phase_id": np.resize(np.arange(6, dtype=np.int32), len(values)),
        "val_phase_id": np.resize(np.arange(6, dtype=np.int32), len(values)),
        "region": "arm",
        "teacher_checkpoint_fingerprint": "a" * 64,
        "source_dataset_fingerprint": "dataset",
        "split_provenance": {"train": {}, "validation": {}},
        "train_motion_ids": np.repeat(np.arange(2, dtype=np.int64), 6),
    }
    offline = fit_synergy_region(
        train,
        validation,
        output_path=tmp_path / "offline",
        config=SynergyFitConfig(**common),
        **kwargs,
    )
    candidate_fingerprint = offline["selected_metrics"]["candidate_basis_fingerprint"]
    dynamic_report = _dynamic_report(
        region="arm",
        rank=1,
        candidate_fingerprint=candidate_fingerprint,
    )

    gated = fit_synergy_region(
        train,
        validation,
        output_path=tmp_path / "dynamic",
        config=SynergyFitConfig(
            require_dynamic_coverage=True,
            expected_environment_fingerprint=_sha("environment"),
            expected_rollout_manifest_fingerprint=_sha("rollout"),
            **common,
        ),
        dynamic_coverage_reports={"1": dynamic_report},
        **kwargs,
    )
    assert gated["selected_rank"] == 1
    assert gated["selected_metrics"]["dynamic_coverage"]["passed"] is True
    assert gated["selected_metrics"]["eligible"] is True


def test_multirank_dynamic_fit_persists_candidates_then_finalizes_smallest_passing_rank(
    tmp_path,
):
    names = ("m0", "m1", "m2")
    transform = SignalTransform(
        kind="identity_nonnegative_activation",
        raw_signal_kind=MUSCLE_ACTIVATION_SOURCE,
        formula="activation",
        actuator_names=names,
        roundoff_policy=UNIT_INTERVAL_ROUNDOFF_POLICY,
        physical_signal_schema_version=PHYSICAL_SIGNAL_SCHEMA_VERSION,
        muscle_channel_contract=_muscle_contract(names),
    )
    coefficients = np.asarray(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.8, 0.4],
        ]
        * 2,
        dtype=np.float64,
    )
    true_basis = np.asarray(
        [[0.8, 0.1], [0.2, 0.7], [0.3, 0.4]],
        dtype=np.float64,
    )
    values = coefficients @ true_basis.T
    train = SynergySignal(values, names, ACTIVATION_SIGNAL_KIND, transform)
    validation = SynergySignal(values[::-1].copy(), names, ACTIVATION_SIGNAL_KIND, transform)
    config = SynergyFitConfig(
        ranks=(1, 2),
        require_dynamic_coverage=True,
        expected_environment_fingerprint=_sha("environment"),
        expected_rollout_manifest_fingerprint=_sha("rollout"),
        seeds=(0, 1),
        max_iter=100,
        split_half_repeats=1,
        bootstrap_repeats=1,
        cross_trial_max_trials=2,
        min_val_global_vaf=0.0,
        min_val_local_vaf_quantile=0.0,
        min_initialization_similarity=0.0,
        min_split_half_similarity=0.0,
        min_bootstrap_similarity=0.0,
        min_cross_trial_similarity=0.0,
        max_basis_condition_number=1.0e12,
        min_effective_rank_fraction=0.0,
    )
    kwargs = {
        "train_phase_id": np.resize(np.arange(6, dtype=np.int32), len(values)),
        "val_phase_id": np.resize(np.arange(6, dtype=np.int32), len(values)),
        "region": "arm",
        "teacher_checkpoint_fingerprint": "a" * 64,
        "source_dataset_fingerprint": "dataset",
        "split_provenance": {"train": {}, "validation": {}},
        "train_motion_ids": np.repeat(np.arange(2, dtype=np.int64), 6),
    }
    candidate_root = tmp_path / "candidate_stage"
    with pytest.raises(BasisNotEligibleForEarlyControl, match="second-stage"):
        fit_synergy_region(
            train,
            validation,
            output_path=candidate_root,
            config=config,
            **kwargs,
        )

    inventory = json.loads(
        (candidate_root / "candidate_inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["candidate_ranks"] == [1, 2]
    assert [item["rank"] for item in inventory["candidates"]] == [1, 2]
    for item in inventory["candidates"]:
        candidate = load_synergy_basis(
            candidate_root / item["candidate_artifact_path"]
        )
        assert (
            candidate_basis_fingerprint(
                candidate.basis,
                muscle_names=candidate.muscle_names,
                signal_kind=ACTIVATION_SIGNAL_KIND,
                region="arm",
            )
            == item["candidate_basis_fingerprint"]
        )

    reports = {
        str(item["rank"]): _dynamic_report(
            region="arm",
            rank=item["rank"],
            candidate_fingerprint=item["candidate_basis_fingerprint"],
            passed=item["rank"] == 2,
        )
        for item in inventory["candidates"]
    }
    finalized = fit_synergy_region(
        train,
        validation,
        output_path=tmp_path / "finalized_stage",
        config=config,
        dynamic_coverage_reports=reports,
        **kwargs,
    )
    assert finalized["selected_rank"] == 2


def test_numerical_condition_and_effective_rank_are_fail_closed_gates():
    config = SynergyFitConfig(
        max_basis_condition_number=10.0,
        min_effective_rank_fraction=0.75,
    ).validated()
    reasons = _rank_rejection_reasons(
        val_global_vaf=1.0,
        val_local_quantile=1.0,
        initialization=1.0,
        split_half=1.0,
        bootstrap=1.0,
        cross_trial={"available": True, "mean_similarity": 1.0},
        primitive_group_min_vaf=None,
        basis_condition_number_value=11.0,
        effective_rank_fraction=0.5,
        config=config,
    )
    assert "basis_condition_number_above_threshold_or_nonfinite" in reasons
    assert "effective_rank_fraction_below_threshold_or_nonfinite" in reasons
