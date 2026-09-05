from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from musclemimic.badminton import peasd_formal_release as release
from musclemimic.badminton.action_registry import resolve


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return path


def _source_record(path: Path, binding: str) -> dict:
    return {
        "path": str(path.resolve()),
        "content_sha256": _sha(path),
        "artifact_fingerprint": binding,
    }


def _metrics(*, include_stage3: bool) -> dict:
    result = {
        "physiology": {
            "m_channel": {
                "channel_count": 3,
                "channel_ids": ["M1", "M2", "M3"],
                "anchor_loss_by_channel": {"M1": 0.10, "M2": 0.11, "M3": 0.12},
                "correlation_by_channel": {"M1": 0.70, "M2": 0.60, "M3": 0.50},
                "peak_phase_error_by_channel": {
                    "M1": 0.03,
                    "M2": 0.04,
                    "M3": 0.05,
                },
                "onset_error_s_by_channel": {"M1": 0.02, "M2": 0.03, "M3": 0.04},
                "co_contraction_by_pair": {"M1__M2": 0.25},
            },
            "action_activation": {
                "action_rate": 30.0,
                "activation_rate": 30.0,
                "activation_energy": 8.0,
                "action_saturation_fraction": 0.01,
                "activation_saturation_fraction": 0.02,
            },
        },
        "tracking_safety": {
            "fall_rate": 0.0,
            "early_termination_rate": 0.01,
            "joint_position_error_m": 0.02,
            "keypoint_position_error_m": 0.03,
            "root_position_error_m": 0.01,
            "tracking_error_m": 0.025,
        },
        "latent": {
            "context_response_l2": 0.4,
            "blank_context_response_l2": 0.2,
            "shuffled_context_head_loss": 0.8,
            "synergy_head_loss": 0.4,
            "synergy_head_correlation": 0.6,
        },
    }
    if include_stage3:
        result["stage3"] = {
            "hit_rate": 0.8,
            "no_fall_rate": 0.95,
            "opponent_back_landing_rate": 0.6,
            "impact_position_error_m": 0.08,
            "legal_landing_rate": 0.7,
            "recovery_complete_rate": 0.75,
            "normalized_control_energy": 0.3,
        }
    return result


def _seal_evaluation(
    path: Path,
    *,
    action: str,
    stage1_binding: str,
    stage2_binding: str,
    stage3_binding: str | None,
    execution: dict | None = None,
    statistical_scope: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    spec = resolve(action)
    reported_metrics = (
        metrics if metrics is not None else _metrics(include_stage3=spec.stage3_applicable)
    )
    source_ids = ["physiology", "stage1", "stage2"]
    if spec.stage3_applicable:
        source_ids.append("stage3")
    source_artifacts = {}
    for source_id in source_ids:
        source_path = _write(
            path.with_name(f"{path.stem}_{source_id}_source.json"),
            {
                "schema_version": f"{source_id}_formal_evaluator_fixture_v1",
                "metrics": reported_metrics,
            },
        )
        source_artifacts[source_id] = {
            "path": str(source_path.resolve()),
            "content_sha256": _sha(source_path),
            "schema_version": f"{source_id}_formal_evaluator_fixture_v1",
        }
    required_paths = list(release.COMMON_REQUIRED_METRIC_PATHS)
    if spec.stage3_applicable:
        required_paths.extend(release.STAGE3_REQUIRED_METRIC_PATHS)
    metric_provenance = {}
    for dotted_path in required_paths:
        try:
            value = release._metric_at(reported_metrics, dotted_path)
        except ValueError:
            continue
        metric_provenance[dotted_path] = {
            "source": release._expected_metric_source_id(dotted_path),
            "json_path": f"metrics.{dotted_path}",
            "value_sha256": release._canonical_value_sha256(value),
        }
    unsigned = {
        "schema_version": release.COMPLETE_EVALUATION_SCHEMA_VERSION,
        "action": {"slug": spec.slug, "action_id": spec.action_id},
        "execution": execution
        or {
            "mode": "formal",
            "completed": True,
            "passed": True,
            "dry_run": False,
            "placeholder": False,
        },
        "upstream_bindings": release._expected_upstream_bindings(
            stage1_binding=stage1_binding,
            stage2_binding=stage2_binding,
            stage3_binding=stage3_binding,
            stage3_applicable=spec.stage3_applicable,
        ),
        "statistical_scope": statistical_scope
        or {
            "physiology_unit": "trial_subject_session",
            "rl_unit": "independent_training_seed",
            "rl_training_seeds": [0, 1, 2],
            "episode_frame_feed_as_independent_n": False,
            "significance_claimed": False,
            "population_level_effect_claimed": False,
            "population_physiology_claimed": False,
        },
        "source_artifacts": source_artifacts,
        "metric_provenance": metric_provenance,
        "metrics": reported_metrics,
    }
    return _write(
        path,
        {**unsigned, "binding_sha256": release._canonical_sha256(unsigned)},
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    action: str,
) -> dict:
    spec = resolve(action)
    bindings = {
        "stage1": "1" * 64,
        "shared": "2" * 64,
        "stage2_index": "3" * 64,
        "stage2": "4" * 64,
        "stage3_index": "5" * 64,
        "stage3": "6" * 64,
    }
    stage1 = {
        "schema_version": "stage1_peasd_teacher_promotion_v1",
        "action": spec.slug,
        "action_id": spec.action_id,
        "passed": True,
        "binding_sha256": bindings["stage1"],
    }
    stage1_path = _write(tmp_path / "stage1_promotion.json", stage1)

    shared = {
        "schema_version": "stage2_shared_inputs_v1",
        "action": {
            "slug": spec.slug,
            "action_id": spec.action_id,
            "tube_action_id": spec.emg_trial_actions[0],
        },
        "stage1_peasd": {
            "promotion": _source_record(stage1_path, bindings["stage1"]),
            "promotion_binding_sha256": bindings["stage1"],
        },
        "binding_sha256": bindings["shared"],
    }
    shared_path = _write(tmp_path / "stage2_shared.json", shared)
    stage2_index = {
        "schema_version": "stage2_context_family_index_v1",
        "action": copy.deepcopy(shared["action"]),
        "shared_inputs": _source_record(shared_path, bindings["shared"]),
        "binding_sha256": bindings["stage2_index"],
    }
    stage2_index_path = _write(tmp_path / "stage2_index.json", stage2_index)
    stage2 = {
        "schema_version": "stage2_context_family_gate_v1",
        "action": copy.deepcopy(shared["action"]),
        "passed": True,
        "family_index": _source_record(stage2_index_path, bindings["stage2_index"]),
        "binding_sha256": bindings["stage2"],
    }
    stage2_path = _write(tmp_path / "stage2_gate.json", stage2)

    stage3_index = None
    stage3_index_path = None
    stage3 = None
    stage3_path = None
    if spec.stage3_applicable:
        stage3_index = {
            "schema_version": "stage3_peasd_family_index_v1",
            "action": copy.deepcopy(shared["action"]),
            "stage2_context_family": {
                "gate": _source_record(stage2_path, bindings["stage2"]),
            },
            "binding_sha256": bindings["stage3_index"],
        }
        stage3_index_path = _write(tmp_path / "stage3_index.json", stage3_index)
        stage3 = {
            "schema_version": "stage3_peasd_family_gate_v1",
            "passed": True,
            "family_index": _source_record(
                stage3_index_path, bindings["stage3_index"]
            ),
            "binding_sha256": bindings["stage3"],
        }
        stage3_path = _write(tmp_path / "stage3_gate.json", stage3)

    monkeypatch.setattr(
        release,
        "validate_stage1_peasd_teacher_promotion",
        lambda _path, expected_action=None: copy.deepcopy(stage1),
    )
    monkeypatch.setattr(
        release,
        "validate_stage2_context_family_gate",
        lambda _path, require_pass=True: copy.deepcopy(stage2),
    )
    monkeypatch.setattr(
        release,
        "validate_stage2_context_family_index",
        lambda _path: copy.deepcopy(stage2_index),
    )
    monkeypatch.setattr(
        release,
        "validate_stage2_shared_inputs",
        lambda _path, expected_action=None: copy.deepcopy(shared),
    )
    if spec.stage3_applicable:
        monkeypatch.setattr(
            release,
            "validate_stage3_peasd_family_gate",
            lambda _path, require_pass=True: copy.deepcopy(stage3),
        )
        monkeypatch.setattr(
            release,
            "validate_stage3_peasd_family_index",
            lambda _path: copy.deepcopy(stage3_index),
        )

    evaluation_path = _seal_evaluation(
        tmp_path / "complete_evaluation.json",
        action=spec.slug,
        stage1_binding=bindings["stage1"],
        stage2_binding=bindings["stage2"],
        stage3_binding=bindings["stage3"] if spec.stage3_applicable else None,
    )
    return {
        "spec": spec,
        "bindings": bindings,
        "stage1": stage1,
        "stage1_path": stage1_path,
        "shared": shared,
        "shared_path": shared_path,
        "stage2_index": stage2_index,
        "stage2_index_path": stage2_index_path,
        "stage2": stage2,
        "stage2_path": stage2_path,
        "stage3_index": stage3_index,
        "stage3_index_path": stage3_index_path,
        "stage3": stage3,
        "stage3_path": stage3_path,
        "evaluation_path": evaluation_path,
    }


def _build_kwargs(fixture: dict) -> dict:
    applicable = fixture["spec"].stage3_applicable
    return {
        "action": fixture["spec"].slug,
        "stage1_peasd_promotion": fixture["stage1_path"],
        "stage2_context_family_gate": fixture["stage2_path"],
        "stage3_peasd_family_gate": fixture["stage3_path"],
        "stage3_not_applicable": not applicable,
        "complete_evaluation_evidence": fixture["evaluation_path"],
    }


def test_clear_release_rebuilds_complete_lineage_and_metrics(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    report = release.build_peasd_formal_release_report(**_build_kwargs(fixture))

    assert report["schema_version"] == release.FORMAL_RELEASE_SCHEMA_VERSION
    assert report["passed"] is True
    assert report["upstream"]["stage3_peasd_family"]["status"] == "passed"
    assert report["reported_metrics"]["stage3"]["hit_rate"] == 0.8
    assert (
        "physiology.m_channel.correlation_by_channel"
        in report["required_reported_metric_paths"]["common"]
    )
    assert report["acceptance_policy"]["release_added_numeric_thresholds"] is False
    assert report["statistical_scope"]["significance_claimed"] is False

    output = release._write_immutable(tmp_path / "release.json", report)
    assert release.validate_peasd_formal_release_report(output) == report


def test_chinajump_requires_and_preserves_explicit_stage3_na(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="chinajump")
    report = release.build_peasd_formal_release_report(**_build_kwargs(fixture))

    disposition = report["upstream"]["stage3_peasd_family"]
    assert disposition == {
        "status": "not_applicable",
        "applicable": False,
        "gate": None,
        "reason": "action_registry.stage3_applicable=false",
        "missing_treated_as_pass": False,
    }
    assert report["required_reported_metric_paths"]["stage3"]["status"] == "not_applicable"
    assert "stage3" not in report["reported_metrics"]
    assert "racket-control claim" in report["claim_scope"]["excluded"]
    assert "shuttle-hit or return claim" in report["claim_scope"]["excluded"]


def test_chinajump_missing_is_not_implicitly_na(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="chinajump")
    kwargs = _build_kwargs(fixture)
    kwargs["stage3_not_applicable"] = False

    with pytest.raises(ValueError, match="explicit --stage3-not-applicable"):
        release.build_peasd_formal_release_report(**kwargs)


def test_chinajump_rejects_inapplicable_hitting_gate(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="chinajump")
    fake_gate = _write(tmp_path / "fake_stage3.json", {"passed": True})
    kwargs = _build_kwargs(fixture)
    kwargs["stage3_peasd_family_gate"] = fake_gate

    with pytest.raises(ValueError, match="must not bind a hitting gate"):
        release.build_peasd_formal_release_report(**kwargs)


@pytest.mark.parametrize("stage3_not_applicable", [False, True])
def test_lift_cannot_release_while_stage3_assets_are_missing(
    tmp_path, monkeypatch, stage3_not_applicable
):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_lift")
    kwargs = _build_kwargs(fixture)
    kwargs["stage3_peasd_family_gate"] = None
    kwargs["stage3_not_applicable"] = stage3_not_applicable

    with pytest.raises(ValueError, match="Stage-3 applicable"):
        release.build_peasd_formal_release_report(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "dry_run"),
        ("completed", False),
        ("passed", False),
        ("dry_run", True),
        ("placeholder", True),
        ("failed", True),
    ],
)
def test_incomplete_dry_run_placeholder_or_failed_evaluation_is_rejected(
    tmp_path, monkeypatch, field, value
):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    execution = {
        "mode": "formal",
        "completed": True,
        "passed": True,
        "dry_run": False,
        "placeholder": False,
    }
    execution[field] = value
    fixture["evaluation_path"] = _seal_evaluation(
        tmp_path / "bad_execution.json",
        action="forehand_clear",
        stage1_binding=fixture["bindings"]["stage1"],
        stage2_binding=fixture["bindings"]["stage2"],
        stage3_binding=fixture["bindings"]["stage3"],
        execution=execution,
    )

    with pytest.raises(ValueError, match="dry-run, placeholder, failed, or incomplete"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


@pytest.mark.parametrize(
    "path",
    [
        "physiology.action_activation.activation_energy",
        "tracking_safety.keypoint_position_error_m",
        "latent.synergy_head_loss",
        "stage3.recovery_complete_rate",
    ],
)
def test_every_required_metric_group_fails_closed_when_missing(
    tmp_path, monkeypatch, path
):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    metrics = _metrics(include_stage3=True)
    cursor = metrics
    components = path.split(".")
    for component in components[:-1]:
        cursor = cursor[component]
    cursor.pop(components[-1])
    fixture["evaluation_path"] = _seal_evaluation(
        tmp_path / "missing_metric.json",
        action="forehand_clear",
        stage1_binding=fixture["bindings"]["stage1"],
        stage2_binding=fixture["bindings"]["stage2"],
        stage3_binding=fixture["bindings"]["stage3"],
        metrics=metrics,
    )

    with pytest.raises(ValueError, match=f"missing required metric {path}"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_m_channel_evidence_must_cover_every_declared_channel(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    metrics = _metrics(include_stage3=True)
    metrics["physiology"]["m_channel"]["onset_error_s_by_channel"].pop("M2")
    fixture["evaluation_path"] = _seal_evaluation(
        tmp_path / "incomplete_channels.json",
        action="forehand_clear",
        stage1_binding=fixture["bindings"]["stage1"],
        stage2_binding=fixture["bindings"]["stage2"],
        stage3_binding=fixture["bindings"]["stage3"],
        metrics=metrics,
    )

    with pytest.raises(ValueError, match="report every channel exactly once"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("latent.synergy_head_loss", float("nan"), "must be finite"),
        ("latent.synergy_head_correlation", 1.1, r"must lie in \[-1,1\]"),
        ("stage3.hit_rate", 1.1, r"must lie in \[0,1\]"),
        ("stage3.normalized_control_energy", -0.1, "must be non-negative"),
    ],
)
def test_nonfinite_and_out_of_domain_metrics_are_rejected(
    tmp_path, monkeypatch, path, value, message
):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    metrics = _metrics(include_stage3=True)
    cursor = metrics
    components = path.split(".")
    for component in components[:-1]:
        cursor = cursor[component]
    cursor[components[-1]] = value
    if isinstance(value, float) and value != value:
        payload = json.loads(fixture["evaluation_path"].read_text(encoding="utf-8"))
        payload["metrics"] = metrics
        fixture["evaluation_path"] = _write(
            tmp_path / "invalid_metric.json", payload
        )
    else:
        fixture["evaluation_path"] = _seal_evaluation(
            tmp_path / "invalid_metric.json",
            action="forehand_clear",
            stage1_binding=fixture["bindings"]["stage1"],
            stage2_binding=fixture["bindings"]["stage2"],
            stage3_binding=fixture["bindings"]["stage3"],
            metrics=metrics,
        )

    with pytest.raises(ValueError, match=message):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_statistical_scope_cannot_claim_significance_or_inflate_n(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    scope = {
        "physiology_unit": "trial_subject_session",
        "rl_unit": "episode",
        "rl_training_seeds": [0, 1, 2],
        "episode_frame_feed_as_independent_n": True,
        "significance_claimed": True,
        "population_level_effect_claimed": False,
        "population_physiology_claimed": False,
    }
    fixture["evaluation_path"] = _seal_evaluation(
        tmp_path / "bad_statistics.json",
        action="forehand_clear",
        stage1_binding=fixture["bindings"]["stage1"],
        stage2_binding=fixture["bindings"]["stage2"],
        stage3_binding=fixture["bindings"]["stage3"],
        statistical_scope=scope,
    )

    with pytest.raises(ValueError, match="seed as the RL unit"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_evaluation_must_bind_exact_upstream_gates(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    fixture["evaluation_path"] = _seal_evaluation(
        tmp_path / "wrong_upstream.json",
        action="forehand_clear",
        stage1_binding="9" * 64,
        stage2_binding=fixture["bindings"]["stage2"],
        stage3_binding=fixture["bindings"]["stage3"],
    )

    with pytest.raises(ValueError, match="different upstream gates"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_stage2_must_descend_from_the_supplied_stage1_promotion(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    other = _write(tmp_path / "other_stage1.json", {"other": True})
    fixture["shared"]["stage1_peasd"]["promotion"] = _source_record(
        other, fixture["bindings"]["stage1"]
    )

    with pytest.raises(ValueError, match="different source"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_stage3_must_descend_from_the_supplied_stage2_family(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    other = _write(tmp_path / "other_stage2.json", {"other": True})
    fixture["stage3_index"]["stage2_context_family"]["gate"] = _source_record(
        other, fixture["bindings"]["stage2"]
    )

    with pytest.raises(ValueError, match="different source"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_changed_evaluation_invalidates_an_existing_release(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    report = release.build_peasd_formal_release_report(**_build_kwargs(fixture))
    output = release._write_immutable(tmp_path / "release.json", report)

    evidence = json.loads(fixture["evaluation_path"].read_text(encoding="utf-8"))
    evidence["metrics"]["latent"]["synergy_head_loss"] = 0.45
    evidence.pop("binding_sha256")
    evidence["binding_sha256"] = release._canonical_sha256(evidence)
    _write(fixture["evaluation_path"], evidence)

    with pytest.raises(ValueError, match="immutable source artifact"):
        release.validate_peasd_formal_release_report(output)


def test_self_bound_metric_edit_cannot_bypass_source_provenance(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    evidence = json.loads(fixture["evaluation_path"].read_text(encoding="utf-8"))
    evidence["metrics"]["tracking_safety"]["fall_rate"] = 0.5
    evidence.pop("binding_sha256")
    evidence["binding_sha256"] = release._canonical_sha256(evidence)
    _write(fixture["evaluation_path"], evidence)

    with pytest.raises(ValueError, match="immutable source artifact"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_metric_source_replacement_invalidates_complete_evidence(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    evidence = json.loads(fixture["evaluation_path"].read_text(encoding="utf-8"))
    source_path = Path(evidence["source_artifacts"]["stage2"]["path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["metrics"]["latent"]["synergy_head_loss"] = 0.9
    _write(source_path, source)

    with pytest.raises(ValueError, match="source content changed"):
        release.build_peasd_formal_release_report(**_build_kwargs(fixture))


def test_release_binding_and_immutable_writer_reject_tamper(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    report = release.build_peasd_formal_release_report(**_build_kwargs(fixture))
    output = release._write_immutable(tmp_path / "release.json", report)
    release._write_immutable(output, report)

    tampered = copy.deepcopy(report)
    tampered["claim_scope"]["supported"].append("invented claim")
    with pytest.raises(FileExistsError, match="immutable PEASD"):
        release._write_immutable(output, tampered)

    _write(output, tampered)
    with pytest.raises(ValueError, match="binding mismatch"):
        release.validate_peasd_formal_release_report(output)


def test_cli_build_and_validate(tmp_path, monkeypatch, capsys):
    fixture = _fixture(tmp_path, monkeypatch, action="forehand_clear")
    output = tmp_path / "formal_release.json"
    assert (
        release.main(
            [
                "build",
                "--action",
                "forehand_clear",
                "--stage1-peasd-promotion",
                str(fixture["stage1_path"]),
                "--stage2-context-family-gate",
                str(fixture["stage2_path"]),
                "--stage3-peasd-family-gate",
                str(fixture["stage3_path"]),
                "--complete-evaluation-evidence",
                str(fixture["evaluation_path"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert str(output) in capsys.readouterr().out
    assert release.main(["validate", "--release", str(output)]) == 0
