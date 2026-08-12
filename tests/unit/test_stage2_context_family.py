from __future__ import annotations

import json

import pytest

from musclemimic.badminton import stage2_context_family as family


def _index(*, deltas: tuple[float, float, float] = (0.4, 0.3, 0.2)) -> dict:
    arms = {}
    for arm in family.STAGE2_ARMS:
        records = []
        for seed in family.EXACT_SEEDS:
            loss = 1.0
            if arm == "S2-D":
                loss += deltas[seed]
            records.append(
                {
                    "seed": seed,
                    "emg_synergy_head_loss": loss,
                    "emg_synergy_head_correlation": 0.5,
                    "emg_blank_context_posterior_mu_l2": 0.25,
                    "emg_blank_context_action_mse": 0.125,
                }
            )
        arms[arm] = {"seeds": records}
    return {
        "schema_version": family.FAMILY_INDEX_SCHEMA_VERSION,
        "action": {
            "slug": "chinajump",
            "action_id": "ChinaJump",
            "tube_action_id": "china_jump_high_clear",
        },
        "arms": arms,
        "direct_s2a_evidence": {
            "required": False,
            "status": "not_applicable",
            "complete_s2a": False,
        },
        "direct_s2a_family_promotion": {
            "required": False,
            "status": "not_applicable",
        },
        "binding_sha256": "a" * 64,
    }


def test_paired_gate_requires_three_of_three_seed_wins(tmp_path, monkeypatch):
    source = tmp_path / "family_index.json"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(family, "validate_stage2_context_family_index", lambda _path: _index())

    gate = family.build_stage2_context_family_gate(family_index=source)

    assert gate["passed"] is True
    assert [pair["seed"] for pair in gate["primary_hypothesis"]["pairs"]] == [0, 1, 2]
    assert all(pair["passed"] for pair in gate["primary_hypothesis"]["pairs"])
    assert gate["primary_hypothesis"]["statistics"]["exact_one_sided_sign_test_p"] == 0.125
    assert gate["primary_hypothesis"]["statistics"]["significance_claimed"] is False


def test_paired_gate_fails_if_one_seed_does_not_improve(tmp_path, monkeypatch):
    source = tmp_path / "family_index.json"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        family,
        "validate_stage2_context_family_index",
        lambda _path: _index(deltas=(0.4, 0.0, 0.2)),
    )

    gate = family.build_stage2_context_family_gate(family_index=source)

    assert gate["passed"] is False
    assert [pair["passed"] for pair in gate["primary_hypothesis"]["pairs"]] == [True, False, True]


def test_paired_gate_rejects_nonfinite_emg_head_metric(tmp_path, monkeypatch):
    source = tmp_path / "family_index.json"
    source.write_text("{}\n", encoding="utf-8")
    payload = _index()
    payload["arms"]["S2-C"]["seeds"][1]["emg_synergy_head_loss"] = float("nan")
    monkeypatch.setattr(family, "validate_stage2_context_family_index", lambda _path: payload)

    with pytest.raises(ValueError, match="non-finite"):
        family.build_stage2_context_family_gate(family_index=source)


@pytest.mark.parametrize("arm", ["S2-C", "S2-E"])
def test_real_context_arms_require_positive_blank_context_response(arm):
    metrics = {
        "emg_synergy_head_loss": 0.5,
        "emg_synergy_head_correlation": 0.25,
        "emg_blank_context_posterior_mu_l2": 0.0,
        "emg_blank_context_action_mse": 0.1,
    }

    with pytest.raises(ValueError, match="no positive blank-context posterior response"):
        family._context_response_metrics(metrics, arm=arm, seed=0)


def test_shuffled_context_requires_finite_bounded_head_diagnostics():
    metrics = {
        "emg_synergy_head_loss": 0.5,
        "emg_synergy_head_correlation": 1.01,
        "emg_blank_context_posterior_mu_l2": 0.0,
        "emg_blank_context_action_mse": 0.1,
    }

    with pytest.raises(ValueError, match="invalid emg_synergy_head_correlation"):
        family._context_response_metrics(metrics, arm="S2-D", seed=2)


def test_family_index_requires_all_four_arms_before_rebuild(monkeypatch):
    payload = _index()
    payload["exact_arms"] = list(family.STAGE2_ARMS)
    payload["exact_seeds"] = list(family.EXACT_SEEDS)
    payload["primary_metric"] = family.PRIMARY_METRIC
    payload["shared_inputs"] = {"path": "/shared"}
    payload["architecture_lock"] = {"path": "/lock"}
    payload["arms"].pop("S2-E")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    payload["binding_sha256"] = family.canonical_json_sha256(unsigned)
    monkeypatch.setattr(
        family,
        "build_stage2_context_family_index",
        lambda **_kwargs: pytest.fail("must reject missing E before rebuild"),
    )

    with pytest.raises(ValueError, match="exact B/C/D/E"):
        family.validate_stage2_context_family_index(payload)


def test_shared_manifest_binding_tamper_fails_before_source_rebuild(monkeypatch):
    payload = {
        "schema_version": family.SHARED_INPUTS_SCHEMA_VERSION,
        "action": {
            "slug": "chinajump",
            "action_id": "ChinaJump",
            "tube_action_id": "china_jump_high_clear",
        },
        "binding_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        family,
        "build_stage2_shared_inputs",
        lambda **_kwargs: pytest.fail("must reject tamper before rebuilding sources"),
    )

    with pytest.raises(ValueError, match="binding mismatch"):
        family.validate_stage2_shared_inputs(payload)


def test_immutable_writer_rejects_lineage_overwrite(tmp_path):
    path = tmp_path / "shared.json"
    family._write_immutable(path, {"lineage": "one"})
    family._write_immutable(path, {"lineage": "one"})

    with pytest.raises(FileExistsError, match="immutable Stage-2 artifact"):
        family._write_immutable(path, {"lineage": "two"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"lineage": "one"}


def _cd_plan(*, shuffled: bool) -> dict:
    command = [
        "scripts/run_fullbody_training.sh",
        "--latent",
        "--config",
        "/config.yaml",
        "--output_dir",
        "/arm-d" if shuffled else "/arm-c",
        "--latent_dim",
        "4",
        "--seed",
        "0",
    ]
    if shuffled:
        command.append("--emg_shuffle_context_ablation")
    return {
        "stage2_context_family": {
            "schema_version": "stage2_context_arm_binding_v1",
            "arm": "S2-D" if shuffled else "S2-C",
            "shared_inputs_binding_sha256": "1" * 64,
            "architecture_lock_binding_sha256": "2" * 64,
            "treatment": {
                "emg_privileged_enabled": True,
                "emg_context_dropout": 0.25,
                "emg_shuffle_context_ablation": shuffled,
            },
        },
        "lifecycle_inputs": {"shared": True},
        "emg_privileged": {
            "enabled": True,
            "synergy_dim": 3,
            "reference_manifest": "/tube.json",
            "shuffle_context_ablation": shuffled,
        },
        "jobs": [
            {
                "latent_dim": 4,
                "decoder_type": "fixed_synergy",
                "seed": 0,
                "synergy_basis_expected_fingerprint": "3" * 64,
                "frozen_body_decoder_expected_fingerprint": None,
                "body_synergy_contract_expected_fingerprint": None,
                "body_synergy_portable_core_expected_fingerprint": None,
                "training_command": command,
            }
        ],
    }


def test_cd_plan_comparator_permits_only_shuffle_and_output_identity():
    family._assert_cd_only_shuffle(_cd_plan(shuffled=False), _cd_plan(shuffled=True))


def test_cd_plan_comparator_rejects_second_scientific_difference():
    c_plan = _cd_plan(shuffled=False)
    d_plan = _cd_plan(shuffled=True)
    d_plan["stage2_context_family"]["treatment"]["emg_context_dropout"] = 0.0

    with pytest.raises(ValueError, match="beyond arm identity"):
        family._assert_cd_only_shuffle(c_plan, d_plan)
