from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

import musclemimic.badminton.stage3_task_causal as task_causal_module
from environment.overall_environment.src.stage3_lab import (
    ConstantGripProvider,
    Stage3ActionRouter,
    Stage3LABController,
)
from musclemimic.badminton.stage3_task_causal import (
    TASK_EFFECTS_SCHEMA_VERSION,
    build_mask_aware_task_effects,
    config_template,
    resolve_task_causal_cli_path,
    task_outcome_schemas,
    validate_symmetric_intervention_epsilons,
    validate_task_causal_branch_registry,
    validate_task_event_schema,
)


def _schemas():
    return task_outcome_schemas(
        muscle_names=("muscle_a", "muscle_b"),
        joint_names=("joint_a", "joint_b"),
    )


def _event_row(schema, **values):
    result = np.zeros((len(schema["feature_names"]),), dtype=np.float32)
    for name, value in values.items():
        result[schema["feature_names"].index(name)] = value
    return result


def test_task_event_schema_requires_presence_masked_sentinels() -> None:
    schemas = validate_task_event_schema(_schemas())
    assert schemas["impact_outcome"]["missing_event_contract"]["storage_sentinel"] == 0.0
    assert schemas["landing_outcome"]["masked_value_contracts"]

    unsafe = _schemas()
    unsafe["impact_outcome"].pop("masked_value_contracts")
    with pytest.raises(ValueError, match="masked value contracts"):
        validate_task_event_schema(unsafe)


def test_formal_task_causal_registry_requires_only_selected_synergy() -> None:
    complete = {
        "best_synergy": {"direction_source": {"analysis_inputs": "s.npz"}},
    }
    assert set(validate_task_causal_branch_registry(complete)) == {"best_synergy"}
    with pytest.raises(ValueError, match="exactly best_synergy"):
        validate_task_causal_branch_registry(
            {
                **complete,
                "best_direct": {"direction_source": {"analysis_inputs": "d.npz"}},
            }
        )


def test_task_causal_requires_symmetric_epsilon_pairs() -> None:
    np.testing.assert_array_equal(
        validate_symmetric_intervention_epsilons([-1.0, -0.5, 0.5, 1.0]),
        [-1.0, -0.5, 0.5, 1.0],
    )
    with pytest.raises(ValueError, match="symmetric"):
        validate_symmetric_intervention_epsilons([-0.5, 0.25, 0.5])


def test_task_causal_paths_use_cli_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "artifacts" / "evidence.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    assert resolve_task_causal_cli_path("artifacts/evidence.json", strict=True) == artifact
    assert resolve_task_causal_cli_path("outputs/task") == tmp_path / "outputs" / "task"


def test_default_trunk_bodies_exist_in_released_stage3_scene() -> None:
    root = Path(__file__).resolve().parents[2]
    scene = root / "environment/overall_environment/assets/overall_incoming_hit_scene.xml"
    names = {element.attrib.get("name") for element in ElementTree.parse(scene).getroot().iter("body")}
    assert {"Full Body", "torso"} <= names
    schema = task_outcome_schemas(
        muscle_names=("muscle",),
        joint_names=("joint",),
    )
    assert any("Full Body" in name for name in schema["trunk_state"]["feature_names"])


def test_public_task_causal_template_matches_builtin_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    public = json.loads((root / "configs/public/latent_task_causal_v2_template.json").read_text(encoding="utf-8"))
    assert public == config_template()
    assert set(public["branches"]) == {"best_synergy"}
    assert public["claim_gate"]["full354_latent_intervention"] == "not_applicable_no_latent_coordinate"
    assert public["output_dir"] == "outputs/synergy_v3/stage3_task_causal"
    canonical_spec = (root / "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^\s*max_episode_steps:\s*(\d+)\s*$", canonical_spec, re.MULTILINE)
    assert match is not None
    assert public["rollout_horizon_steps"] >= int(match.group(1))


def test_builtin_template_routes_to_selected_synergy_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    payload = config_template()
    payload["output_dir"] = "outputs/not-created"
    config_path = tmp_path / "task_causal.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    observed = {}

    def stop_after_source_binding(evaluation, *, synergy_selection, family):
        observed.update(
            evaluation=evaluation,
            synergy_selection=synergy_selection,
            family=family,
        )
        raise RuntimeError("selected-synergy-source-validation-reached")

    monkeypatch.setattr(task_causal_module, "load_branch_context", stop_after_source_binding)
    with pytest.raises(RuntimeError, match="selected-synergy-source-validation-reached"):
        task_causal_module.run_task_causal(config_path, dry_run=True)

    assert observed == {
        "evaluation": payload["synergy_evaluation"],
        "synergy_selection": payload["synergy_selection"],
        "family": "best_synergy",
    }


def test_effective_latent_override_is_explicit_and_preserves_other_routing() -> None:
    class Runtime:
        state_dim = 2
        latent_dim = 1
        action_dim = 1
        schema_hash = "runtime"

        @staticmethod
        def prior_raw_numpy(state):
            return np.zeros((1,)), np.zeros((1,))

        @staticmethod
        def prior_raw_jax(state):
            return np.zeros((1,)), np.zeros((1,))

        @staticmethod
        def decoder_numpy(state, latent):
            return np.asarray(latent)

        @staticmethod
        def decoder_jax(state, latent):
            return latent

    router = Stage3ActionRouter(
        all_actuator_names=("body", "right_grip", "left_neutral"),
        body_actuator_names=("body",),
        right_grip_actuator_names=("right_grip",),
        left_neutral_actuator_names=("left_neutral",),
        expected_sizes=(1, 1, 1),
    )
    controller = Stage3LABController(
        runtime=Runtime(),
        router=router,
        right_grip_provider=ConstantGripProvider((0.7,)),
    )
    normal = controller.decode_task_numpy(
        lab_state=np.asarray([0.0, 0.0]),
        task_action=np.asarray([0.0]),
    )
    intervened = controller.decode_task_with_latent_override_numpy(
        lab_state=np.asarray([0.0, 0.0]),
        task_action=np.asarray([0.0]),
        effective_latent=np.asarray([0.9]),
    )
    assert normal.latent[0] == pytest.approx(0.0)
    assert intervened.latent[0] == pytest.approx(0.9)
    np.testing.assert_allclose(intervened.full_action, [0.9, 0.7, 0.0])
    np.testing.assert_allclose(intervened.raw_latent, [0.0])


def test_fingerless_rigid_fixture_has_no_hand_provider_or_hidden_action() -> None:
    class Runtime:
        state_dim = 1
        latent_dim = 1
        action_dim = 2

        @staticmethod
        def prior_raw_numpy(state):
            return np.zeros((1,)), np.zeros((1,))

        @staticmethod
        def prior_raw_jax(state):
            return np.zeros((1,)), np.zeros((1,))

        @staticmethod
        def decoder_numpy(state, latent):
            return np.asarray([latent[0], -latent[0]])

        @staticmethod
        def decoder_jax(state, latent):
            return np.asarray([latent[0], -latent[0]])

    router = Stage3ActionRouter(
        all_actuator_names=("body_a", "body_b"),
        body_actuator_names=("body_a", "body_b"),
        right_grip_actuator_names=(),
        left_neutral_actuator_names=(),
        expected_sizes=(2, 0, 0),
    )
    assert router.fixture_mode == "rigid_tool_fingerless"
    controller = Stage3LABController(runtime=Runtime(), router=router)
    output = controller.decode_task_numpy(
        lab_state=np.asarray([0.0]),
        task_action=np.asarray([0.5]),
    )
    np.testing.assert_allclose(output.full_action, output.body_action)
    assert output.full_action.shape == (2,)
    assert output.right_grip_action.shape == (0,)
    assert output.left_neutral_action.shape == (0,)
    assert controller.control_manifest["full_action_dim"] == 2
    assert controller.control_manifest["grip_provider_schema_hash"] is None

    with pytest.raises(ValueError, match="must not install a hand provider"):
        Stage3LABController(
            runtime=Runtime(),
            router=router,
            right_grip_provider=ConstantGripProvider(()),
        )


def test_mask_aware_effects_never_subtract_missing_event_sentinels(tmp_path) -> None:
    schemas = _schemas()
    impact_schema = schemas["impact_outcome"]
    landing_schema = schemas["landing_outcome"]
    baseline_impact = np.stack(
        [
            _event_row(
                impact_schema,
                hit_present=1,
                impact_measurement_present=1,
                impact_position_error_present=1,
                impact_position_error_m=0.2,
                impact_timing_error_present=1,
                impact_timing_signed_error_s=0.05,
                stringbed_normal_error_present=1,
                stringbed_normal_error_rad=0.1,
                racket_linear_velocity_error_present=1,
                racket_linear_velocity_error_m_s=0.3,
                racket_angular_velocity_error_present=1,
                racket_angular_velocity_error_rad_s=0.4,
            ),
            _event_row(impact_schema, miss_present=1),
        ]
    )
    baseline_landing = np.stack(
        [
            _event_row(
                landing_schema,
                return_landing_present=1,
                landing_error_present=1,
                landing_error_m=0.5,
                landing_xy_present=1,
                landing_x_m=4.0,
                landing_y_m=0.2,
                flight_resolved_present=1,
                ground_contact_present=1,
                ground_contact_x_m=4.0,
                ground_contact_y_m=0.2,
                apex_measurement_present=1,
                apex_height_m=4.5,
                recovery_complete_present=1,
            ),
            _event_row(
                landing_schema,
                ground_contact_present=1,
                ground_contact_x_m=-2.0,
                ground_contact_y_m=0.0,
                apex_measurement_present=1,
                apex_height_m=2.0,
                unresolved_flight_present=1,
            ),
        ]
    )
    changed_impact = np.zeros((2, 1, 2, baseline_impact.shape[-1]), dtype=np.float32)
    changed_landing = np.zeros((2, 1, 2, baseline_landing.shape[-1]), dtype=np.float32)
    # sample 0: one both-present measurement and one lost event
    changed_impact[0, 0, 0] = baseline_impact[0]
    changed_impact[0, 0, 0, impact_schema["feature_names"].index("impact_position_error_m")] = 0.3
    changed_impact[0, 0, 1] = _event_row(impact_schema, miss_present=1)
    changed_landing[0, 0, 0] = baseline_landing[0]
    changed_landing[0, 0, 0, landing_schema["feature_names"].index("landing_error_m")] = 0.7
    changed_landing[0, 0, 1] = _event_row(
        landing_schema,
        ground_contact_present=1,
        ground_contact_x_m=-1.0,
        apex_measurement_present=1,
        apex_height_m=2.5,
        unresolved_flight_present=1,
    )
    # sample 1: one gained event and one neither-present event
    changed_impact[1, 0, 0] = baseline_impact[0]
    changed_impact[1, 0, 1] = baseline_impact[1]
    changed_landing[1, 0, 0] = baseline_landing[0]
    changed_landing[1, 0, 1] = baseline_landing[1]

    baseline_path = tmp_path / "baseline.npz"
    perturbed_path = tmp_path / "perturbed.npz"
    np.savez_compressed(
        baseline_path,
        impact_outcome=baseline_impact,
        landing_outcome=baseline_landing,
    )
    np.savez_compressed(
        perturbed_path,
        impact_outcome=changed_impact,
        landing_outcome=changed_landing,
    )
    manifest = build_mask_aware_task_effects(
        baseline_records=baseline_path,
        perturbed_records=perturbed_path,
        outcome_schemas=schemas,
        output_npz=tmp_path / "task_effects.npz",
    )
    assert manifest["schema_version"] == TASK_EFFECTS_SCHEMA_VERSION
    assert manifest["zero_sentinel_used_as_measurement"] is False
    assert manifest["event_pair_counts"]["impact_outcome"] == {
        "both_present": 1,
        "lost_event": 1,
        "gained_event": 1,
        "neither_present": 1,
    }
    with np.load(tmp_path / "task_effects.npz", allow_pickle=False) as effects:
        delta = effects["impact_outcome__impact_position_error_m__delta"]
        mask = effects["impact_outcome__impact_position_error_m__both_present_mask"]
    assert mask.tolist() == [[[True, False]], [[False, False]]]
    assert delta[0, 0, 0] == pytest.approx(0.1)
    assert np.isnan(delta[0, 0, 1])
    assert np.isnan(delta[1]).all()


def test_mask_aware_effects_reject_nonzero_value_when_presence_is_false(tmp_path) -> None:
    schemas = _schemas()
    impact = _event_row(
        schemas["impact_outcome"],
        miss_present=1,
        impact_position_error_m=123.0,
    )[None, :]
    landing = _event_row(
        schemas["landing_outcome"],
        unresolved_flight_present=1,
        apex_measurement_present=1,
        apex_height_m=2.0,
    )[None, :]
    baseline = tmp_path / "baseline.npz"
    perturbed = tmp_path / "perturbed.npz"
    np.savez_compressed(baseline, impact_outcome=impact, landing_outcome=landing)
    np.savez_compressed(
        perturbed,
        impact_outcome=impact[:, None, None, :],
        landing_outcome=landing[:, None, None, :],
    )
    with pytest.raises(ValueError, match="storage sentinel"):
        build_mask_aware_task_effects(
            baseline_records=baseline,
            perturbed_records=perturbed,
            outcome_schemas=schemas,
            output_npz=tmp_path / "unsafe.npz",
        )
