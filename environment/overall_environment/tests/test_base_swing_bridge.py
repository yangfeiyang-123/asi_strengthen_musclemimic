from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.base_swing_bridge import (  # noqa: E402
    BaseSwingBridge,
    SwingPhaseConfig,
    compose_selected_physical_correction,
    interpolate_correction_prior,
    selected_correction_window,
)
from environment.overall_environment.src.incoming_shuttle_hit_env import (  # noqa: E402
    IncomingShuttleHitEnv,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.overall_environment.src.shuttle_feeder import build_feed_bank  # noqa: E402

SCENE_XML = default_incoming_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="incoming scene XML not built; run environment.overall_environment.src.incoming_scene",
)


def test_swing_phase_reaches_contact_at_intercept() -> None:
    cfg = SwingPhaseConfig(swing_duration_s=1.2, contact_phase=0.55)
    intercept = 1.0
    # phase 0 before the swing window opens
    start = intercept - cfg.contact_phase * cfg.swing_duration_s
    assert cfg.phase_at(start - 0.01, intercept) == pytest.approx(0.0, abs=1e-6)
    # phase hits contact_phase exactly at intercept time
    assert cfg.phase_at(intercept, intercept) == pytest.approx(cfg.contact_phase, abs=1e-6)
    # phase saturates at 1 after the swing completes
    assert cfg.phase_at(intercept + cfg.swing_duration_s, intercept) == pytest.approx(1.0, abs=1e-6)


def test_selected_physical_correction_is_independent_and_local() -> None:
    inherited = np.linspace(-0.8, 0.8, 12, dtype=np.float32)[None, :]
    indices = (2, 5, 9)
    zero = np.zeros((1, 3), dtype=np.float32)
    scales = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
    residual_scale = np.full(12, 0.25, dtype=np.float32)

    identity = compose_selected_physical_correction(
        inherited,
        zero,
        selected_indices=indices,
        physical_scales=scales,
        inherited_residual_scale=residual_scale,
        window=np.asarray([1.0]),
    )
    np.testing.assert_array_equal(identity, inherited)

    changed = compose_selected_physical_correction(
        inherited,
        np.asarray([[0.3, -0.2, 0.7]], dtype=np.float32),
        selected_indices=indices,
        physical_scales=scales,
        inherited_residual_scale=residual_scale,
        window=np.asarray([1.0]),
    )
    unselected = sorted(set(range(inherited.shape[-1])) - set(indices))
    np.testing.assert_array_equal(changed[:, unselected], inherited[:, unselected])
    # Changing wrist authority cannot alter the inherited branch itself.
    changed_wrist_scale = compose_selected_physical_correction(
        inherited,
        np.asarray([[0.3, -0.2, 0.7]], dtype=np.float32),
        selected_indices=indices,
        physical_scales=np.asarray([0.2, 0.4, 0.9], dtype=np.float32),
        inherited_residual_scale=residual_scale,
        window=np.asarray([1.0]),
    )
    np.testing.assert_array_equal(changed_wrist_scale[:, unselected], inherited[:, unselected])
    np.testing.assert_array_equal(changed_wrist_scale[:, indices[:2]], changed[:, indices[:2]])


def test_selected_correction_numpy_and_jax_composition_match() -> None:
    import jax.numpy as jnp

    inherited = np.asarray([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)
    raw = np.asarray([[0.6, -0.7]], dtype=np.float32)
    kwargs = {
        "selected_indices": (1, 3),
        "physical_scales": np.asarray([0.3, 0.5], dtype=np.float32),
        "inherited_residual_scale": np.asarray([0.25] * 4, dtype=np.float32),
        "window": np.asarray([0.75], dtype=np.float32),
    }
    numpy_result = compose_selected_physical_correction(inherited, raw, **kwargs)
    jax_result = compose_selected_physical_correction(
        jnp.asarray(inherited),
        jnp.asarray(raw),
        **{**kwargs, "array_module": jnp},
    )
    np.testing.assert_allclose(np.asarray(jax_result), numpy_result, atol=1e-7)

    tti = np.asarray([0.8, 0.675, 0.0, -0.075, -0.2], dtype=np.float32)
    numpy_window = selected_correction_window(
        tti,
        open_s=0.70,
        close_s=-0.10,
        smoothing_s=0.05,
    )
    jax_window = selected_correction_window(
        jnp.asarray(tti),
        open_s=0.70,
        close_s=-0.10,
        smoothing_s=0.05,
        array_module=jnp,
    )
    np.testing.assert_allclose(np.asarray(jax_window), numpy_window, atol=1e-7)
    assert numpy_window[0] == 0.0
    assert numpy_window[2] == 1.0
    assert numpy_window[-1] == 0.0


def test_teacher_correction_prior_interpolation_matches_numpy_and_jax() -> None:
    import jax.numpy as jnp

    knot_time = np.asarray([-0.1, 0.0, 0.2], dtype=np.float32)
    knot_raw = np.asarray(
        [[-1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
        dtype=np.float32,
    )
    query = np.asarray([-0.2, -0.05, 0.1, 0.4], dtype=np.float32)
    expected = np.asarray(
        [[-1.0, 0.0], [-0.5, 0.5], [1.0, 2.0], [2.0, 3.0]],
        dtype=np.float32,
    )
    numpy_result = interpolate_correction_prior(
        query,
        knot_time_to_intercept_s=knot_time,
        knot_correction_raw=knot_raw,
    )
    jax_result = interpolate_correction_prior(
        jnp.asarray(query),
        knot_time_to_intercept_s=jnp.asarray(knot_time),
        knot_correction_raw=jnp.asarray(knot_raw),
        array_module=jnp,
    )
    np.testing.assert_allclose(numpy_result, expected, atol=1e-7)
    np.testing.assert_allclose(np.asarray(jax_result), expected, atol=1e-7)


def _make_synthetic_base(tmp_path: Path, model: mujoco.MjModel, *, skills: list[str]) -> Path:
    """Build a tiny valid frozen base export against the hitting scene's body."""
    sys.path.insert(0, str(REPO_ROOT / "environment" / "overall_environment" / "src"))
    from action_manifest import reconstruct_action_manifest  # noqa: E402
    from body_obs_adapter import reconstruct_body_obs_schema  # noqa: E402
    from frozen_body_policy import (  # noqa: E402
        ActorCheckpointShapeReport,
        ActorCheckpointSpec,
        FrozenBodyPolicyManifest,
        _manifest_to_dict,
        _save_npz_tree,
    )

    # Reuse the incoming scene's own actuators/joints so build_from_mujoco works.
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    # A schema whose joints/actuators all exist in the hitting scene: use the
    # root free joint + a couple of real hinge joints + all actuators, no touch.
    root_joint = "root"
    hinge_joints = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    )[:4]
    kinematic = 5 + sum(1 for _ in hinge_joints) + 6 + sum(1 for _ in hinge_joints)
    muscle = 5 * len(actuator_names)
    goal_size = 1
    total = kinematic + muscle + 0 + goal_size

    schema = {
        "total_size": total,
        "kinematic_size": kinematic,
        "muscle_size": muscle,
        "touch_size": 0,
        "goal_size": goal_size,
        "action_size": len(actuator_names),
        "root_joint_name": root_joint,
        "joint_names": list(hinge_joints),
        "actuator_names": actuator_names,
        "touch_sensor_names": [],
        "observation_names": [],
    }

    condition = len(skills) if len(skills) > 1 else 0
    obs_size = total + condition
    hidden = (32,)
    rng = np.random.default_rng(0)
    actor = {
        "Dense_0": {
            "kernel": (rng.normal(0, 0.01, (obs_size, hidden[0]))).astype(np.float32),
            "bias": np.zeros(hidden[0], dtype=np.float32),
        },
        "Dense_1": {
            "kernel": (rng.normal(0, 0.01, (hidden[0], len(actuator_names)))).astype(np.float32),
            "bias": np.zeros(len(actuator_names), dtype=np.float32),
        },
    }
    stats = {
        "mean": np.zeros(obs_size, dtype=np.float32),
        "var": np.ones(obs_size, dtype=np.float32),
        "count": np.asarray(1.0, dtype=np.float32),
    }
    root = tmp_path / "base"
    root.mkdir(parents=True, exist_ok=True)
    _save_npz_tree(root / "params.npz", {"actor": actor})
    _save_npz_tree(root / "run_stats.npz", {"RunningMeanStd_0": stats})
    spec = ActorCheckpointSpec(
        obs_size=obs_size,
        action_size=len(actuator_names),
        actor_hidden_layers=hidden,
        critic_hidden_layers=hidden,
        activation="tanh",
        init_std=0.1,
        learnable_std=False,
        use_layernorm=False,
        layernorm_eps=1e-6,
    )
    manifest = FrozenBodyPolicyManifest(
        schema_version=1,
        source_checkpoint="test",
        tensor_format="npz",
        has_tensors=True,
        actor_spec=spec,
        shape_report=ActorCheckpointShapeReport(
            True, (hidden[-1], len(actuator_names)), (len(actuator_names),), (obs_size,), "test"
        ),
        params_file="params.npz",
        run_stats_file="run_stats.npz",
        body_obs_schema_file="body_obs_schema.json",
        action_manifest_file="action_manifest.json",
    )
    (root / "manifest.json").write_text(json.dumps(_manifest_to_dict(manifest)), encoding="utf-8")
    (root / "body_obs_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (root / "action_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "env_name": "MjxMyoFullBody",
                "disable_fingers": True,
                "action_size": len(actuator_names),
                "actuator_names": actuator_names,
                "obs_size": obs_size,
                "obs_fields": [],
                "control_min": -1.0,
                "control_max": 1.0,
            }
        ),
        encoding="utf-8",
    )
    if condition:
        (root / "skill_manifest.json").write_text(
            json.dumps({"actions": skills, "condition_size": condition}), encoding="utf-8"
        )
    return root


def test_single_skill_bridge_combines_base_and_residual(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    bridge = BaseSwingBridge(artifact, model, residual_scale=0.3)

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    residual = np.zeros(model.nu)
    combined, base = bridge.combined_action(model, data, residual, phase=0.5)
    # zero residual -> combined equals clipped base
    np.testing.assert_allclose(combined, np.clip(base, -1.0, 1.0), atol=1e-6)
    assert combined.shape == (model.nu,)
    assert np.isfinite(combined).all()

    nonzero = np.ones(model.nu)
    combined2, _ = bridge.combined_action(model, data, nonzero, phase=0.5)
    assert np.any(np.abs(combined2 - combined) > 1e-6)


def test_per_actuator_residual_authority_is_exact_and_bound(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    override_name = "DELT1"
    override_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, override_name)
    assert override_id >= 0
    bridge = BaseSwingBridge(
        artifact,
        model,
        residual_scale=0.25,
        residual_scale_overrides={override_name: 1.0},
    )
    assert bridge.residual_scale_vector[override_id] == pytest.approx(1.0)
    assert np.count_nonzero(bridge.residual_scale_vector == 1.0) == 1
    assert np.all(np.delete(bridge.residual_scale_vector, override_id) == pytest.approx(0.25))
    binding = bridge.control_binding
    assert binding["residual_scale_overrides"] == [
        {"actuator_name": override_name, "actuator_id": override_id, "scale": 1.0}
    ]
    assert len(binding["residual_scale_vector_sha256"]) == 64

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    residual = np.ones(model.nu)
    combined, base = bridge.combined_action(model, data, residual, phase=0.5)
    np.testing.assert_allclose(
        combined,
        np.clip(base + bridge.residual_scale_vector * residual, -1.0, 1.0),
        atol=1e-6,
    )


def test_per_actuator_residual_authority_rejects_unknown_or_unsafe_values(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    with pytest.raises(ValueError, match="unknown model actuators"):
        BaseSwingBridge(
            artifact,
            model,
            residual_scale_overrides={"not_an_actuator": 1.0},
        )
    with pytest.raises(ValueError, match=r"finite number in \[0, 2\]"):
        BaseSwingBridge(
            artifact,
            model,
            residual_scale_overrides={"DELT1": 2.1},
        )


def test_residual_authority_schedule_preserves_source_then_reaches_target(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    override_name = "DELT1"
    override_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, override_name)
    bridge = BaseSwingBridge(
        artifact,
        model,
        residual_scale=0.25,
        residual_scale_overrides={override_name: 1.0},
        residual_scale_schedule={"initial_scale": 0.25, "ramp_steps": 1_000_000},
    )
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    residual = np.zeros(model.nu)
    residual[override_id] = 0.4

    initial, base = bridge.combined_action(
        model,
        data,
        residual,
        phase=0.5,
        residual_authority_progress=0.0,
    )
    final, _ = bridge.combined_action(
        model,
        data,
        residual,
        phase=0.5,
        residual_authority_progress=1.0,
    )
    assert initial[override_id] == pytest.approx(np.clip(base[override_id] + 0.25 * residual[override_id], -1.0, 1.0))
    assert final[override_id] == pytest.approx(np.clip(base[override_id] + residual[override_id], -1.0, 1.0))
    schedule = bridge.control_binding["residual_scale_schedule"]
    assert schedule["ramp_steps"] == 1_000_000
    assert schedule["scheduled_actuators"][0]["actuator_id"] == override_id
    assert bridge.jax_arrays()["residual_scale_ramp_steps"] == 1_000_000


def test_residual_authority_schedule_requires_overrides(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    with pytest.raises(ValueError, match="requires residual_scale_overrides"):
        BaseSwingBridge(
            artifact,
            model,
            residual_scale_schedule={"initial_scale": 0.25, "ramp_steps": 10},
        )


def test_cpu_environment_passes_configured_feed_timing_to_base_bridge(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=build_feed_bank(1, seed=29),
        base_policy_artifact=artifact,
        swing_duration_s=1.1,
        contact_phase=0.73,
    )
    assert env.base_bridge is not None
    assert env.base_bridge.phase_config.swing_duration_s == pytest.approx(1.1)
    assert env.base_bridge.phase_config.contact_phase == pytest.approx(0.73)
    binding = env.control_manifest["frozen_base_residual"]
    assert binding["schema_version"] == "incoming_hit_frozen_base_residual_v1"
    assert binding["residual_scale"] == pytest.approx(0.3)
    assert binding["swing_duration_s"] == pytest.approx(1.1)
    assert binding["contact_phase"] == pytest.approx(0.73)
    assert len(binding["artifact_content_sha256"]) == 64

    changed = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=build_feed_bank(1, seed=29),
        base_policy_artifact=artifact,
        residual_scale=0.1,
        swing_duration_s=1.1,
        contact_phase=0.73,
    )
    assert changed.control_hash != env.control_hash


def test_cpu_residual_diagnostics_report_composed_base_action(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=build_feed_bank(1, seed=31),
        base_policy_artifact=artifact,
        residual_scale=0.12,
    )
    env.reset(feed_index=0)
    _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(env.action_size))
    assert info["normalized_control_energy"] > 0.0
    assert info["body_action_rms"] > 0.0


def test_cpu_residual_diagnostics_report_override_group(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear"])
    env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=build_feed_bank(1, seed=31),
        base_policy_artifact=artifact,
        residual_scale=0.25,
        residual_scale_overrides={"DELT1": 1.0, "DELT2": 1.0},
    )
    env.reset(feed_index=0)
    action = np.zeros(env.action_size)
    action[env.base_bridge.residual_override_indices] = 0.5
    _obs, _reward, _terminated, _truncated, info = env.step(action)
    assert info["residual_override_action_rms"] == pytest.approx(0.5)
    assert 0.0 <= info["residual_override_composed_saturation_fraction"] <= 1.0


def test_multi_skill_bridge_requires_and_selects_skill(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear", "smash"])
    with pytest.raises(ValueError, match="skill selection"):
        BaseSwingBridge(artifact, model)
    with pytest.raises(ValueError, match="unknown skill"):
        BaseSwingBridge(artifact, model, skill="lob")

    clear = BaseSwingBridge(artifact, model, skill="clear")
    smash = BaseSwingBridge(artifact, model, skill="smash")
    np.testing.assert_array_equal(clear.skill_onehot, [1.0, 0.0])
    np.testing.assert_array_equal(smash.skill_onehot, [0.0, 1.0])


def test_jax_arrays_export_shapes(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    artifact = _make_synthetic_base(tmp_path, model, skills=["clear", "smash"])
    bridge = BaseSwingBridge(artifact, model, skill="smash")
    arrays = bridge.jax_arrays()
    assert arrays["obs_size"] == bridge.schema.total_size + 2
    assert arrays["skill_onehot"].shape == (2,)
    assert len(arrays["layers"]) == 2
