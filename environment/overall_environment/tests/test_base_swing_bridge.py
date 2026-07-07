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
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402

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
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]
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
        shape_report=ActorCheckpointShapeReport(True, (hidden[-1], len(actuator_names)), (len(actuator_names),), (obs_size,), "test"),
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
