import copy
import json
import math
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import musclemimic_models
import numpy as np
import pytest

from src.grip.build_right_hand_racket_grip_scene import build_scene
from src.grip.evaluate_right_hand_racket_grip import evaluate
from src.grip.grip_math import angle_between_vectors, normalized
from src.grip.grip_objectives import joint_limit_margin_cost, mean_site_error, weighted_site_target_residuals
from src.grip.hand_racket_model_map import load_model_map
from src.grip.paths import REPO_ROOT, racket_xml_path, scene_xml_path, target_config_path
from src.grip.right_hand_racket_grip_env import RightHandRacketGripEnv
from src.grip.solve_right_hand_racket_grip import (
    hand_site_positions,
    mean_error_meets_threshold,
    quality_exit_code,
    racket_local_targets_to_world,
    solve_reference,
)
from src.grip.target_config import GripTargetConfig, load_grip_target_config
from src.grip.train_right_hand_racket_grip import run_baseline
from src.grip.train_right_hand_racket_grip_policy import train_policy
from src.grip.validate_right_hand_racket_grip import validate_grip
from src.grip.visualize_grip_sites import collect_site_positions


def _default_raw_config():
    return copy.deepcopy(load_grip_target_config().raw)


def _write_config(tmp_path, raw):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _write_impossible_acceptance_config(tmp_path):
    raw = _default_raw_config()
    raw["training_acceptance"] = {
        **raw["training_acceptance"],
        "max_mean_site_error_m": 0.0,
        "max_racket_translation_drift_m_2s": 0.0,
        "max_racket_orientation_drift_deg_2s": 0.0,
        "min_handle_contacts": 10_000,
    }
    return _write_config(tmp_path, raw)


def _write_acceptance_override(tmp_path, **overrides):
    raw = _default_raw_config()
    raw["training_acceptance"] = {**raw["training_acceptance"], **overrides}
    return _write_config(tmp_path, raw)


def _build_smoke_env(tmp_path):
    scene = tmp_path / "grip_scene.xml"
    reference = tmp_path / "reference.json"
    build_scene(scene)
    solve_reference(scene, target_config_path(), reference, max_nfev=2)
    return RightHandRacketGripEnv(scene, target_config_path(), reference)


def _build_smoke_paths(tmp_path):
    scene = tmp_path / "grip_scene.xml"
    reference = tmp_path / "reference.json"
    build_scene(scene)
    solve_reference(scene, target_config_path(), reference, max_nfev=2)
    return scene, target_config_path(), reference


def test_package_discovery_includes_local_src_package():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    includes = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "src*" in includes


def test_repo_paths_resolve_existing_racket_asset():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / ".git").exists()
    assert racket_xml_path().is_file()
    assert racket_xml_path().name == "badminton_racket_rigid.xml"


def test_build_grip_scene_contains_required_sites(tmp_path):
    from src.grip.build_right_hand_racket_grip_scene import build_scene

    out = tmp_path / "grip_scene.xml"
    build_scene(output_xml=out)
    model = mujoco.MjModel.from_xml_path(str(out))
    model_map = load_model_map(model)
    assert model_map.ok, model_map.missing
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handle_grip") >= 0
    for index in range(8):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"handle_bevel_{index:02d}") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_bevel_right_site") >= 0


def test_build_grip_scene_uses_dedicated_handle_contact_filter(tmp_path):
    from src.grip.build_right_hand_racket_grip_scene import build_scene

    out = tmp_path / "grip_scene.xml"
    build_scene(output_xml=out)
    model = mujoco.MjModel.from_xml_path(str(out))

    handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handle_grip")
    femur_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_femur1_col")
    thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "distal_thumb_r_coll_2")
    proximal_thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proximal_thumb_r_coll")
    assert handle_id >= 0
    assert femur_id >= 0
    assert thumb_id >= 0
    assert proximal_thumb_id >= 0
    assert int(model.geom_contype[handle_id]) == 0
    assert int(model.geom_conaffinity[handle_id]) == 0
    for index in range(8):
        bevel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"handle_bevel_{index:02d}")
        assert bevel_id >= 0
        assert int(model.geom_contype[bevel_id]) == 16
        assert int(model.geom_conaffinity[bevel_id]) == 0
        assert int(model.geom_condim[bevel_id]) == 4
    assert int(model.geom_conaffinity[thumb_id]) & 16
    assert not (int(model.geom_conaffinity[proximal_thumb_id]) & 16)
    assert not (int(model.geom_conaffinity[femur_id]) & 16)


def test_build_grip_scene_is_repeat_call_deterministic(tmp_path):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    script = f"""
from pathlib import Path
from src.grip.build_right_hand_racket_grip_scene import build_scene

first = Path({str(first)!r})
second = Path({str(second)!r})
build_scene(output_xml=first)
build_scene(output_xml=second)
if first.read_bytes() != second.read_bytes():
    raise SystemExit("same-process builds produced different XML bytes")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_grip_scene_omits_absolute_venv_asset_paths(tmp_path):
    from src.grip.build_right_hand_racket_grip_scene import build_scene

    out = tmp_path / "grip_scene.xml"
    build_scene(output_xml=out)
    text = out.read_text(encoding="utf-8")
    assert "/data3/" not in text
    assert ".venv" not in text

    root = ET.parse(out).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    for attr in ("meshdir", "texturedir"):
        value = compiler.attrib.get(attr)
        if value is not None:
            assert not Path(value).is_absolute()


def test_angle_between_vectors_returns_degrees():
    assert angle_between_vectors(np.array([1, 0, 0]), np.array([0, 1, 0])) == 90.0


def test_normalized_zero_vector_returns_zeros():
    assert np.allclose(normalized(np.array([0.0, 0.0, 0.0])), np.zeros(3))


def test_normalized_rejects_invalid_eps():
    for eps in (0.0, -1e-9, float("nan")):
        with pytest.raises(ValueError, match="eps"):
            normalized(np.array([1.0, 0.0, 0.0]), eps=eps)


def test_weighted_site_target_residuals_and_mean_error():
    current = {"palm": np.array([0.0, 0.0, 0.0]), "thumb": np.array([1.0, 0.0, 0.0])}
    target = {"palm": np.array([0.0, 0.0, 0.0]), "thumb": np.array([0.5, 0.0, 0.0])}
    weights = {"palm": 1.0, "thumb": 2.0}
    residuals = weighted_site_target_residuals(current, target, weights)
    assert residuals.shape == (6,)
    assert np.allclose(residuals[-3:], np.array([1.0, 0.0, 0.0]))
    assert mean_site_error(current, target) == 0.25


def test_weighted_site_target_residuals_defaults_missing_weight_to_one():
    current = {"thumb": np.array([1.0, 0.0, 0.0])}
    target = {"thumb": np.array([0.5, 0.0, 0.0])}
    residuals = weighted_site_target_residuals(current, target, {})
    assert np.allclose(residuals, np.array([0.5, 0.0, 0.0]))


def test_mean_site_error_empty_targets_returns_zero():
    assert mean_site_error({}, {}) == 0.0


def test_joint_limit_margin_cost_empty_ranges_returns_zero():
    assert joint_limit_margin_cost(np.array([]), np.empty((0, 2))) == 0.0


def test_joint_limit_margin_cost_rejects_non_empty_qpos_with_empty_ranges():
    with pytest.raises(ValueError, match="ranges"):
        joint_limit_margin_cost(np.array([0.0]), np.empty((0, 2)))


def test_joint_limit_margin_cost_rejects_invalid_empty_range_shape():
    with pytest.raises(ValueError, match="ranges"):
        joint_limit_margin_cost(np.array([]), np.array([]))


def test_joint_limit_margin_cost_rejects_non_finite_qpos_or_margin():
    with pytest.raises(ValueError, match="finite"):
        joint_limit_margin_cost(np.array([float("nan")]), np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="margin"):
        joint_limit_margin_cost(np.array([0.5]), np.array([[0.0, 1.0]]), margin=float("nan"))


def test_collect_site_positions_from_generated_scene(tmp_path):
    out = tmp_path / "grip_scene.xml"
    build_scene(out)
    positions = collect_site_positions(out)
    assert "rh_palm_grip_site" in positions
    assert "grip_pose_site" in positions
    assert positions["rh_palm_grip_site"].shape == (3,)


def test_target_config_default_path_is_repo_level_configs():
    path = target_config_path()
    assert path == REPO_ROOT / "configs" / "right_hand_racket_grip_targets.json"
    assert path.parent.is_dir()


def test_load_default_grip_targets():
    config = load_grip_target_config()
    assert isinstance(config, GripTargetConfig)
    assert config.handle_radius_m == 0.0132
    assert set(config.target_points_racket_local) == {"palm", "thumb", "index", "middle", "ring", "pinky"}


def test_solve_reference_writes_dimensionally_valid_json(tmp_path):
    scene = tmp_path / "grip_scene.xml"
    reference = tmp_path / "reference.json"
    build_scene(scene)
    result = solve_reference(scene, target_config_path(), reference, max_nfev=2)
    assert reference.is_file()
    raw = json.loads(reference.read_text())
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert len(raw["qpos"]) == model.nq
    assert len(raw["qvel"]) == model.nv
    assert len(raw["racket_freejoint_qpos"]) == 7
    assert "site_errors_m" in raw
    assert result["mean_site_error_m"] >= 0.0


def test_grip_env_reset_and_step_returns_finite_reward(tmp_path):
    env = _build_smoke_env(tmp_path)
    obs, info = env.reset()
    assert obs.ndim == 1
    assert info["mean_site_error_m"] >= 0.0
    action = np.zeros(env.action_size, dtype=float)
    obs, reward, terminated, truncated, info = env.step(action)
    assert np.isfinite(reward)
    assert "reward_terms" in info
    assert terminated is False
    assert truncated is False


def test_grip_env_reports_filtered_and_raw_contacts(tmp_path):
    env = _build_smoke_env(tmp_path)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert "raw_contact_count" in info
    assert "illegal_handle_contact_count" in info
    assert "max_handle_penetration_m" in info
    assert info["contact_count"] <= info["raw_contact_count"]
    assert info["illegal_handle_contact_count"] >= 0
    assert info["max_handle_penetration_m"] >= 0.0


def test_grip_env_contact_reward_uses_filtered_contacts(tmp_path):
    env = _build_smoke_env(tmp_path)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    reward_terms = info["reward_terms"]
    assert reward_terms["r_contact"] <= 1.0
    if info["raw_contact_count"] > info["contact_count"]:
        expected_contact_reward = env.reward_weights["contact"] * min(info["contact_count"] / 4.0, 1.0)
        assert reward_terms["r_contact"] == pytest.approx(expected_contact_reward)


def test_grip_env_reward_terms_include_configured_and_planned_terms(tmp_path):
    env = _build_smoke_env(tmp_path)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert set(info["reward_terms"]) == {
        "r_site_match",
        "r_v_shape",
        "r_anti_panhandle",
        "r_anti_thumb_grip",
        "r_racket_pose",
        "r_racket_orient",
        "r_contact",
        "r_no_slip",
        "r_reference_pose",
        "r_effort",
        "r_joint_limits",
        "r_no_penetration",
    }


def test_grip_env_reward_terms_use_real_pose_and_contact_diagnostics(tmp_path):
    env = _build_smoke_env(tmp_path)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    for key in (
        "v_shape_error",
        "anti_panhandle_error",
        "anti_thumb_grip_error",
        "thumb_index_y_gap_m",
        "v_bisector_theta_deg",
        "palm_theta_deg",
        "thumb_theta_deg",
        "racket_translation_error_m",
        "racket_orientation_error_deg",
        "grip_slip_m",
        "reference_pose_error",
        "joint_limit_margin_cost",
    ):
        assert math.isfinite(info[key])
    for key in (
        "v_shape_error",
        "anti_panhandle_error",
        "anti_thumb_grip_error",
        "racket_translation_error_m",
        "racket_orientation_error_deg",
        "grip_slip_m",
        "reference_pose_error",
        "joint_limit_margin_cost",
    ):
        assert info[key] >= 0.0

    reward_terms = info["reward_terms"]
    assert reward_terms["r_v_shape"] == pytest.approx(-env.reward_weights["v_shape"] * info["v_shape_error"])
    assert reward_terms["r_anti_panhandle"] == pytest.approx(
        -env.reward_weights["anti_panhandle"] * info["anti_panhandle_error"],
    )
    assert reward_terms["r_anti_thumb_grip"] == pytest.approx(
        -env.reward_weights["anti_thumb_grip"] * info["anti_thumb_grip_error"],
    )
    assert reward_terms["r_racket_pose"] == pytest.approx(
        -env.reward_weights["racket_pose"] * info["racket_translation_error_m"],
    )
    assert reward_terms["r_racket_orient"] == pytest.approx(
        -env.reward_weights["racket_orient"] * info["racket_orientation_error_deg"] / 180.0,
    )
    assert reward_terms["r_no_slip"] == pytest.approx(-env.reward_weights["no_slip"] * info["grip_slip_m"])
    assert reward_terms["r_reference_pose"] == pytest.approx(
        -env.reward_weights["reference_pose"] * info["reference_pose_error"],
    )
    assert reward_terms["r_joint_limits"] == pytest.approx(
        -env.reward_weights["joint_limits"] * info["joint_limit_margin_cost"],
    )
    assert reward_terms["r_no_penetration"] == pytest.approx(
        -env.reward_weights["no_penetration"] * info["max_handle_penetration_m"],
    )


def test_evaluate_right_hand_racket_grip_returns_finite_metrics(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)

    metrics = evaluate(scene, targets, reference, episodes=1, steps=2)

    assert metrics["episodes"] == 1
    assert metrics["steps_executed"] > 0
    assert metrics["finite"] is True
    for key in (
        "mean_reward",
        "mean_site_error_m",
        "contact_count",
        "illegal_handle_contact_count",
        "max_handle_penetration_m",
        "raw_contact_count",
        "translation_drift_m",
        "orientation_drift_deg",
        "recovery_mean_site_error_m",
        "recovery_orientation_drift_deg",
    ):
        assert math.isfinite(metrics[key])
    assert "reward_terms_mean" in metrics
    assert "site_errors_m" in metrics
    assert "pass" in metrics


def test_evaluate_accepts_episodes_and_steps_as_positional_args(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)

    metrics = evaluate(scene, targets, reference, 1, 2)

    assert metrics["episodes"] == 1
    assert metrics["steps_requested"] == 2
    assert metrics["finite"] is True


def test_run_baseline_writes_json_metrics(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)
    out = tmp_path / "baseline_metrics.json"

    metrics = run_baseline(scene, targets, reference, out, steps=2)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == metrics
    assert written["mode"] == "zero_action_reference_hold"
    assert written["finite"] is True


def test_train_policy_writes_checkpoint_and_metrics(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)
    out = tmp_path / "policy"

    metrics = train_policy(scene, targets, reference, out_dir=out, total_steps=4, rollout_steps=2, seed=0)

    assert metrics["mode"] == "ppo_right_hand_racket_grip"
    assert metrics["global_step"] == 4
    assert (out / "policy_latest.pt").is_file()
    assert (out / "metrics.json").is_file()


def test_validate_grip_reports_real_racket_drift_pass_booleans(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)

    metrics = validate_grip(scene, targets, reference, steps=2)

    assert {
        "mean_site_error_m",
        "translation_drift_m",
        "orientation_drift_deg",
        "contact_count",
        "illegal_handle_contact_count",
        "max_handle_penetration_m",
        "finite",
        "recovery_mean_site_error_m",
        "recovery_orientation_drift_deg",
    }.issubset(metrics)
    assert {
        "mean_site_error_m",
        "translation_drift_m",
        "orientation_drift_deg",
        "contact_count",
        "illegal_handle_contact_count",
        "max_handle_penetration_m",
        "finite",
        "recovery_mean_site_error_m",
        "recovery_orientation_drift_deg",
    }.issubset(metrics["pass"])
    assert math.isfinite(metrics["translation_drift_m"])
    assert math.isfinite(metrics["orientation_drift_deg"])
    assert math.isfinite(metrics["recovery_mean_site_error_m"])
    assert math.isfinite(metrics["recovery_orientation_drift_deg"])
    assert isinstance(metrics["pass"]["translation_drift_m"], bool)
    assert isinstance(metrics["pass"]["orientation_drift_deg"], bool)
    assert isinstance(metrics["pass"]["illegal_handle_contact_count"], bool)
    assert isinstance(metrics["pass"]["max_handle_penetration_m"], bool)
    assert isinstance(metrics["pass"]["recovery_mean_site_error_m"], bool)


def test_validate_grip_reports_all_acceptance_thresholds(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)

    metrics = validate_grip(scene, targets, reference, steps=1)

    assert set(metrics["thresholds"]) == {
        "max_mean_site_error_m",
        "max_racket_translation_drift_m_2s",
        "max_racket_orientation_drift_deg_2s",
        "min_handle_contacts",
        "perturb_force_n",
        "perturb_torque_nm",
        "perturb_recovery_s",
        "max_recovery_site_error_m",
        "max_recovery_orientation_error_deg",
    }


def test_validate_grip_rejects_invalid_acceptance_threshold_types_and_domains(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    invalid_overrides = [
        {"max_mean_site_error_m": True},
        {"max_racket_translation_drift_m_2s": "0.1"},
        {"max_racket_orientation_drift_deg_2s": -1.0},
        {"min_handle_contacts": 1.5},
        {"min_handle_contacts": -1},
        {"perturb_force_n": "2.0"},
        {"perturb_torque_nm": True},
        {"perturb_recovery_s": -0.1},
        {"max_recovery_site_error_m": float("nan")},
        {"max_recovery_orientation_error_deg": -1.0},
    ]

    for override in invalid_overrides:
        targets = _write_acceptance_override(tmp_path, **override)
        with pytest.raises(ValueError, match="training_acceptance"):
            validate_grip(scene, targets, reference, steps=1)


def test_validate_grip_respects_zero_recovery_duration(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    targets = _write_acceptance_override(tmp_path, perturb_recovery_s=0.0)

    metrics = validate_grip(scene, targets, reference, steps=1)

    assert metrics["thresholds"]["perturb_recovery_s"] == 0.0
    assert metrics["recovery_steps_executed"] == 0


def test_validate_grip_direct_cli_prints_json(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "src/grip/validate_right_hand_racket_grip.py",
            "--xml",
            str(scene),
            "--targets",
            str(targets),
            "--reference",
            str(reference),
            "--steps",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "pass" in payload
    assert "False" not in result.stdout


def test_validate_grip_module_cli_prints_json(tmp_path):
    scene, targets, reference = _build_smoke_paths(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.grip.validate_right_hand_racket_grip",
            "--xml",
            str(scene),
            "--targets",
            str(targets),
            "--reference",
            str(reference),
            "--steps",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["finite"] is True
    assert "pass" in payload


def test_validate_grip_strict_only_fails_on_acceptance_failure(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    targets = _write_impossible_acceptance_config(tmp_path)
    base_command = [
        sys.executable,
        "-m",
        "src.grip.validate_right_hand_racket_grip",
        "--xml",
        str(scene),
        "--targets",
        str(targets),
        "--reference",
        str(reference),
        "--steps",
        "1",
    ]

    default_result = subprocess.run(
        base_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    strict_result = subprocess.run(
        [*base_command, "--strict"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    default_payload = json.loads(default_result.stdout)
    strict_payload = json.loads(strict_result.stdout)
    assert default_payload["finite"] is True
    assert default_payload["acceptance_pass"] is False
    assert default_result.returncode == 0
    assert strict_payload["finite"] is True
    assert strict_payload["acceptance_pass"] is False
    assert strict_result.returncode == 2


def test_grip_env_exposes_non_empty_contact_geom_sets(tmp_path):
    env = _build_smoke_env(tmp_path)
    geom_sets = env.contact_geom_id_sets()

    assert geom_sets["handle"]
    assert geom_sets["right_hand"]


def test_solve_reference_meets_ik_quality_and_reports_recomputable_errors(tmp_path):
    scene = tmp_path / "grip_scene.xml"
    reference = tmp_path / "reference.json"
    build_scene(scene)

    result = solve_reference(scene, target_config_path(), reference, max_nfev=200)
    raw = json.loads(reference.read_text())

    assert result["mean_site_error_m"] < 0.03
    assert result["meets_ik_mean_threshold"] is True
    assert raw["meets_ik_mean_threshold"] is True
    assert len(raw["racket_freejoint_qpos"]) == 7

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model_map = load_model_map(model)
    qpos = np.array(raw["qpos"], dtype=float)
    current_sites = hand_site_positions(model, data, qpos, model_map)
    target_sites = racket_local_targets_to_world(model, data, qpos, load_grip_target_config(target_config_path()), model_map)
    recomputed_errors = {
        name: float(np.linalg.norm(current_sites[name] - target_sites[name]))
        for name in sorted(target_sites)
    }

    assert recomputed_errors == pytest.approx(raw["site_errors_m"], abs=1e-9)
    assert float(np.mean(list(recomputed_errors.values()))) == pytest.approx(raw["objective_breakdown"]["mean_site_error_m"])


def test_quality_threshold_comparison_is_inclusive():
    assert mean_error_meets_threshold(0.03, 0.03) is True
    assert mean_error_meets_threshold(0.0300001, 0.03) is False
    assert mean_error_meets_threshold(0.03, None) is None


def test_quality_exit_code_requires_override_for_poor_reference():
    poor_result = {"meets_ik_mean_threshold": False}
    good_result = {"meets_ik_mean_threshold": True}

    assert quality_exit_code(poor_result, allow_poor_reference=False) == 2
    assert quality_exit_code(poor_result, allow_poor_reference=True) == 0
    assert quality_exit_code(good_result, allow_poor_reference=False) == 0


def test_default_myofullbody_contains_right_hand_finger_joints_and_muscles():
    model = mujoco.MjModel.from_xml_path(str(musclemimic_models.get_xml_path("myofullbody")))
    model_map = load_model_map(model, require_racket=False, require_grip_sites=False)
    assert model_map.ok
    assert model_map.hand_bodies["palm"] == "lunate_r"
    assert model_map.hand_bodies["thumb"] == "distal_thumb_r"
    assert model_map.hand_bodies["index"] == "2distph_r"
    assert model_map.hand_bodies["middle"] == "3distph_r"
    assert model_map.hand_bodies["ring"] == "4distph_r"
    assert model_map.hand_bodies["pinky"] == "5distph_r"
    for joint_name in (
        "cmc_flexion_r",
        "mcp2_flexion_r",
        "mp_flexion_r",
        "ip_flexion_r",
        "pm2_flexion_r",
        "md2_flexion_r",
        "pm3_flexion_r",
        "md3_flexion_r",
        "pm4_flexion_r",
        "md4_flexion_r",
        "pm5_flexion_r",
        "md5_flexion_r",
    ):
        assert joint_name in model_map.right_hand_joint_names
    assert "FDS2" in model_map.right_hand_actuator_names
    assert "FPL" in model_map.right_hand_actuator_names


def test_target_point_conversion_uses_racket_local_cylinder():
    config = load_grip_target_config()
    palm = config.target_xyz("palm")
    assert palm[1] == 0.086
    assert math.isclose(palm[0], -0.0132, abs_tol=1e-9)
    assert math.isclose(palm[2], 0.0, abs_tol=1e-9)


def test_rejects_missing_top_level_key(tmp_path):
    raw = _default_raw_config()
    del raw["handle_radius_m"]

    with pytest.raises(ValueError, match=r"missing top-level key.*handle_radius_m"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_non_object_target_points(tmp_path):
    raw = _default_raw_config()
    raw["target_points_racket_local"] = []

    with pytest.raises(ValueError, match=r"target_points_racket_local.*object"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_missing_required_target(tmp_path):
    raw = _default_raw_config()
    del raw["target_points_racket_local"]["pinky"]

    with pytest.raises(ValueError, match=r"missing required target.*pinky"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_missing_point_field(tmp_path):
    raw = _default_raw_config()
    del raw["target_points_racket_local"]["thumb"]["weight"]

    with pytest.raises(ValueError, match=r"thumb.*missing.*weight"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_non_finite_numeric_value(tmp_path):
    raw = _default_raw_config()
    raw["target_points_racket_local"]["index"]["theta_deg"] = float("nan")

    with pytest.raises(ValueError, match=r"index\.theta_deg.*finite"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_invalid_radius(tmp_path):
    raw = _default_raw_config()
    raw["handle_radius_m"] = 0

    with pytest.raises(ValueError, match=r"handle_radius_m.*> 0"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_invalid_weight(tmp_path):
    raw = _default_raw_config()
    raw["target_points_racket_local"]["middle"]["weight"] = -1

    with pytest.raises(ValueError, match=r"middle\.weight.*> 0"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_string_numeric_value(tmp_path):
    raw = _default_raw_config()
    raw["handle_radius_m"] = "0.014"

    with pytest.raises(ValueError, match=r"handle_radius_m.*JSON number"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_rejects_boolean_numeric_value(tmp_path):
    raw = _default_raw_config()
    raw["target_points_racket_local"]["thumb"]["weight"] = True

    with pytest.raises(ValueError, match=r"thumb\.weight.*JSON number"):
        load_grip_target_config(_write_config(tmp_path, raw))


def test_grip_training_config_includes_default_off_swing_disturbance():
    from src.grip.right_hand_racket_grip_env import load_training_config

    cfg = load_training_config("configs/right_hand_racket_grip_training.yaml")

    assert cfg["swing_disturbance"]["enabled"] is False
    assert cfg["swing_disturbance"]["force_scale_n"] == 0.0
    assert cfg["swing_disturbance"]["torque_scale_nm"] == 0.0
    assert cfg["swing_disturbance"]["phase_start"] == 0.0
    assert cfg["swing_disturbance"]["phase_end"] == 1.0
    assert "perturbation" in cfg
    assert "ppo" in cfg
    assert "curriculum" in cfg


def test_swing_disturbance_profile_is_zero_outside_phase_window():
    from src.grip.right_hand_racket_grip_env import swing_disturbance_profile

    force, torque = swing_disturbance_profile(
        phase=0.1,
        phase_start=0.4,
        phase_end=0.6,
        force_scale_n=2.0,
        torque_scale_nm=0.03,
    )

    assert force.tolist() == [0.0, 0.0, 0.0]
    assert torque.tolist() == [0.0, 0.0, 0.0]


def test_swing_disturbance_profile_peaks_inside_phase_window():
    from src.grip.right_hand_racket_grip_env import swing_disturbance_profile

    force, torque = swing_disturbance_profile(
        phase=0.5,
        phase_start=0.4,
        phase_end=0.6,
        force_scale_n=2.0,
        torque_scale_nm=0.03,
    )

    assert force.tolist() == [2.0, 0.0, 0.0]
    assert torque.tolist() == [0.0, 0.03, 0.0]
