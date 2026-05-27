from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from environment.overall_environment.src.build_overall_environment import build_overall_scene
from environment.overall_environment.src.overall_env import (
    OverallBadmintonEnvironment,
    _configure_viewer_visuals,
    _parse_args,
)
from environment.overall_environment.src.paths import (
    court_xml_path,
    default_overall_scene_path,
    racket_xml_path,
    shuttlecock_xml_path,
)


def _name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, obj_type, name)


def test_overall_paths_point_to_existing_assets():
    assert court_xml_path().is_file()
    assert racket_xml_path().is_file()
    assert shuttlecock_xml_path().is_file()
    assert "environment/overall_environment/assets/overall_badminton_scene.xml" in str(
        default_overall_scene_path()
    )


def test_build_overall_scene_loads_court_person_racket_and_shuttle(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"

    build_overall_scene(out)

    model = mujoco.MjModel.from_xml_path(str(out))
    assert _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_court_static") >= 0
    assert _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket") >= 0
    assert _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_shuttle") >= 0
    assert _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_racket_free") >= 0
    assert _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free") >= 0
    assert _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready") >= 0
    assert _name_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view") >= 0


def test_build_overall_scene_preserves_musculoskeletal_visual_assets(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"

    build_overall_scene(out)

    xml_text = out.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(out))
    assert 'meshdir="mimic_msk_model"' in xml_text
    assert 'file="meshes/hat_skull.stl"' in xml_text
    assert 'type="mesh"' in xml_text
    assert (out.parent / "mimic_msk_model" / "meshes" / "hat_skull.stl").is_file()
    assert (out.parent / "mimic_msk_model" / "scene" / "musclemimic_skybox.png").is_file()
    assert model.nmesh > 100
    assert model.ntendon > 400


def test_ready_key_places_shuttle_on_ground_and_racket_near_right_hand(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)

    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    cork_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_cork_contact_site")
    assert 0.0 <= float(data.site_xpos[cork_site, 2]) <= 0.035

    palm_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    palm_to_grip = np.linalg.norm(data.site_xpos[palm_site] - data.site_xpos[grip_site])
    assert palm_to_grip < 0.12


def test_overall_environment_reset_reports_expected_scene_objects(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    env = OverallBadmintonEnvironment(out)

    obs, info = env.reset()

    assert obs.shape == (env.model.nq + env.model.nv,)
    assert info["keyframe"] == "overall_ready"
    assert info["has_court"] is True
    assert info["has_racket"] is True
    assert info["has_shuttlecock"] is True
    assert info["shuttle_cork_height_m"] >= 0.0


def test_overall_environment_pose_servo_simulation_remains_finite(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    env = OverallBadmintonEnvironment(out)
    env.reset()

    for _ in range(50):
        obs, info = env.step(pose_servo=True)

    assert np.isfinite(obs).all()
    assert info["has_racket"] is True
    assert info["has_shuttlecock"] is True


def test_overall_env_cli_accepts_viewer_flag():
    args = _parse_args(["--viewer"])

    assert args.viewer is True
    assert args.simulate is False


def test_overall_env_cli_requires_explicit_simulation():
    args = _parse_args(["--viewer", "--simulate"])

    assert args.viewer is True
    assert args.simulate is True
    assert args.free_simulate is False
    assert args.pose_servo is False


def test_overall_env_cli_accepts_native_viewer_flag():
    args = _parse_args(["--native-viewer"])

    assert args.native_viewer is True


def test_overall_viewer_visuals_default_to_musculoskeletal_groups():
    class Viewer:
        opt = mujoco.MjvOption()

    viewer = Viewer()
    viewer.opt.geomgroup[:] = 1
    viewer.opt.sitegroup[:] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1

    _configure_viewer_visuals(viewer)

    assert viewer.opt.geomgroup[:6].tolist() == [1, 1, 1, 0, 0, 0]
    assert viewer.opt.sitegroup[:6].tolist() == [1, 1, 1, 0, 0, 0]
    assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] == 1
    assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] == 1
    assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == 0
    assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] == 0


def test_overall_viewer_debug_visuals_enable_debug_layers():
    class Viewer:
        opt = mujoco.MjvOption()

    viewer = Viewer()
    viewer.opt.geomgroup[:] = 0
    viewer.opt.sitegroup[:] = 0

    _configure_viewer_visuals(viewer, debug_visuals=True)

    assert viewer.opt.geomgroup[:6].tolist() == [1, 1, 1, 1, 1, 1]
    assert viewer.opt.sitegroup[:6].tolist() == [1, 1, 1, 1, 1, 1]
    assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] == 1
    assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == 1


def test_overall_env_runs_as_portable_direct_script():
    script = Path("environment/overall_environment/src/overall_env.py")
    xml = Path("environment/overall_environment/assets/overall_badminton_scene.xml")

    result = subprocess.run(
        [sys.executable, str(script), "--xml", str(xml)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"has_court": true' in result.stdout
    assert '"has_racket": true' in result.stdout
    assert '"has_shuttlecock": true' in result.stdout
