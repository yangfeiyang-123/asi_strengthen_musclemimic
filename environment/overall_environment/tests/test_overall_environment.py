from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environment.overall_environment.src.build_overall_environment import (
    HAND_GRIP_SITES as OVERALL_HAND_GRIP_SITES,
    build_overall_scene,
)
from environment.overall_environment.src.overall_env import (
    OverallBadmintonEnvironment,
    _configure_viewer_visuals,
    _parse_args,
)
from environment.overall_environment.src.paths import (
    court_xml_path,
    default_overall_scene_path,
    default_overall_training_scene_path,
    racket_xml_path,
    shuttlecock_xml_path,
)
from src.grip.build_right_hand_racket_grip_scene import HAND_GRIP_SITES as GRIP_HAND_GRIP_SITES
from src.grip.build_right_hand_racket_grip_seed import build_grip_seed
from src.grip.grip_seed import load_grip_seed
from src.grip.paths import grip_seed_json_path


def _name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, obj_type, name)


def _has_fullbody_racket_exclude(xml_path: Path) -> bool:
    root = ET.parse(xml_path).getroot()
    for exclude in root.findall("./contact/exclude"):
        if {exclude.attrib.get("body1"), exclude.attrib.get("body2")} == {
            "Full Body",
            "overall_racket",
        }:
            return True
    return False


def test_overall_paths_point_to_existing_assets():
    assert court_xml_path().is_file()
    assert racket_xml_path().is_file()
    assert shuttlecock_xml_path().is_file()
    assert "environment/overall_environment/assets/overall_badminton_scene.xml" in str(
        default_overall_scene_path()
    )
    assert "environment/overall_environment/assets/overall_badminton_training_scene.xml" in str(
        default_overall_training_scene_path()
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
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION
    assert _has_fullbody_racket_exclude(out)


def test_build_training_scene_has_actuation_and_hand_racket_contact(tmp_path):
    out = tmp_path / "overall_badminton_training_scene.xml"

    build_overall_scene(out, mode="training")

    model = mujoco.MjModel.from_xml_path(str(out))
    assert model.nu > 0
    assert not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    assert not _has_fullbody_racket_exclude(out)


def test_build_training_scene_can_enable_soft_weld(tmp_path):
    out = tmp_path / "overall_badminton_training_scene.xml"

    build_overall_scene(out, mode="training", enable_soft_weld=True)

    model = mujoco.MjModel.from_xml_path(str(out))
    weld_id = _name_id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        "overall_right_hand_racket_soft_weld",
    )
    assert weld_id >= 0


def test_training_scene_pose_servo_simulation_remains_finite(tmp_path):
    out = tmp_path / "overall_badminton_training_scene.xml"
    build_overall_scene(out, mode="training")
    env = OverallBadmintonEnvironment(out)
    env.reset()

    for _ in range(100):
        obs, info = env.step(pose_servo=True)

    assert np.isfinite(obs).all()
    assert info["has_racket"] is True
    assert info["has_shuttlecock"] is True


def test_build_overall_scene_uses_standard_octagonal_racket_handle(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"

    build_overall_scene(out)

    model = mujoco.MjModel.from_xml_path(str(out))
    handle_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "overall_handle_grip")
    assert handle_id >= 0
    assert int(model.geom_contype[handle_id]) == 0
    assert int(model.geom_conaffinity[handle_id]) == 0
    assert float(model.geom_rgba[handle_id, 3]) <= 0.05

    for index in range(8):
        bevel_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"overall_handle_bevel_{index:02d}")
        assert bevel_id >= 0
        assert int(model.geom_type[bevel_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        assert int(model.geom_contype[bevel_id]) == 4
        assert int(model.geom_conaffinity[bevel_id]) == 4
        assert float(model.geom_rgba[bevel_id, 3]) == pytest.approx(1.0)


def test_initial_pose_faces_the_net(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"

    build_overall_scene(out)

    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    root_joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_adr = int(model.jnt_qposadr[root_joint])
    root_xy = data.qpos[root_adr : root_adr + 2]
    net_xy = np.array([0.0, 0.0])
    root_to_net = net_xy - root_xy
    root_to_net = root_to_net / np.linalg.norm(root_to_net)

    right_shoulder = data.xpos[_name_id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus_r")]
    left_shoulder = data.xpos[_name_id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus_l")]
    shoulder_lateral = left_shoulder - right_shoulder
    shoulder_lateral[2] = 0.0
    shoulder_lateral = shoulder_lateral / np.linalg.norm(shoulder_lateral)
    anatomical_forward = np.cross(shoulder_lateral, np.array([0.0, 0.0, 1.0]))[:2]
    anatomical_forward = anatomical_forward / np.linalg.norm(anatomical_forward)

    assert float(np.dot(anatomical_forward, root_to_net)) > 0.95


def test_build_overall_scene_preserves_musculoskeletal_visual_assets(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"

    build_overall_scene(out)

    xml_text = out.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(out))
    assert 'meshdir="mimic_msk_model"' in xml_text
    assert 'file="meshes/hat_skull.stl"' in xml_text
    assert 'type="mesh"' in xml_text
    assert (out.parent / "mimic_msk_model" / "meshes" / "hat_skull.stl").is_file()
    assert "skybox" not in xml_text
    assert not (out.parent / "mimic_msk_model" / "scene").exists()
    assert model.nmesh > 100
    assert model.ntendon > 400


def test_initial_pose_places_shuttle_on_ground_and_racket_in_right_hand(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)

    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_forward(model, data)

    cork_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_cork_contact_site")
    assert 0.0 <= float(data.site_xpos[cork_site, 2]) <= 0.035

    shuttle_com_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_shuttle_com")
    shuttle_nose_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_shuttle_nose")
    nose_vector = data.site_xpos[shuttle_nose_site] - data.site_xpos[shuttle_com_site]
    assert abs(float(nose_vector[2])) < 1e-6
    assert np.linalg.norm(nose_vector[:2]) > 0.02

    palm_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    palm_to_grip = np.linalg.norm(data.site_xpos[palm_site] - data.site_xpos[grip_site])
    assert 0.04 < palm_to_grip < 0.08
    assert np.isclose(model.key_qpos[key_id, 0], -2.5)

    racket_joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_racket_free")
    shuttle_joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    racket_adr = int(model.jnt_qposadr[racket_joint])
    shuttle_adr = int(model.jnt_qposadr[shuttle_joint])
    assert model.key_qpos[key_id, racket_adr] < 0.0
    assert model.key_qpos[key_id, shuttle_adr] < 0.0


def test_overall_grip_sites_match_standalone_grip_reference():
    assert OVERALL_HAND_GRIP_SITES == GRIP_HAND_GRIP_SITES


def test_overall_ready_uses_default_right_hand_grip_seed(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    model = mujoco.MjModel.from_xml_path(str(out))
    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    seed = load_grip_seed(grip_seed_json_path())
    seed_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))

    for joint_name in seed.right_hand_joint_names:
        reference_joint = _name_id(seed_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        overall_joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert reference_joint >= 0
        assert overall_joint >= 0
        reference_adr = int(seed_model.jnt_qposadr[reference_joint])
        overall_adr = int(model.jnt_qposadr[overall_joint])
        assert np.isclose(model.key_qpos[key_id, overall_adr], seed.qpos[reference_adr])


def test_overall_ready_preserves_seed_hand_to_racket_grip_distances(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    seed = load_grip_seed(grip_seed_json_path())
    seed_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))
    seed_data = mujoco.MjData(seed_model)
    seed_data.qpos[:] = seed.qpos
    seed_data.qvel[:] = seed.qvel
    mujoco.mj_forward(seed_model, seed_data)

    hand_sites = (
        "rh_palm_grip_site",
        "rh_thumb_pad_site",
        "rh_index_pad_site",
        "rh_middle_pad_site",
        "rh_ring_pad_site",
        "rh_pinky_pad_site",
    )
    seed_grip = _name_id(seed_model, mujoco.mjtObj.mjOBJ_SITE, "grip_pose_site")
    overall_grip = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")

    for site_name in hand_sites:
        seed_site = _name_id(seed_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        overall_site = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        seed_distance = np.linalg.norm(seed_data.site_xpos[seed_site] - seed_data.site_xpos[seed_grip])
        overall_distance = np.linalg.norm(data.site_xpos[overall_site] - data.site_xpos[overall_grip])
        assert overall_distance == pytest.approx(seed_distance, abs=1e-6)


def test_overall_ready_can_use_explicit_grip_seed(tmp_path):
    from src.grip.build_right_hand_racket_grip_scene import build_scene
    from src.grip.paths import target_config_path
    from src.grip.solve_right_hand_racket_grip import solve_reference

    grip_scene = tmp_path / "grip_scene.xml"
    reference = tmp_path / "reference.json"
    seed_path = tmp_path / "right_hand_racket_grip_seed.json"
    overall = tmp_path / "overall_badminton_scene.xml"

    build_scene(grip_scene)
    solve_reference(grip_scene, target_config_path(), reference, max_nfev=2)
    build_grip_seed(
        grip_scene,
        target_config_path(),
        seed_path,
        initial_reference=reference,
        max_nfev=4,
        render=False,
    )
    build_overall_scene(overall, grip_seed=seed_path)

    model = mujoco.MjModel.from_xml_path(str(overall))
    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    seed = load_grip_seed(seed_path)
    seed_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))
    for joint_name in seed.right_hand_joint_names:
        seed_joint = _name_id(seed_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        overall_joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        seed_adr = int(seed_model.jnt_qposadr[seed_joint])
        overall_adr = int(model.jnt_qposadr[overall_joint])
        assert model.key_qpos[key_id, overall_adr] == pytest.approx(seed.qpos[seed_adr])


def test_initial_pose_has_no_net_or_hand_racket_penetration(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    contact_names = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        contact_names.append((geom1, geom2))

    assert not any("overall_net" in name for pair in contact_names for name in pair)
    assert not any("overall_handle_grip" in name for pair in contact_names for name in pair)
    assert np.max(np.abs(data.qfrc_actuator)) == 0.0


def test_generated_scene_has_shuttle_support_and_matte_court_materials(tmp_path):
    out = tmp_path / "overall_badminton_scene.xml"
    build_overall_scene(out)
    root = ET.parse(out).getroot()

    support_geom = root.find(".//geom[@name='overall_skirt_ground_support']")
    assert support_geom is not None
    assert support_geom.attrib["type"] == "ellipsoid"
    assert support_geom.attrib["group"] == "3"

    materials = {
        material.attrib["name"]: material
        for material in root.findall("./asset/material")
        if "name" in material.attrib
    }
    assert materials["overall_mat_floor"].attrib["rgba"] == "0.015 0.34 0.14 1"
    assert materials["overall_mat_floor"].attrib["reflectance"] == "0"
    assert materials["overall_mat_floor"].attrib["specular"] == "0"
    assert materials["MatPlane"].attrib["rgba"] == "0.13 0.13 0.13 1"
    assert materials["MatPlane"].attrib["reflectance"] == "0"
    assert "texture" not in materials["MatPlane"].attrib


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
