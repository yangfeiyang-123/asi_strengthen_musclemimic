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
import pytest

from src.grip.build_right_hand_racket_grip_scene import build_scene
from src.grip.hand_racket_model_map import load_model_map
from src.grip.paths import REPO_ROOT, racket_xml_path, scene_xml_path, target_config_path
from src.grip.target_config import GripTargetConfig, load_grip_target_config
from src.grip.visualize_grip_sites import collect_site_positions


def _default_raw_config():
    return copy.deepcopy(load_grip_target_config().raw)


def _write_config(tmp_path, raw):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


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
    assert config.handle_radius_m == 0.014
    assert set(config.target_points_racket_local) == {"palm", "thumb", "index", "middle", "ring", "pinky"}


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
    assert palm[1] == 0.085
    assert math.isclose(palm[0], -0.014, abs_tol=1e-9)
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
