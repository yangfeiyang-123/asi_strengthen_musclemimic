from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402

SCENE_XML = default_incoming_scene_path()
WELD_NAME = "overall_right_hand_racket_soft_weld"

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="incoming scene XML not built; run environment.overall_environment.src.incoming_scene",
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


@pytest.fixture(scope="module")
def ready_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    assert key_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def test_build_scene_has_hard_weld(model: mujoco.MjModel) -> None:
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
    assert weld_id >= 0
    assert float(model.eq_solref[weld_id][0]) <= 0.005 + 1e-9

    root = ET.parse(SCENE_XML).getroot()
    weld = root.find(f".//equality/weld[@name='{WELD_NAME}']")
    assert weld is not None
    assert {weld.attrib["body1"], weld.attrib["body2"]} == {"thirdmc_r", "overall_racket"}


def test_root_position_at_own_half_center(model: mujoco.MjModel, ready_data: mujoco.MjData) -> None:
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    adr = int(model.jnt_qposadr[root_joint])
    assert ready_data.qpos[adr] == pytest.approx(-3.35, abs=1e-6)
    assert ready_data.qpos[adr + 1] == pytest.approx(0.0, abs=1e-6)
    quat = np.asarray(ready_data.qpos[adr + 3 : adr + 7], dtype=float)
    # facing +x: the default ready quat rotates +90 deg about z
    assert quat == pytest.approx([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], abs=1e-6)


def test_shuttle_hold_pose_airborne_on_opposite_half(model: mujoco.MjModel, ready_data: mujoco.MjData) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    adr = int(model.jnt_qposadr[joint])
    assert float(ready_data.qpos[adr]) > 0.0
    assert float(ready_data.qpos[adr + 2]) > 1.0


def test_hand_racket_contact_excluded(model: mujoco.MjModel) -> None:
    root = ET.parse(SCENE_XML).getroot()
    excludes = [
        {exclude.attrib.get("body1"), exclude.attrib.get("body2")}
        for exclude in root.findall(".//contact/exclude")
    ]
    assert {"Full Body", "overall_racket"} in excludes


def test_weld_holds_under_gravity(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    before = float(np.linalg.norm(data.site_xpos[palm] - data.site_xpos[grip]))
    for _ in range(200):
        mujoco.mj_step(model, data)
    after = float(np.linalg.norm(data.site_xpos[palm] - data.site_xpos[grip]))
    assert abs(after - before) < 0.02
