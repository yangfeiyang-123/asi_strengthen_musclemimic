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

from environment.overall_environment.src.incoming_scene import (  # noqa: E402
    build_incoming_hit_scene,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402

WELD_NAME = "overall_right_hand_racket_soft_weld"


@pytest.fixture(scope="module")
def scene_xml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("incoming_exact_child") / default_incoming_scene_path().name
    return build_incoming_hit_scene(path)


@pytest.fixture(scope="module")
def model(scene_xml: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_xml))


@pytest.fixture(scope="module")
def ready_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    assert key_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def test_build_scene_has_exact_child_without_weld_or_finger_actions(
    scene_xml: Path,
    model: mujoco.MjModel,
) -> None:
    assert model.nu == 354
    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    assert racket >= 0
    parent = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        int(model.body_parentid[racket]),
    )
    assert parent == "thirdmc_r"
    assert mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "overall_racket_free",
    ) < 0
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
    assert weld_id < 0

    root = ET.parse(scene_xml).getroot()
    weld = root.find(f".//equality/weld[@name='{WELD_NAME}']")
    assert weld is None


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


def test_hand_racket_contact_disabled_by_collision_masks(
    scene_xml: Path,
    model: mujoco.MjModel,
) -> None:
    root = ET.parse(scene_xml).getroot()
    excludes = [
        {exclude.attrib.get("body1"), exclude.attrib.get("body2")}
        for exclude in root.findall(".//contact/exclude")
    ]
    # A descendant racket must not need a redundant ancestor-child exclude.
    assert {"Full Body", "overall_racket"} not in excludes

    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    human = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Full Body")

    def is_descendant(body_id: int, ancestor_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == ancestor_id:
                return True
            current = int(model.body_parentid[current])
        return False

    racket_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if is_descendant(int(model.geom_bodyid[geom_id]), racket)
    ]
    human_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if is_descendant(int(model.geom_bodyid[geom_id]), human)
        and not is_descendant(int(model.geom_bodyid[geom_id]), racket)
    ]
    compatible = [
        (racket_geom, human_geom)
        for racket_geom in racket_geoms
        for human_geom in human_geoms
        if (
            int(model.geom_contype[racket_geom])
            & int(model.geom_conaffinity[human_geom])
        )
        or (
            int(model.geom_contype[human_geom])
            & int(model.geom_conaffinity[racket_geom])
        )
    ]
    assert compatible == []


def test_exact_child_holds_under_gravity(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thirdmc_r")
    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")

    def palm_to_racket_position() -> np.ndarray:
        palm_rot = np.asarray(data.xmat[palm], dtype=float).reshape(3, 3)
        return palm_rot.T @ (np.asarray(data.xpos[racket]) - np.asarray(data.xpos[palm]))

    before = palm_to_racket_position()
    for _ in range(200):
        mujoco.mj_step(model, data)
    after = palm_to_racket_position()
    assert after == pytest.approx(before, abs=1e-9)
