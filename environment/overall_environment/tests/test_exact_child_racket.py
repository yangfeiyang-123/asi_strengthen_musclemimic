from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.badminton_physics import BadmintonPhysics  # noqa: E402
from environment.overall_environment.src.badminton_physics_mjx import body_dof_mask  # noqa: E402
from environment.overall_environment.src.incoming_scene import build_incoming_hit_scene  # noqa: E402
from environment.overall_environment.src.incoming_shuttle_hit_env import (  # noqa: E402
    IncomingShuttleHitEnv,
)
from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (  # noqa: E402
    IncomingHitMjxEnv,
)
from environment.overall_environment.src.racket_attachment import (  # noqa: E402
    DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH,
    canonical_contract_fingerprint,
    load_racket_attachment_contract,
)
from environment.overall_environment.src.shuttle_feeder import sample_feed  # noqa: E402
from environment.overall_environment.src.stage3_lab import stage3_attachment_report  # noqa: E402
from environment.racket.src.racket_stringbed import apply_stringbed_force  # noqa: E402
from musclemimic.environments.humanoids.myofullbody import (  # noqa: E402
    FINGER_JOINT_NAMES,
    FINGER_MUSCLE_NAMES,
)
from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket  # noqa: E402


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, index) or ""


@pytest.fixture(scope="module")
def exact_scene(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, mujoco.MjModel]:
    output = tmp_path_factory.mktemp("exact_child_racket") / "incoming.xml"
    build_incoming_hit_scene(output)
    return output, mujoco.MjModel.from_xml_path(str(output))


@pytest.fixture(scope="module")
def stage2_env() -> MyoFullBodyRacket:
    return MyoFullBodyRacket(disable_fingers=True)


def test_contract_canonical_hash_and_tamper_detection(tmp_path: Path) -> None:
    document = json.loads(DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert canonical_contract_fingerprint(document) == document["fingerprint"]
    contract = load_racket_attachment_contract()
    assert contract.fingerprint == document["fingerprint"]
    assert contract.attachment_mode == "exact_child"
    assert contract.native_hand_racket_contact is False
    assert contract.stringbed_contact_model == "custom_force_event_rebound_v1"
    assert contract.native_stringbed_proxy_shuttle_contact is False
    assert contract.native_racket_frame_shuttle_contact is True

    document["relative_pose"]["position_m"][0] += 0.001
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_racket_attachment_contract(tampered)


def test_contract_rejects_unknown_fields_even_with_recomputed_hash(tmp_path: Path) -> None:
    document = json.loads(DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH.read_text(encoding="utf-8"))
    document["unversioned_override"] = True
    document["fingerprint"] = canonical_contract_fingerprint(document)
    path = tmp_path / "unknown_field.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_racket_attachment_contract(path)


def test_stage2_exposes_honest_contract_fingerprint(stage2_env: MyoFullBodyRacket) -> None:
    contract = load_racket_attachment_contract()
    assert stage2_env.racket_attachment_uses_canonical_contract is True
    assert stage2_env.racket_attachment_contract_fingerprint == contract.fingerprint
    assert stage2_env.racket_attachment_effective_fingerprint == contract.fingerprint

    overridden = MyoFullBodyRacket(disable_fingers=True, racket_mass_scale=0.5)
    assert overridden.racket_attachment_uses_canonical_contract is False
    assert overridden.racket_attachment_contract_fingerprint is None
    assert overridden.racket_attachment_effective_fingerprint != contract.fingerprint


def test_exact_child_builder_removes_finger_and_racket_dofs(
    exact_scene: tuple[Path, mujoco.MjModel],
) -> None:
    path, model = exact_scene
    assert model.nu == 354

    joint_names = {_name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)}
    actuator_names = {_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)}
    assert not joint_names.intersection(FINGER_JOINT_NAMES)
    assert not actuator_names.intersection(FINGER_MUSCLE_NAMES)
    assert "overall_racket_free" not in joint_names

    root = ET.parse(path).getroot()
    assert root.find(".//joint[@name='overall_racket_free']") is None
    assert root.find(".//equality/weld[@name='overall_right_hand_racket_soft_weld']") is None
    for weld in root.findall(".//equality/weld"):
        assert "overall_racket" not in {weld.attrib.get("body1"), weld.attrib.get("body2")}

    custom_text = {text.attrib["name"]: text.attrib["data"] for text in root.findall("./custom/text")}
    contract = load_racket_attachment_contract()
    assert custom_text["overall_racket_attachment_contract_fingerprint"] == contract.fingerprint
    assert custom_text["overall_racket_attachment_contract_schema"] == contract.schema
    assert custom_text["overall_racket_attachment_contract_path"] == str(
        DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH.relative_to(REPO_ROOT)
    )
    assert custom_text["overall_racket_attachment_mode"] == "exact_child"
    assert custom_text["overall_finger_mode"] == "removed"
    assert custom_text["overall_stringbed_contact_model"] == contract.stringbed_contact_model
    assert custom_text["overall_native_stringbed_proxy_shuttle_contact"] == "false"
    assert custom_text["overall_native_racket_frame_shuttle_contact"] == "true"


def test_exact_child_matches_stage2_attachment_and_racket_physics(
    exact_scene: tuple[Path, mujoco.MjModel],
    stage2_env: MyoFullBodyRacket,
) -> None:
    _, stage3 = exact_scene
    stage2 = stage2_env._model
    contract = load_racket_attachment_contract()

    stage3_racket = mujoco.mj_name2id(stage3, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    stage2_racket = mujoco.mj_name2id(stage2, mujoco.mjtObj.mjOBJ_BODY, "racket_racket")
    assert min(stage2_racket, stage3_racket) >= 0
    assert (
        _name(
            stage3,
            mujoco.mjtObj.mjOBJ_BODY,
            int(stage3.body_parentid[stage3_racket]),
        )
        == contract.parent_body
    )

    assert stage3.body_pos[stage3_racket] == pytest.approx(
        contract.relative_position_m,
        abs=1e-9,
    )
    assert stage3.body_quat[stage3_racket] == pytest.approx(
        contract.relative_quaternion_wxyz,
        abs=1e-6,
    )
    assert stage3.body_pos[stage3_racket] == pytest.approx(stage2.body_pos[stage2_racket])
    assert stage3.body_quat[stage3_racket] == pytest.approx(stage2.body_quat[stage2_racket])
    assert float(stage3.body_mass[stage3_racket]) == pytest.approx(contract.racket_mass_kg)
    assert stage3.body_mass[stage3_racket] == pytest.approx(stage2.body_mass[stage2_racket])
    assert stage3.body_ipos[stage3_racket] == pytest.approx(contract.racket_center_of_mass_m)
    assert stage3.body_inertia[stage3_racket] == pytest.approx(contract.racket_diagonal_inertia_kg_m2)
    assert stage3.body_inertia[stage3_racket] == pytest.approx(stage2.body_inertia[stage2_racket])

    stage3_site = mujoco.mj_name2id(
        stage3,
        mujoco.mjtObj.mjOBJ_SITE,
        "overall_stringbed_center_site",
    )
    stage2_site = mujoco.mj_name2id(
        stage2,
        mujoco.mjtObj.mjOBJ_SITE,
        "racket_stringbed_center_site",
    )
    assert min(stage2_site, stage3_site) >= 0
    assert int(stage3.site_bodyid[stage3_site]) == stage3_racket
    assert stage3.site_pos[stage3_site] == pytest.approx(contract.stringbed_position_m)
    assert stage3.site_quat[stage3_site] == pytest.approx(contract.stringbed_quaternion_wxyz)
    assert stage3.site_pos[stage3_site] == pytest.approx(stage2.site_pos[stage2_site])
    assert stage3.site_quat[stage3_site] == pytest.approx(stage2.site_quat[stage2_site])


def test_exact_child_collision_masks_isolate_the_human(
    exact_scene: tuple[Path, mujoco.MjModel],
) -> None:
    path, model = exact_scene
    racket_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    human_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Full Body")
    assert min(racket_body, human_root) >= 0

    def is_descendant(body_id: int, ancestor_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == ancestor_id:
                return True
            current = int(model.body_parentid[current])
        return False

    racket_bodies = {body_id for body_id in range(model.nbody) if is_descendant(body_id, racket_body)}
    human_bodies = {
        body_id for body_id in range(model.nbody) if is_descendant(body_id, human_root) and body_id not in racket_bodies
    }
    racket_geoms = [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) in racket_bodies]
    human_geoms = [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) in human_bodies]
    colliding_racket_geoms = [
        geom_id for geom_id in racket_geoms if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
    ]
    assert colliding_racket_geoms
    proxy_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "overall_stringbed_ground_contact_proxy",
    )
    assert proxy_geom in colliding_racket_geoms
    frame_geoms = [geom_id for geom_id in colliding_racket_geoms if geom_id != proxy_geom]
    assert frame_geoms
    assert (
        int(model.geom_contype[proxy_geom]),
        int(model.geom_conaffinity[proxy_geom]),
    ) == (16, 16)
    assert {(int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])) for geom_id in frame_geoms} == {
        (4, 4)
    }
    for racket_geom in colliding_racket_geoms:
        for human_geom in human_geoms:
            compatible = (int(model.geom_contype[racket_geom]) & int(model.geom_conaffinity[human_geom])) or (
                int(model.geom_contype[human_geom]) & int(model.geom_conaffinity[racket_geom])
            )
            assert compatible == 0

    shuttle_root = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "overall_shuttle",
    )
    shuttle_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if is_descendant(int(model.geom_bodyid[geom_id]), shuttle_root)
        and (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
    ]
    shuttle_geoms_by_name = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id): geom_id for geom_id in shuttle_geoms
    }
    assert set(shuttle_geoms_by_name) == {
        "overall_cork_collision",
        "overall_skirt_ground_support",
    }
    assert (
        int(model.geom_contype[shuttle_geoms_by_name["overall_cork_collision"]]),
        int(model.geom_conaffinity[shuttle_geoms_by_name["overall_cork_collision"]]),
    ) == (1, 13)
    assert (
        int(model.geom_contype[shuttle_geoms_by_name["overall_skirt_ground_support"]]),
        int(model.geom_conaffinity[shuttle_geoms_by_name["overall_skirt_ground_support"]]),
    ) == (1, 9)
    mask_compatible_pairs = sum(
        1
        for racket_geom in colliding_racket_geoms
        for shuttle_geom in shuttle_geoms
        if (int(model.geom_contype[racket_geom]) & int(model.geom_conaffinity[shuttle_geom]))
        or (int(model.geom_contype[shuttle_geom]) & int(model.geom_conaffinity[racket_geom]))
    )
    assert mask_compatible_pairs == len(frame_geoms)
    for shuttle_geom in shuttle_geoms:
        assert not (int(model.geom_contype[proxy_geom]) & int(model.geom_conaffinity[shuttle_geom]))
        assert not (int(model.geom_contype[shuttle_geom]) & int(model.geom_conaffinity[proxy_geom]))
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "overall_floor_collision")
    assert floor >= 0
    assert (int(model.geom_contype[proxy_geom]) & int(model.geom_conaffinity[floor])) or (
        int(model.geom_contype[floor]) & int(model.geom_conaffinity[proxy_geom])
    )

    # A descendant racket must not rely on a redundant ancestor/child exclude.
    root = ET.parse(path).getroot()
    excludes = [
        {exclude.attrib.get("body1"), exclude.attrib.get("body2")} for exclude in root.findall("./contact/exclude")
    ]
    assert {"Full Body", "overall_racket"} not in excludes


def test_custom_stringbed_contact_still_transmits_force_to_exact_child(
    exact_scene: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = exact_scene
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    shuttle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_shuttle")
    cork_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "overall_cork_contact_site",
    )
    shuttle_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "overall_shuttle_free",
    )
    assert min(racket, shuttle, cork_site, shuttle_joint) >= 0

    racket_rot = np.asarray(data.xmat[racket], dtype=float).reshape(3, 3)
    target = np.asarray(data.xpos[racket]) + racket_rot @ np.array([0.0, 0.532, 0.01])
    shuttle_adr = int(model.jnt_qposadr[shuttle_joint])
    data.qpos[shuttle_adr : shuttle_adr + 3] = 0.0
    data.qpos[shuttle_adr + 3 : shuttle_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    cork_offset = np.asarray(data.site_xpos[cork_site]) - np.asarray(data.xpos[shuttle])
    data.qpos[shuttle_adr : shuttle_adr + 3] = target - cork_offset
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:] = 0.0

    contact = apply_stringbed_force(
        model,
        data,
        racket_body_name="overall_racket",
        shuttle_body_name="overall_shuttle",
        shuttle_contact_site_name="overall_cork_contact_site",
    )
    assert contact["active"] is True
    assert float(contact["normal_force_n"]) > 0.0
    assert np.linalg.norm(data.qfrc_applied) > 0.0


def test_attachment_report_rejects_tampered_embedded_contract_metadata(
    exact_scene: tuple[Path, mujoco.MjModel],
    tmp_path: Path,
) -> None:
    path, model = exact_scene
    report = stage3_attachment_report(model, path)
    assert report["contract_passed"] is True
    assert report["contract_checks"]["single_custom_stringbed_model"] is True
    assert report["contract_checks"]["no_native_stringbed_proxy_shuttle_contact"] is True
    assert report["contract_checks"]["native_racket_frame_shuttle_contact_preserved"] is True
    assert report["contract_checks"]["no_native_racket_frame_skirt_support_contact"] is True
    assert report["native_racket_frame_cork_contact_enabled"] is True
    assert report["native_racket_frame_skirt_support_contact_enabled"] is False

    tree = ET.parse(path)
    node = tree.getroot().find("./custom/text[@name='overall_racket_attachment_contract_fingerprint']")
    assert node is not None
    node.set("data", "sha256:" + "0" * 64)
    tampered = tmp_path / "tampered_embedded_contract.xml"
    tree.write(tampered, encoding="utf-8", xml_declaration=True)
    tampered_report = stage3_attachment_report(model, tampered)
    assert tampered_report["contract_passed"] is False
    assert tampered_report["contract_checks"]["embedded_contract_fingerprint"] is False


def test_nondefault_contract_path_flows_into_cpu_and_mjx_manifests(
    tmp_path: Path,
) -> None:
    custom_contract = tmp_path / "custom_racket_contract.json"
    custom_contract.write_bytes(DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH.read_bytes())
    scene = build_incoming_hit_scene(
        tmp_path / "custom_contract_scene.xml",
        racket_attachment_contract=custom_contract,
    )
    expected_path = str(custom_contract.resolve())

    cpu_env = IncomingShuttleHitEnv(scene)
    assert cpu_env.control_manifest["racket_attachment"]["contract_path"] == expected_path
    assert cpu_env.control_manifest["racket_attachment"]["contract_passed"] is True

    feed = sample_feed(np.random.default_rng(37))
    mjx_env = IncomingHitMjxEnv(xml=scene, feed_bank=[feed], impl="jax")
    assert mjx_env.control_manifest["racket_attachment"]["contract_path"] == expected_path
    assert mjx_env.control_manifest["racket_attachment"]["contract_passed"] is True


def test_high_speed_event_has_equal_opposite_racket_chain_reaction(
    exact_scene: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = exact_scene
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    shuttle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_shuttle")
    cork_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_cork_contact_site")
    shuttle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    racket_rot = np.asarray(data.xmat[racket], dtype=float).reshape(3, 3)
    normal = racket_rot[:, 2]
    target = np.asarray(data.xpos[racket]) + racket_rot @ np.array([0.0, 0.532, 0.005])
    qadr = int(model.jnt_qposadr[shuttle_joint])
    dadr = int(model.jnt_dofadr[shuttle_joint])
    data.qpos[qadr : qadr + 3] = 0.0
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    cork_offset = np.asarray(data.site_xpos[cork_site]) - np.asarray(data.xpos[shuttle])
    data.qpos[qadr : qadr + 3] = target - cork_offset
    data.qvel[dadr : dadr + 3] = -8.0 * normal
    mujoco.mj_forward(model, data)

    # The native planar proxy must not contribute a simultaneous MuJoCo contact.
    proxy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "overall_stringbed_ground_contact_proxy")
    shuttle_bodies = {shuttle}
    native_proxy_pairs = 0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        other = geom2 if geom1 == proxy else (geom1 if geom2 == proxy else -1)
        if other >= 0 and int(model.geom_bodyid[other]) in shuttle_bodies:
            native_proxy_pairs += 1
    assert native_proxy_pairs == 0

    diag = BadmintonPhysics().substep(model, data)
    assert diag["event_rebound_used"] is True
    assert diag["event_stringbed_force_suppressed"] is True
    shuttle_impulse = np.asarray(diag["event_impulse_on_shuttle_world_ns"])
    racket_impulse = np.asarray(diag["event_impulse_on_racket_world_ns"])
    np.testing.assert_allclose(shuttle_impulse + racket_impulse, 0.0, atol=1e-12)
    before = np.asarray(diag["event_shuttle_velocity_before_world_m_s"])
    after = np.asarray(diag["event_shuttle_velocity_after_world_m_s"])
    np.testing.assert_allclose(
        shuttle_impulse,
        float(model.body_mass[shuttle]) * (after - before),
        atol=1e-12,
    )

    generalized_impulse = np.asarray(diag["event_reaction_generalized_impulse_ns"])
    ancestor_mask = body_dof_mask(model, racket)
    assert np.linalg.norm(generalized_impulse[ancestor_mask]) > 0.0
    np.testing.assert_allclose(generalized_impulse[~ancestor_mask], 0.0, atol=1e-12)
