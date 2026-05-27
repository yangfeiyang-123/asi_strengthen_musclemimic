from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import mujoco
import musclemimic_models
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.paths import (
    court_xml_path,
    default_overall_scene_path,
    grip_reference_json_path,
    grip_reference_xml_path,
    racket_xml_path,
    shuttlecock_xml_path,
)

READY_KEYFRAME = "overall_ready"
OVERALL_CAMERA_NAME = "overall_view"
HUMAN_ROOT_FREEJOINT = "root"
SHUTTLE_FREEJOINT = "overall_shuttle_free"
RACKET_FREEJOINT = "overall_racket_free"
PORTABLE_MSK_ASSET_DIR = "mimic_msk_model"
INITIAL_HUMAN_ROOT_POS = np.array([-2.5, 0.0, 1.0], dtype=float)
INITIAL_SHUTTLE_POS = np.array([-3.35, -1.35, 0.034], dtype=float)
INITIAL_SHUTTLE_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=float)

HAND_GRIP_SITES: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("lunate_r", "rh_palm_grip_site", (0.0, 0.0, 0.0)),
    ("distal_thumb_r", "rh_thumb_pad_site", (0.0, 0.0, 0.0)),
    ("2distph_r", "rh_index_pad_site", (0.0, 0.0, 0.0)),
    ("3distph_r", "rh_middle_pad_site", (0.0, 0.0, 0.0)),
    ("4distph_r", "rh_ring_pad_site", (0.0, 0.0, 0.0)),
    ("5distph_r", "rh_pinky_pad_site", (0.0, 0.0, 0.0)),
)
SITE_SIZE = (0.006, 0.006, 0.006)
HAND_SITE_RGBA = (0.1, 0.7, 1.0, 1.0)


def build_overall_scene(output_xml: str | Path | None = None) -> Path:
    """Build a combined court + MyoFullBody + held racket + grounded shuttle scene."""
    out_path = Path(output_xml) if output_xml is not None else default_overall_scene_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _copy_msk_visual_assets(tmp_path / PORTABLE_MSK_ASSET_DIR)
        base_spec = mujoco.MjSpec.from_file(str(musclemimic_models.get_xml_path("myofullbody")))
        _add_hand_grip_sites(base_spec)

        for component, source, removals in (
            ("court", court_xml_path(), ()),
            ("racket", racket_xml_path(), ()),
            ("shuttle", shuttlecock_xml_path(), ("floor",)),
        ):
            prefixed_xml = tmp_path / f"overall_{component}.xml"
            _write_prefixed_component_xml(source, prefixed_xml, "overall", remove_worldbody_names=removals)
            component_spec = mujoco.MjSpec.from_file(str(prefixed_xml))
            frame = base_spec.worldbody.add_frame(name=f"overall_{component}_frame")
            base_spec.attach(component_spec, frame=frame)

        base_spec.compile()
        raw_xml = tmp_path / "overall_raw.xml"
        base_spec.to_file(str(raw_xml))
        _postprocess_attached_xml(raw_xml)
        _disable_actuation(raw_xml)
        _exclude_person_racket_contacts(raw_xml)
        _separate_racket_collision_group(raw_xml)
        _add_overall_camera(raw_xml)
        qpos = _overall_ready_qpos(raw_xml)
        _add_ready_keyframe(raw_xml, qpos)
        _apply_ready_as_initial_pose(raw_xml, qpos)
        mujoco.MjModel.from_xml_path(str(raw_xml))
        out_path.write_bytes(raw_xml.read_bytes())
        _copy_msk_visual_assets(out_path.parent / PORTABLE_MSK_ASSET_DIR)

    return out_path


def _copy_msk_visual_assets(destination: Path) -> None:
    source_root = Path(musclemimic_models.get_xml_path("myofullbody")).resolve().parents[1]
    if destination.exists():
        shutil.rmtree(destination)
    for directory_name in ("meshes",):
        source = source_root / directory_name
        target = destination / directory_name
        if not source.is_dir():
            raise FileNotFoundError(f"missing MyoFullBody visual asset directory: {source}")
        shutil.copytree(source, target, dirs_exist_ok=True)


def _add_hand_grip_sites(spec: mujoco.MjSpec) -> None:
    for body_name, site_name, pos in HAND_GRIP_SITES:
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"missing MyoFullBody right-hand body: {body_name}")
        if any(site.name == site_name for site in body.sites):
            continue
        body.add_site(name=site_name, pos=pos, size=SITE_SIZE, rgba=HAND_SITE_RGBA)


def _write_prefixed_component_xml(
    source: Path,
    destination: Path,
    prefix: str,
    *,
    remove_worldbody_names: Iterable[str] = (),
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    removals = set(remove_worldbody_names)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.attrib.get("name") in removals:
                worldbody.remove(child)

    for elem in root.iter():
        if "name" in elem.attrib:
            elem.set("name", _prefixed(prefix, elem.attrib["name"]))
        if "class" in elem.attrib:
            elem.set("class", _prefixed(prefix, elem.attrib["class"]))
        if "childclass" in elem.attrib:
            elem.set("childclass", _prefixed(prefix, elem.attrib["childclass"]))
        for ref_attr in ("material", "texture", "mesh", "hfield"):
            if ref_attr in elem.attrib:
                elem.set(ref_attr, _prefixed(prefix, elem.attrib[ref_attr]))

    destination.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def _prefixed(prefix: str, name: str) -> str:
    if name.startswith(f"{prefix}_"):
        return name
    return f"{prefix}_{name}"


def _strip_attachment_namespace(root: ET.Element) -> None:
    for elem in root.iter():
        for attr in ("name", "class", "childclass", "material", "texture", "mesh", "hfield"):
            value = elem.attrib.get(attr)
            if value is not None and value.startswith("/"):
                elem.set(attr, value[1:])


def _deduplicate_attached_main_defaults(root: ET.Element) -> None:
    main_index = 0
    seen: dict[str, int] = {}
    for default in root.findall(".//default"):
        class_name = default.attrib.get("class")
        if class_name is None:
            continue
        if class_name == "main":
            default.set("class", f"overall_attached_main_{main_index}")
            main_index += 1
            continue
        seen[class_name] = seen.get(class_name, 0) + 1
        if seen[class_name] > 1:
            default.set("class", f"{class_name}_{seen[class_name]}")


def _make_asset_paths_portable(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", PORTABLE_MSK_ASSET_DIR)
        compiler.set("texturedir", PORTABLE_MSK_ASSET_DIR)
    for asset in root.findall("asset"):
        for child in list(asset):
            if child.tag == "texture" and child.attrib.get("type") == "skybox":
                asset.remove(child)
                continue
            if child.tag == "texture" and "file" in child.attrib:
                texture_file = Path(child.attrib["file"])
                if texture_file.is_absolute():
                    asset.remove(child)


def _add_shuttle_ground_support(root: ET.Element) -> None:
    shuttle_body = root.find(".//body[@name='overall_shuttle']")
    if shuttle_body is None:
        raise ValueError("generated XML is missing the overall_shuttle body")
    if shuttle_body.find("geom[@name='overall_skirt_ground_support']") is not None:
        return
    ET.SubElement(
        shuttle_body,
        "geom",
        {
            "conaffinity": "1",
            "contype": "1",
            "friction": "0.8 0.01 0.001",
            "group": "3",
            "name": "overall_skirt_ground_support",
            "pos": "0 0 -0.035",
            "rgba": "0 0 0 0",
            "size": "0.0325 0.0325 0.030",
            "solimp": "0.90 0.95 0.001 0.5 2",
            "solref": "0.002 1",
            "type": "ellipsoid",
        },
    )


def _tune_scene_materials(root: ET.Element) -> None:
    material_overrides = {
        "MatPlane": {
            "rgba": "0.13 0.13 0.13 1",
            "reflectance": "0",
            "shininess": "0",
            "specular": "0",
        },
        "overall_mat_floor": {
            "rgba": "0.015 0.34 0.14 1",
            "reflectance": "0",
            "shininess": "0",
            "specular": "0",
        },
        "overall_mat_line": {
            "rgba": "1 1 0.94 1",
            "reflectance": "0",
            "shininess": "0",
            "specular": "0",
        },
        "overall_mat_net_tape": {
            "rgba": "0.92 0.92 0.86 1",
            "reflectance": "0",
            "shininess": "0",
            "specular": "0",
        },
        "overall_mat_net_cord": {
            "reflectance": "0",
            "shininess": "0",
            "specular": "0",
        },
        "overall_mat_post": {
            "reflectance": "0",
            "shininess": "0",
            "specular": "0",
        },
    }
    for material in root.findall("./asset/material"):
        name = material.attrib.get("name")
        overrides = material_overrides.get(name)
        if overrides is None:
            continue
        if name == "MatPlane":
            for texture_attr in ("texture", "texrepeat", "texuniform"):
                material.attrib.pop(texture_attr, None)
        for attr, value in overrides.items():
            material.set(attr, value)

    floor_geom = root.find(".//geom[@name='floor']")
    if floor_geom is not None:
        floor_geom.set("rgba", "0.13 0.13 0.13 1")


def _sort_attributes(root: ET.Element) -> None:
    for elem in root.iter():
        if len(elem.attrib) > 1:
            attributes = sorted(elem.attrib.items())
            elem.attrib.clear()
            elem.attrib.update(attributes)


def _postprocess_attached_xml(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    _strip_attachment_namespace(root)
    _deduplicate_attached_main_defaults(root)
    _make_asset_paths_portable(root)
    _add_shuttle_ground_support(root)
    _tune_scene_materials(root)
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _disable_actuation(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("actuation", "disable")
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _exclude_person_racket_contacts(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for exclude in contact.findall("exclude"):
        if {
            exclude.attrib.get("body1"),
            exclude.attrib.get("body2"),
        } == {"Full Body", "overall_racket"}:
            return
    ET.SubElement(
        contact,
        "exclude",
        {
            "body1": "Full Body",
            "body2": "overall_racket",
        },
    )
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _separate_racket_collision_group(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")
        geom_class = geom.attrib.get("class", "")
        if name in {"floor", "overall_floor_collision"}:
            geom.set("conaffinity", "5")
        if geom_class in {"overall_frame_contact", "overall_stringbed_ground_contact"}:
            geom.set("contype", "4")
            geom.set("conaffinity", "4")
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _add_overall_camera(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("generated XML has no worldbody")
    for camera in list(worldbody.findall("camera")):
        if camera.attrib.get("name") == OVERALL_CAMERA_NAME:
            worldbody.remove(camera)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": OVERALL_CAMERA_NAME,
            "mode": "fixed",
            "pos": "8 -8 5.2",
            "xyaxes": "0.707107 0.707107 0 -0.353553 0.353553 0.866025",
        },
    )
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _add_ready_keyframe(path: Path, qpos: np.ndarray) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.attrib.get("name") == READY_KEYFRAME:
            keyframe.remove(key)
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": READY_KEYFRAME,
            "qpos": " ".join(f"{value:.17g}" for value in qpos),
        },
    )
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _apply_ready_as_initial_pose(path: Path, qpos: np.ndarray) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    tree = ET.parse(path)
    root = tree.getroot()
    reference = json.loads(grip_reference_json_path().read_text(encoding="utf-8"))
    right_hand_joint_names = set(reference["right_hand_joint_names"])

    joint_by_name = {
        joint.attrib["name"]: joint
        for joint in root.findall(".//joint")
        if "name" in joint.attrib
    }
    for joint_name in right_hand_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint = joint_by_name.get(joint_name)
        if joint_id < 0 or joint is None:
            continue
        joint.set("ref", f"{qpos[int(model.jnt_qposadr[joint_id])]:.17g}")

    body_by_freejoint = _body_by_freejoint_name(root)
    for joint_name in (HUMAN_ROOT_FREEJOINT, RACKET_FREEJOINT, SHUTTLE_FREEJOINT):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body = body_by_freejoint.get(joint_name)
        if joint_id < 0 or body is None:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        body.set("pos", " ".join(f"{value:.17g}" for value in qpos[adr : adr + 3]))
        body.set("quat", " ".join(f"{value:.17g}" for value in qpos[adr + 3 : adr + 7]))

    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _body_by_freejoint_name(root: ET.Element) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for body in root.findall(".//body"):
        for joint in body.findall("joint"):
            joint_name = joint.attrib.get("name")
            if joint_name is not None and joint.attrib.get("type") == "free":
                result[joint_name] = body
    return result


def _overall_ready_qpos(xml_path: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos = np.array(model.qpos0, dtype=float)
    reference = json.loads(grip_reference_json_path().read_text(encoding="utf-8"))
    reference_qpos = np.asarray(reference["qpos"], dtype=float)
    reference_model = mujoco.MjModel.from_xml_path(str(grip_reference_xml_path()))

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HUMAN_ROOT_FREEJOINT)
    if root_id < 0:
        raise ValueError(f"missing joint {HUMAN_ROOT_FREEJOINT!r}")
    root_adr = int(model.jnt_qposadr[root_id])
    qpos[root_adr : root_adr + 3] = INITIAL_HUMAN_ROOT_POS

    right_hand_joint_names = set(reference["right_hand_joint_names"])
    for joint_id in range(reference_model.njnt):
        joint_name = mujoco.mj_id2name(reference_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name not in right_hand_joint_names:
            continue
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if target_id < 0:
            continue
        width = _joint_qpos_width(reference_model, joint_id)
        source_adr = int(reference_model.jnt_qposadr[joint_id])
        target_adr = int(model.jnt_qposadr[target_id])
        qpos[target_adr : target_adr + width] = reference_qpos[source_adr : source_adr + width]

    _place_racket_at_right_hand(model, qpos, reference)

    shuttle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SHUTTLE_FREEJOINT)
    if shuttle_id < 0:
        raise ValueError(f"missing joint {SHUTTLE_FREEJOINT!r}")
    shuttle_adr = int(model.jnt_qposadr[shuttle_id])
    qpos[shuttle_adr : shuttle_adr + 7] = np.concatenate(
        [INITIAL_SHUTTLE_POS, INITIAL_SHUTTLE_QUAT]
    )
    return qpos


def _place_racket_at_right_hand(model: mujoco.MjModel, qpos: np.ndarray, reference: dict[str, object]) -> None:
    racket_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, RACKET_FREEJOINT)
    palm_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    if racket_id < 0 or palm_site_id < 0 or grip_site_id < 0:
        raise ValueError("missing racket freejoint or right-hand grip sites")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    racket_reference = np.asarray(reference["racket_freejoint_qpos"], dtype=float)
    racket_quat = racket_reference[3:7]
    racket_quat = racket_quat / np.linalg.norm(racket_quat)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, racket_quat)
    rotation = rotation.reshape(3, 3)

    grip_local = np.array(model.site_pos[grip_site_id], dtype=float)
    racket_pos = np.array(data.site_xpos[palm_site_id], dtype=float) - rotation @ grip_local
    racket_adr = int(model.jnt_qposadr[racket_id])
    qpos[racket_adr : racket_adr + 7] = np.concatenate([racket_pos, racket_quat])


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the overall badminton MuJoCo scene.")
    parser.add_argument("--out", type=Path, default=default_overall_scene_path(), help="Output XML path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_overall_scene(args.out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
