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
    default_overall_training_scene_path,
    grip_reference_json_path,
    grip_reference_xml_path,
    grip_seed_json_path,
    racket_xml_path,
    shuttlecock_xml_path,
)
from environment.overall_environment.src.racket_attachment import (
    RacketAttachmentContract,
    load_racket_attachment_contract,
    validate_racket_spec_against_contract,
)
from musclemimic.environments.humanoids.myofullbody import remove_finger_dofs
from src.grip.grip_seed import GripSeed, apply_seed_right_hand_joints, load_grip_seed

READY_KEYFRAME = "overall_ready"
OVERALL_CAMERA_NAME = "overall_view"
HUMAN_ROOT_FREEJOINT = "root"
SHUTTLE_FREEJOINT = "overall_shuttle_free"
RACKET_FREEJOINT = "overall_racket_free"
PORTABLE_MSK_ASSET_DIR = "mimic_msk_model"
INITIAL_HUMAN_ROOT_POS = np.array([-2.5, 0.0, 1.0], dtype=float)
INITIAL_HUMAN_ROOT_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=float)
INITIAL_SHUTTLE_POS = np.array([-3.35, -1.35, 0.034], dtype=float)
INITIAL_SHUTTLE_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=float)
SCENE_MODES = ("inspection", "training")
ATTACHMENT_MODES = ("legacy_free", "exact_child")
FINGER_MODES = ("full", "removed")
SOFT_WELD_NAME = "overall_right_hand_racket_soft_weld"
SOFT_WELD_BODY1 = "thirdmc_r"
SOFT_WELD_BODY2 = "overall_racket"

HAND_GRIP_SITES: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("thirdmc_r", "rh_palm_grip_site", (0.0, 0.0, 0.0)),
    ("distal_thumb_r", "rh_thumb_pad_site", (0.015, -0.018, -0.007)),
    ("2distph_r", "rh_index_pad_site", (0.003, -0.018, 0.0055)),
    ("3distph_r", "rh_middle_pad_site", (0.002, -0.019, 0.003)),
    ("4distph_r", "rh_ring_pad_site", (-0.004, -0.019, 0.003)),
    ("5distph_r", "rh_pinky_pad_site", (-0.005, -0.018, 0.0)),
)
SITE_SIZE = (0.006, 0.006, 0.006)
HAND_SITE_RGBA = (0.1, 0.7, 1.0, 1.0)


def build_overall_scene(
    output_xml: str | Path | None = None,
    *,
    grip_seed: str | Path | None = None,
    mode: str = "inspection",
    attachment_mode: str = "legacy_free",
    finger_mode: str = "full",
    racket_attachment_contract: str | Path | RacketAttachmentContract | None = None,
    enable_actuation: bool | None = None,
    enable_person_racket_contact: bool | None = None,
    enable_soft_weld: bool = False,
    soft_weld_solref: str = "0.02 1",
    soft_weld_solimp: str = "0.8 0.95 0.001",
    human_root_pos: np.ndarray | None = None,
    human_root_quat: np.ndarray | None = None,
    shuttle_qpos: np.ndarray | None = None,
) -> Path:
    """Build a combined court + MyoFullBody + held racket + grounded shuttle scene.

    ``inspection`` mode is passive and contact-safe for visualization. ``training``
    mode exposes the MyoFullBody actuators.  ``legacy_free`` preserves the
    inspection/physical-grip scene with a free racket; production Stage 3 uses
    ``exact_child`` plus ``finger_mode="removed"`` and the versioned rigid
    attachment contract.
    ``human_root_pos``/``human_root_quat``/``shuttle_qpos`` override the default
    ready-pose placement of the person root free joint and the shuttle free joint
    (7 values: pos + quat); ``None`` keeps the historical defaults.
    """
    if mode not in SCENE_MODES:
        raise ValueError(f"mode must be one of {SCENE_MODES}, got {mode!r}")
    if attachment_mode not in ATTACHMENT_MODES:
        raise ValueError(
            f"attachment_mode must be one of {ATTACHMENT_MODES}, got {attachment_mode!r}"
        )
    if finger_mode not in FINGER_MODES:
        raise ValueError(f"finger_mode must be one of {FINGER_MODES}, got {finger_mode!r}")
    contract: RacketAttachmentContract | None = None
    if attachment_mode == "exact_child":
        if finger_mode != "removed":
            raise ValueError("production exact_child requires finger_mode='removed'")
        if isinstance(racket_attachment_contract, RacketAttachmentContract):
            contract = racket_attachment_contract
            contract.verify_asset()
        else:
            contract = load_racket_attachment_contract(racket_attachment_contract)
        if enable_soft_weld:
            raise ValueError("exact_child is jointless and cannot enable a redundant soft weld")
    elif racket_attachment_contract is not None:
        raise ValueError("racket_attachment_contract is only valid for attachment_mode='exact_child'")
    if enable_actuation is None:
        enable_actuation = mode == "training"
    if enable_person_racket_contact is None:
        enable_person_racket_contact = mode == "training" and attachment_mode == "legacy_free"
    if attachment_mode == "exact_child" and enable_person_racket_contact:
        raise ValueError("exact_child contract requires native hand-racket contact to remain disabled")

    default_path = default_overall_training_scene_path() if mode == "training" else default_overall_scene_path()
    out_path = Path(output_xml) if output_xml is not None else default_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _copy_msk_visual_assets(tmp_path / PORTABLE_MSK_ASSET_DIR)
        base_spec = mujoco.MjSpec.from_file(str(musclemimic_models.get_xml_path("myofullbody")))
        if finger_mode == "removed":
            remove_finger_dofs(base_spec)
        _add_hand_grip_sites(base_spec)

        for component, source, removals in (
            ("court", court_xml_path(), ()),
            ("racket", contract.asset_path if contract is not None else racket_xml_path(), ()),
            ("shuttle", shuttlecock_xml_path(), ("floor",)),
        ):
            prefixed_xml = tmp_path / f"overall_{component}.xml"
            _write_prefixed_component_xml(source, prefixed_xml, "overall", remove_worldbody_names=removals)
            component_spec = mujoco.MjSpec.from_file(str(prefixed_xml))
            if component == "racket" and contract is not None:
                source_spec = mujoco.MjSpec.from_file(str(contract.asset_path))
                validate_racket_spec_against_contract(source_spec, contract)
                _attach_exact_child_racket(base_spec, component_spec, contract)
            else:
                frame = base_spec.worldbody.add_frame(name=f"overall_{component}_frame")
                base_spec.attach(component_spec, frame=frame)

        base_spec.compile()
        raw_xml = tmp_path / "overall_raw.xml"
        base_spec.to_file(str(raw_xml))
        _postprocess_attached_xml(raw_xml)
        if contract is not None:
            _record_exact_child_contract(raw_xml, contract, finger_mode=finger_mode)
        if enable_actuation:
            _enable_actuation(raw_xml)
        else:
            _disable_actuation(raw_xml)
        if enable_person_racket_contact:
            _remove_person_racket_contact_excludes(raw_xml)
        elif attachment_mode == "legacy_free":
            _exclude_person_racket_contacts(raw_xml)
        _separate_racket_collision_group(raw_xml)
        _make_mjx_compatible(raw_xml)
        if contract is not None:
            _preserve_racket_shuttle_contacts(
                raw_xml,
                racket_collision_bit=contract.racket_collision_bit,
                stringbed_proxy_geom_name=contract.stringbed_proxy_geom_name,
                stringbed_ground_collision_bit=contract.stringbed_ground_collision_bit,
            )
        _add_overall_camera(raw_xml)
        qpos = _overall_ready_qpos(
            raw_xml,
            grip_seed,
            human_root_pos=human_root_pos,
            human_root_quat=human_root_quat,
            shuttle_qpos=shuttle_qpos,
            attachment_mode=attachment_mode,
        )
        _add_ready_keyframe(raw_xml, qpos)
        _apply_ready_as_initial_pose(raw_xml, qpos, grip_seed, finger_mode=finger_mode)
        if enable_soft_weld:
            _add_hand_racket_soft_weld(
                raw_xml,
                solref=soft_weld_solref,
                solimp=soft_weld_solimp,
            )
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


def _attach_exact_child_racket(
    base_spec: mujoco.MjSpec,
    racket_spec: mujoco.MjSpec,
    contract: RacketAttachmentContract,
) -> None:
    """Attach the already ``overall_``-prefixed racket as a jointless child."""

    parent = base_spec.body(contract.parent_body)
    if parent is None:
        raise ValueError(f"exact-child parent body {contract.parent_body!r} is missing")

    root_name = _prefixed("overall", contract.racket_source_body)
    root_body = racket_spec.body(root_name)
    if root_body is None:
        raise ValueError(f"prefixed racket asset is missing root body {root_name!r}")
    root_bodies = list(racket_spec.worldbody.bodies)
    if len(root_bodies) != 1 or root_bodies[0].name != root_name:
        raise ValueError(
            "exact-child racket asset must contain one world-root body; "
            f"got {[body.name for body in root_bodies]}"
        )

    free_joints = [
        joint for joint in racket_spec.joints if joint.type == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joints) != 1:
        raise ValueError(
            f"exact-child racket asset must contain one source freejoint, got {len(free_joints)}"
        )
    racket_spec.delete(free_joints[0])

    # The source free body uses pos="0 0 1.2" as its spawn pose.  Once the
    # freejoint is removed that pose would become an unwanted fixed offset.
    root_body.pos = [0.0, 0.0, 0.0]
    root_body.quat = [1.0, 0.0, 0.0, 0.0]

    collision_bit = contract.racket_collision_bit
    for geom in racket_spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = collision_bit
            geom.conaffinity = collision_bit

    frame = parent.add_frame(
        name="overall_racket_frame",
        pos=list(contract.relative_position_m),
        quat=list(contract.relative_quaternion_wxyz),
    )
    base_spec.attach(racket_spec, frame=frame)


def _record_exact_child_contract(
    path: Path,
    contract: RacketAttachmentContract,
    *,
    finger_mode: str,
) -> None:
    """Embed the immutable build contract in the generated MuJoCo XML."""

    tree = ET.parse(path)
    root = tree.getroot()
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    values = {
        "overall_racket_attachment_contract_schema": contract.schema,
        "overall_racket_attachment_contract_id": contract.contract_id,
        "overall_racket_attachment_contract_fingerprint": contract.fingerprint,
        "overall_racket_attachment_contract_path": _contract_path_for_metadata(
            contract.source_path
        ),
        "overall_racket_attachment_mode": contract.attachment_mode,
        "overall_finger_mode": finger_mode,
        "overall_stringbed_contact_model": contract.stringbed_contact_model,
        "overall_native_stringbed_proxy_shuttle_contact": str(
            contract.native_stringbed_proxy_shuttle_contact
        ).lower(),
        "overall_native_racket_frame_shuttle_contact": str(
            contract.native_racket_frame_shuttle_contact
        ).lower(),
        "overall_stringbed_proxy_geom_name": _prefixed(
            "overall", contract.stringbed_proxy_geom_name
        ),
    }
    for name, data in values.items():
        for existing in list(custom.findall("text")):
            if existing.attrib.get("name") == name:
                custom.remove(existing)
        ET.SubElement(custom, "text", {"name": name, "data": data})
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _contract_path_for_metadata(path: Path) -> str:
    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    try:
        return str(resolved.relative_to(repository_root))
    except ValueError:
        return str(resolved)


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


def _enable_actuation(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    option = root.find("option")
    if option is not None:
        flag = option.find("flag")
        if flag is not None:
            flag.attrib.pop("actuation", None)
            if not flag.attrib:
                option.remove(flag)
        if not list(option) and not option.attrib and not (option.text or "").strip():
            root.remove(option)
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


def _remove_person_racket_contact_excludes(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        return
    for exclude in list(contact.findall("exclude")):
        if {
            exclude.attrib.get("body1"),
            exclude.attrib.get("body2"),
        } == {"Full Body", "overall_racket"}:
            contact.remove(exclude)
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


def _preserve_racket_shuttle_contacts(
    path: Path,
    *,
    racket_collision_bit: int,
    stringbed_proxy_geom_name: str,
    stringbed_ground_collision_bit: int,
) -> None:
    """Enable native frame contact while excluding native stringbed contact.

    The racket must keep ``(contype, conaffinity)=(bit, bit)``: adding human bit
    1 to the racket would reopen hand-racket contacts.  OR-ing only the cork's
    conaffinity preserves its ground/net contacts while enabling native *frame*
    collision pairs (for example cork ``9 -> 13`` after the MJX ellipsoid
    compatibility pass).  ``overall_skirt_ground_support`` is an invisible
    landing/rest proxy, not a feather-impact model, and must never accept the
    racket bit.  Letting it collide with the frame can resolve a custom
    stringbed event a second time in the same substep and manufacture a false
    outgoing velocity.  The broad planar stringbed proxy is moved to a
    dedicated ground-only bit so the custom force/event model is the only
    shuttle--stringbed model.  Court floors accept that bit to retain racket
    ground contact.
    """

    model = mujoco.MjModel.from_xml_path(str(path))
    shuttle_root = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "overall_shuttle",
    )
    if shuttle_root < 0:
        raise ValueError("generated XML is missing overall_shuttle")

    def is_shuttle_descendant(body_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == shuttle_root:
                return True
            current = int(model.body_parentid[current])
        return False

    accepted_masks: dict[str, int] = {}
    for geom_id in range(model.ngeom):
        if not is_shuttle_descendant(int(model.geom_bodyid[geom_id])):
            continue
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        if contype == 0 and conaffinity == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            raise ValueError("every collidable shuttle geom must have a name")
        if name == "overall_skirt_ground_support":
            continue
        accepted_masks[name] = conaffinity | int(racket_collision_bit)
    if "overall_cork_collision" not in accepted_masks:
        raise ValueError("overall_shuttle has no collidable cork geom")

    tree = ET.parse(path)
    root = tree.getroot()
    geoms_by_name = {
        geom.attrib["name"]: geom
        for geom in root.findall(".//geom")
        if "name" in geom.attrib
    }
    for name, conaffinity in accepted_masks.items():
        geom = geoms_by_name.get(name)
        if geom is None:
            raise ValueError(f"generated XML is missing shuttle geom {name!r}")
        geom.set("conaffinity", str(conaffinity))

    proxy_name = _prefixed("overall", stringbed_proxy_geom_name)
    proxy = geoms_by_name.get(proxy_name)
    if proxy is None:
        raise ValueError(f"generated XML is missing stringbed proxy geom {proxy_name!r}")
    proxy.set("contype", str(int(stringbed_ground_collision_bit)))
    proxy.set("conaffinity", str(int(stringbed_ground_collision_bit)))

    ground_geoms = [
        geom
        for name, geom in geoms_by_name.items()
        if name in {"floor", "overall_floor_collision"}
    ]
    if not ground_geoms:
        raise ValueError("generated XML has no court floor geom for stringbed ground contact")
    for geom in ground_geoms:
        affinity = int(geom.attrib.get("conaffinity", "0"))
        geom.set("conaffinity", str(affinity | int(stringbed_ground_collision_bit)))
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


MJX_ELLIPSOID_CONTYPE_BIT = 8


def _make_mjx_compatible(path: Path) -> None:
    """Rework collision pairs that MuJoCo MJX cannot handle (ellipsoid-box).

    MJX 3.4 has no ELLIPSOID-BOX collision function. Two behavior-preserving
    changes make the composed scene loadable by ``mjx.put_model``:

    1. The hidden shuttle skirt support ellipsoid becomes a sphere with the
       same lateral radius: the side-lying rest height it was calibrated for
       is unchanged; only the never-used vertical tail-down rest shifts 2.5 mm.
    2. Human-body collision ellipsoids (head, thorax, pelvis, fingertips, one
       heel proxy per foot) move to a dedicated contype bit accepted only by
       the ground plane and the shuttle geoms. Their former box partners are
       dropped: the court floor box top is exactly coplanar with the base
       ground plane (both z=0, same tangential friction), so the plane contact
       already provides the identical support surface, and every capsule/box
       body geom keeps colliding with the court floor and net as before.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    ellipsoid_flags: list[tuple[str | None, int, int]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            continue
        ellipsoid_flags.append(
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                int(model.geom_contype[geom_id]),
                int(model.geom_conaffinity[geom_id]),
            )
        )

    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("generated XML has no worldbody")
    ellipsoid_elems = [
        geom for geom in worldbody.iter("geom") if geom.attrib.get("type") == "ellipsoid"
    ]
    if len(ellipsoid_elems) != len(ellipsoid_flags):
        raise ValueError(
            "ellipsoid count mismatch between XML and compiled model: "
            f"{len(ellipsoid_elems)} != {len(ellipsoid_flags)}"
        )

    for elem, (name, contype, conaffinity) in zip(ellipsoid_elems, ellipsoid_flags):
        if name == "overall_skirt_ground_support":
            elem.set("type", "sphere")
            elem.set("size", "0.0325")
            elem.set("conaffinity", str(1 | MJX_ELLIPSOID_CONTYPE_BIT))
            continue
        if contype == 0 and conaffinity == 0:
            continue  # visual-only, e.g. the aero proxy
        elem.set("contype", str(MJX_ELLIPSOID_CONTYPE_BIT))
        elem.set("conaffinity", "0")

    for geom_name, base_conaffinity in (("floor", 5), ("overall_cork_collision", 1)):
        geom = root.find(f".//geom[@name='{geom_name}']")
        if geom is not None:
            geom.set("conaffinity", str(base_conaffinity | MJX_ELLIPSOID_CONTYPE_BIT))

    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _add_hand_racket_soft_weld(path: Path, *, solref: str, solimp: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, READY_KEYFRAME)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    body1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SOFT_WELD_BODY1)
    body2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SOFT_WELD_BODY2)
    if body1_id < 0 or body2_id < 0:
        raise ValueError(
            f"cannot add soft weld; missing bodies {SOFT_WELD_BODY1!r}/{SOFT_WELD_BODY2!r}"
        )

    body1_rot = np.array(data.xmat[body1_id], dtype=float).reshape(3, 3)
    body2_rot = np.array(data.xmat[body2_id], dtype=float).reshape(3, 3)
    rel_pos = body1_rot.T @ (np.array(data.xpos[body2_id]) - np.array(data.xpos[body1_id]))
    rel_rot = body1_rot.T @ body2_rot
    rel_quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(9))
    rel_pose = np.concatenate([rel_pos, rel_quat])

    tree = ET.parse(path)
    root = tree.getroot()
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for weld in list(equality.findall("weld")):
        if weld.attrib.get("name") == SOFT_WELD_NAME:
            equality.remove(weld)
    ET.SubElement(
        equality,
        "weld",
        {
            "body1": SOFT_WELD_BODY1,
            "body2": SOFT_WELD_BODY2,
            "name": SOFT_WELD_NAME,
            "relpose": " ".join(f"{value:.17g}" for value in rel_pose),
            "solimp": solimp,
            "solref": solref,
        },
    )
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


def _apply_ready_as_initial_pose(
    path: Path,
    qpos: np.ndarray,
    grip_seed: str | Path | None = None,
    *,
    finger_mode: str = "full",
) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    tree = ET.parse(path)
    root = tree.getroot()
    right_hand_joint_names: set[str] = set()
    if finger_mode == "full":
        seed_path = Path(grip_seed) if grip_seed is not None else grip_seed_json_path()
        if seed_path.is_file():
            right_hand_joint_names = set(load_grip_seed(seed_path).right_hand_joint_names)
        else:
            reference = json.loads(grip_reference_json_path().read_text(encoding="utf-8"))
            right_hand_joint_names = set(reference["right_hand_joint_names"])

    joint_by_name = {
        joint.attrib["name"]: joint
        for joint in root.findall(".//joint")
        if "name" in joint.attrib
    }
    for joint_name in right_hand_joint_names:
        joint = joint_by_name.get(joint_name)
        if joint is not None:
            joint.attrib.pop("ref", None)

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


def _overall_ready_qpos(
    xml_path: Path,
    grip_seed: str | Path | None = None,
    *,
    human_root_pos: np.ndarray | None = None,
    human_root_quat: np.ndarray | None = None,
    shuttle_qpos: np.ndarray | None = None,
    attachment_mode: str = "legacy_free",
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos = np.array(model.qpos0, dtype=float)

    root_pos = INITIAL_HUMAN_ROOT_POS if human_root_pos is None else np.asarray(human_root_pos, dtype=float)
    root_quat = INITIAL_HUMAN_ROOT_QUAT if human_root_quat is None else np.asarray(human_root_quat, dtype=float)
    if root_pos.shape != (3,):
        raise ValueError(f"human_root_pos must have shape (3,), got {root_pos.shape}")
    if root_quat.shape != (4,):
        raise ValueError(f"human_root_quat must have shape (4,), got {root_quat.shape}")

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HUMAN_ROOT_FREEJOINT)
    if root_id < 0:
        raise ValueError(f"missing joint {HUMAN_ROOT_FREEJOINT!r}")
    root_adr = int(model.jnt_qposadr[root_id])
    qpos[root_adr : root_adr + 3] = root_pos
    qpos[root_adr + 3 : root_adr + 7] = root_quat

    if attachment_mode == "legacy_free":
        seed_path = Path(grip_seed) if grip_seed is not None else grip_seed_json_path()
        seed = load_grip_seed(seed_path) if seed_path.is_file() else None
        if seed is None:
            reference = json.loads(grip_reference_json_path().read_text(encoding="utf-8"))
            _copy_legacy_reference_hand_qpos(model, qpos, reference)
            _place_racket_at_right_hand(model, qpos, reference)
        else:
            apply_seed_right_hand_joints(seed, model, qpos)
            _place_seed_racket_at_right_hand(model, qpos, seed)
    elif attachment_mode != "exact_child":
        raise ValueError(f"unsupported attachment_mode {attachment_mode!r}")

    shuttle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SHUTTLE_FREEJOINT)
    if shuttle_id < 0:
        raise ValueError(f"missing joint {SHUTTLE_FREEJOINT!r}")
    shuttle_adr = int(model.jnt_qposadr[shuttle_id])
    if shuttle_qpos is None:
        shuttle_state = np.concatenate([INITIAL_SHUTTLE_POS, INITIAL_SHUTTLE_QUAT])
    else:
        shuttle_state = np.asarray(shuttle_qpos, dtype=float)
        if shuttle_state.shape != (7,):
            raise ValueError(f"shuttle_qpos must have shape (7,), got {shuttle_state.shape}")
    qpos[shuttle_adr : shuttle_adr + 7] = shuttle_state
    return qpos


def _copy_legacy_reference_hand_qpos(model: mujoco.MjModel, qpos: np.ndarray, reference: dict[str, object]) -> None:
    reference_qpos = np.asarray(reference["qpos"], dtype=float)
    reference_model = mujoco.MjModel.from_xml_path(str(grip_reference_xml_path()))
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


def _place_seed_racket_at_right_hand(model: mujoco.MjModel, qpos: np.ndarray, seed: GripSeed) -> None:
    seed_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))
    seed_data = mujoco.MjData(seed_model)
    seed_data.qpos[:] = seed.qpos
    seed_data.qvel[:] = seed.qvel
    mujoco.mj_forward(seed_model, seed_data)

    target_data = mujoco.MjData(model)
    target_data.qpos[:] = qpos
    mujoco.mj_forward(model, target_data)

    seed_palm_id = mujoco.mj_name2id(seed_model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    target_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    seed_racket_joint_id = mujoco.mj_name2id(
        seed_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        seed.racket_freejoint_name,
    )
    target_racket_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, RACKET_FREEJOINT)
    if min(seed_palm_id, target_palm_id, seed_racket_joint_id, target_racket_joint_id) < 0:
        raise ValueError("missing palm site or racket freejoint for seed racket placement")

    seed_racket_body_id = int(seed_model.jnt_bodyid[seed_racket_joint_id])
    seed_palm_pos = np.array(seed_data.site_xpos[seed_palm_id], dtype=float)
    seed_palm_rot = np.array(seed_data.site_xmat[seed_palm_id], dtype=float).reshape(3, 3)
    seed_racket_pos = np.array(seed_data.xpos[seed_racket_body_id], dtype=float)
    seed_racket_rot = np.array(seed_data.xmat[seed_racket_body_id], dtype=float).reshape(3, 3)

    palm_to_racket_pos = seed_palm_rot.T @ (seed_racket_pos - seed_palm_pos)
    palm_to_racket_rot = seed_palm_rot.T @ seed_racket_rot

    target_palm_pos = np.array(target_data.site_xpos[target_palm_id], dtype=float)
    target_palm_rot = np.array(target_data.site_xmat[target_palm_id], dtype=float).reshape(3, 3)
    target_racket_pos = target_palm_pos + target_palm_rot @ palm_to_racket_pos
    target_racket_rot = target_palm_rot @ palm_to_racket_rot
    target_racket_quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(target_racket_quat, target_racket_rot.reshape(9))

    racket_adr = int(model.jnt_qposadr[target_racket_joint_id])
    qpos[racket_adr : racket_adr + 7] = np.concatenate([target_racket_pos, target_racket_quat])


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the overall badminton MuJoCo scene.")
    parser.add_argument("--out", type=Path, default=None, help="Output XML path.")
    parser.add_argument(
        "--mode",
        choices=SCENE_MODES,
        default="inspection",
        help="Build a passive inspection scene or actuator/contact-enabled training scene.",
    )
    parser.add_argument(
        "--grip-seed",
        type=Path,
        default=None,
        help="Optional right-hand grip seed JSON used for the ready pose.",
    )
    parser.add_argument(
        "--attachment-mode",
        choices=ATTACHMENT_MODES,
        default="legacy_free",
        help="Use the legacy free racket or the production jointless exact child.",
    )
    parser.add_argument(
        "--finger-mode",
        choices=FINGER_MODES,
        default="full",
        help="Keep the physical finger DOFs or remove their joints/actuators/tendons.",
    )
    parser.add_argument(
        "--racket-attachment-contract",
        type=Path,
        default=None,
        help="Versioned contract JSON for --attachment-mode=exact_child.",
    )
    parser.add_argument(
        "--enable-soft-weld",
        action="store_true",
        help="Add a soft equality weld between the right hand and racket for early curriculum stages.",
    )
    parser.add_argument(
        "--soft-weld-solref",
        default="0.02 1",
        help="MuJoCo solref for --enable-soft-weld.",
    )
    parser.add_argument(
        "--soft-weld-solimp",
        default="0.8 0.95 0.001",
        help="MuJoCo solimp for --enable-soft-weld.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_overall_scene(
        args.out,
        grip_seed=args.grip_seed,
        mode=args.mode,
        attachment_mode=args.attachment_mode,
        finger_mode=args.finger_mode,
        racket_attachment_contract=args.racket_attachment_contract,
        enable_soft_weld=args.enable_soft_weld,
        soft_weld_solref=args.soft_weld_solref,
        soft_weld_solimp=args.soft_weld_solimp,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
