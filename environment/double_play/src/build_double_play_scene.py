"""Build the two-player badminton rally scene.

Two MyoFullBody players stand in opposite backcourts of the BWF court, each
with the production jointless exact-child racket on the right hand, sharing
one shuttlecock.  Player 1 keeps the exact names of the single-player
incoming-hit scene (``root``, ``overall_racket``, ``overall_shuttle``, ...);
player 2 is the same human+racket subtree attached with the ``p2_`` prefix and
mirrored through a 180-degree rotation about the world z-axis (the net line).

Build steps:
1. reuse ``build_overall_scene`` (training / exact_child / fingers removed)
   with player 1 placed in the backcourt;
2. attach a second finger-free MyoFullBody (own floor/light removed) carrying
   its own exact-child racket, prefixed ``p2_``;
3. post-process the serialized XML: strip the inner-attach ``/`` namespace,
   move the p2 stringbed ground proxy to the dedicated ground-only collision
   bit, rerun the MJX ellipsoid compatibility pass for the p2 body, enlarge
   contact buffers, and write the mirrored ``double_ready`` keyframe.

Contact-bit summary (identical for both players):
  humans bit 1, racket frames bit 4, stringbed ground proxies bit 16 (floor
  only), MJX-compat ellipsoids bit 8, cork conaffinity 13 = 1|4|8 so it meets
  the floor/net, both racket frames, and the compat ellipsoids.  The stringbed
  itself has no native shuttle contact on either racket: the custom
  force/event model in ``rally_physics`` is the only shuttle-stringbed model.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import musclemimic_models
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.build_overall_environment import (
    _add_hand_grip_sites,
    _attach_exact_child_racket,
    _make_mjx_compatible,
    _sort_attributes,
    _write_prefixed_component_xml,
    build_overall_scene,
)
from environment.overall_environment.src.racket_attachment import (
    RacketAttachmentContract,
    load_racket_attachment_contract,
)
from musclemimic.environments.humanoids.myofullbody import remove_finger_dofs

DOUBLE_READY_KEYFRAME = "double_ready"
P2_PREFIX = "p2_"
SINGLE_READY_KEYFRAME = "overall_ready"

# Backcourt ready placement: between the doubles long service line (|x|=5.92)
# and the short service line, deep enough for clear-to-clear rallies.
P1_ROOT_POS = np.array([-4.6, 0.0, 1.0], dtype=float)
P1_ROOT_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=float)
# Placeholder only: every episode reset overwrites the shuttle free joint.
INITIAL_SHUTTLE_QPOS = np.array(
    [-4.0, -1.0, 0.034, np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=float
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "racket_attachment"
    / "forehand_clear_rigid_v4_custom.json"
)
NCONMAX = 4000
NJMAX = 10000


def default_double_play_scene_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "double_play_scene.xml"


def mirror_free_joint_qpos(qpos7: np.ndarray) -> np.ndarray:
    """Rotate a free-joint (pos, wxyz quat) state 180 degrees about world z."""
    value = np.asarray(qpos7, dtype=float)
    if value.shape != (7,):
        raise ValueError(f"free joint qpos must have shape (7,), got {value.shape}")
    mirrored = value.copy()
    mirrored[0] = -value[0]
    mirrored[1] = -value[1]
    z180 = np.array([0.0, 0.0, 0.0, 1.0])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, z180, value[3:7])
    mirrored[3:7] = rotated
    return mirrored


def _prepare_second_player_spec(contract: RacketAttachmentContract) -> mujoco.MjSpec:
    """Finger-free MyoFullBody with its exact-child racket, world furniture removed."""
    spec = mujoco.MjSpec.from_file(str(musclemimic_models.get_xml_path("myofullbody")))
    remove_finger_dofs(spec)
    _add_hand_grip_sites(spec)
    worldbody = spec.worldbody
    for geom in list(worldbody.geoms):
        spec.delete(geom)  # the player's own floor plane
    for light in list(worldbody.lights):
        spec.delete(light)
    for camera in list(worldbody.cameras):
        spec.delete(camera)
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefixed_racket = Path(tmp_dir) / "racket_prefixed.xml"
        _write_prefixed_component_xml(contract.asset_path, prefixed_racket, "overall")
        racket_spec = mujoco.MjSpec.from_file(str(prefixed_racket))
        _attach_exact_child_racket(spec, racket_spec, contract)
    return spec


def _strip_inner_attach_namespace(root: ET.Element) -> None:
    """Rewrite ``p2_/name`` references left by the nested racket attach.

    ``p2_/main`` would collide with the human's own ``p2_main`` default class,
    so colliding classes are renamed to ``p2_attached_*`` instead.
    """
    class_names = {
        default.attrib.get("class")
        for default in root.findall(".//default")
        if default.attrib.get("class")
    }
    rename: dict[str, str] = {}
    for name in sorted(class_names):
        if f"{P2_PREFIX}/" not in name:
            continue
        target = name.replace(f"{P2_PREFIX}/", P2_PREFIX)
        if target in class_names:
            target = name.replace(f"{P2_PREFIX}/", f"{P2_PREFIX}attached_")
        rename[name] = target
    for elem in root.iter():
        for attr, value in list(elem.attrib.items()):
            if value in rename:
                elem.set(attr, rename[value])
            elif f"{P2_PREFIX}/" in value:
                elem.set(attr, value.replace(f"{P2_PREFIX}/", P2_PREFIX))


def _isolate_p2_stringbed_ground_proxy(root: ET.Element, contract: RacketAttachmentContract) -> None:
    proxy_name = f"{P2_PREFIX}overall_{contract.stringbed_proxy_geom_name}"
    proxy = root.find(f".//geom[@name='{proxy_name}']")
    if proxy is None:
        raise ValueError(f"double scene is missing stringbed proxy geom {proxy_name!r}")
    bit = str(int(contract.stringbed_ground_collision_bit))
    proxy.set("contype", bit)
    proxy.set("conaffinity", bit)


def _restore_post_mjx_bits(path: Path, contract: RacketAttachmentContract) -> None:
    """Re-OR bits that ``_make_mjx_compatible`` resets to its base values.

    The MJX compatibility pass rewrites the cork conaffinity to ``1|8`` and the
    base floor conaffinity to ``5|8``; in the single-scene pipeline the racket
    bit (cork) and the stringbed-ground bit (floor) are OR-ed on afterwards, so
    the double-scene pipeline must do the same after rerunning the pass.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    cork = root.find(".//geom[@name='overall_cork_collision']")
    if cork is None:
        raise ValueError("double scene is missing geom 'overall_cork_collision'")
    affinity = int(cork.attrib.get("conaffinity", "0"))
    cork.set("conaffinity", str(affinity | int(contract.racket_collision_bit)))
    ground_bit = int(contract.stringbed_ground_collision_bit)
    for floor_name in ("floor", "overall_floor_collision"):
        floor = root.find(f".//geom[@name='{floor_name}']")
        if floor is None:
            raise ValueError(f"double scene is missing geom {floor_name!r}")
        floor.set("conaffinity", str(int(floor.attrib.get("conaffinity", "0")) | ground_bit))
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _enlarge_contact_buffers(root: ET.Element) -> None:
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nconmax", str(NCONMAX))
    size.set("njmax", str(NJMAX))


def _polish_visuals(root: ET.Element) -> None:
    """Rendering-only fixes; zero effect on physics.

    1. The green court box top and the dark base plane are exactly coplanar
       (both z=0), which z-fights into large dark patches at higher render
       resolutions.  A visual-only 1 mm green cover (contype/conaffinity 0)
       sits just above both, still below the 2 mm court lines.
    2. ``overall_top_light`` is a 4 m point light over a 15 m court whose
       stretched shadow map draws jagged dark wedges; shadows are disabled.
    3. The offscreen framebuffer is enlarged so 1080p captures work.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("double scene has no worldbody")
    if worldbody.find("geom[@name='overall_floor_visual_cover']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "conaffinity": "0",
                "contype": "0",
                "group": "1",
                "material": "overall_mat_floor",
                "name": "overall_floor_visual_cover",
                "pos": "0 0 0.0006",
                "size": "7.7 4.05 0.0005",
                "type": "box",
            },
        )
    for light in root.iter("light"):
        light.set("castshadow", "false")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")
    global_elem.set("offwidth", "1920")
    global_elem.set("offheight", "1080")


def _joint_qpos_slice(model: mujoco.MjModel, joint_id: int) -> tuple[int, int]:
    joint_type = int(model.jnt_type[joint_id])
    width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (
        4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
    )
    return int(model.jnt_qposadr[joint_id]), width


def _double_ready_qpos(
    single_model: mujoco.MjModel,
    double_model: mujoco.MjModel,
) -> np.ndarray:
    """Map the single-scene ready keyframe onto both players by joint name."""
    key_id = mujoco.mj_name2id(single_model, mujoco.mjtObj.mjOBJ_KEY, SINGLE_READY_KEYFRAME)
    if key_id < 0:
        raise ValueError(f"single scene is missing keyframe {SINGLE_READY_KEYFRAME!r}")
    single_qpos = np.asarray(single_model.key_qpos[key_id], dtype=float)
    qpos = np.asarray(double_model.qpos0, dtype=float).copy()

    for joint_id in range(single_model.njnt):
        name = mujoco.mj_id2name(single_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        adr, width = _joint_qpos_slice(single_model, joint_id)
        value = single_qpos[adr : adr + width]
        for target_name, mirror in ((name, False), (P2_PREFIX + name, True)):
            target_id = mujoco.mj_name2id(double_model, mujoco.mjtObj.mjOBJ_JOINT, target_name)
            if target_id < 0:
                if target_name == P2_PREFIX + "overall_shuttle_free":
                    continue  # the single shuttle is owned by the base scene
                raise ValueError(f"double scene is missing joint {target_name!r}")
            target_adr, target_width = _joint_qpos_slice(double_model, target_id)
            if target_width != width:
                raise ValueError(f"joint width mismatch for {target_name!r}")
            mapped = value.copy()
            if mirror and width == 7:
                mapped = mirror_free_joint_qpos(mapped)
            qpos[target_adr : target_adr + target_width] = mapped
    return qpos


def build_double_play_scene(
    output_xml: str | Path | None = None,
    *,
    racket_attachment_contract: str | Path | RacketAttachmentContract | None = None,
    p1_root_pos: np.ndarray | None = None,
    shuttle_qpos: np.ndarray | None = None,
    keep_intermediate_single_scene: bool = False,
) -> Path:
    """Compose and write the two-player rally scene; returns the XML path."""
    out_path = Path(output_xml) if output_xml is not None else default_double_play_scene_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(racket_attachment_contract, RacketAttachmentContract):
        contract = racket_attachment_contract
        contract.verify_asset()
    else:
        contract = load_racket_attachment_contract(
            racket_attachment_contract if racket_attachment_contract is not None else DEFAULT_CONTRACT_PATH
        )

    root_pos = P1_ROOT_POS if p1_root_pos is None else np.asarray(p1_root_pos, dtype=float)
    shuttle_state = (
        INITIAL_SHUTTLE_QPOS if shuttle_qpos is None else np.asarray(shuttle_qpos, dtype=float)
    )

    # Step 1: the proven single-player composition with P1 in the backcourt.
    # Building into the output directory also installs the portable MSK assets.
    single_path = out_path.parent / f"_single_base_{out_path.stem}.xml"
    build_overall_scene(
        single_path,
        mode="training",
        attachment_mode="exact_child",
        finger_mode="removed",
        racket_attachment_contract=contract,
        enable_person_racket_contact=False,
        enable_soft_weld=False,
        human_root_pos=root_pos,
        human_root_quat=P1_ROOT_QUAT,
        shuttle_qpos=shuttle_state,
    )
    single_model = mujoco.MjModel.from_xml_path(str(single_path))

    # Step 2: attach the mirrored second player.
    base = mujoco.MjSpec.from_file(str(single_path))
    for key in list(base.keys):
        base.delete(key)  # stale nq; the double keyframe is rebuilt below
    p2_spec = _prepare_second_player_spec(contract)
    frame = base.worldbody.add_frame(name="p2_mount")
    base.attach(p2_spec, frame=frame, prefix=P2_PREFIX)
    base.compile()
    base.to_file(str(out_path))

    # Step 3: post-process the serialized double scene.
    tree = ET.parse(out_path)
    root = tree.getroot()
    _strip_inner_attach_namespace(root)
    _isolate_p2_stringbed_ground_proxy(root, contract)
    _enlarge_contact_buffers(root)
    _polish_visuals(root)
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    _sort_attributes(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    _make_mjx_compatible(out_path)
    _restore_post_mjx_bits(out_path, contract)

    double_model = mujoco.MjModel.from_xml_path(str(out_path))
    qpos = _double_ready_qpos(single_model, double_model)
    tree = ET.parse(out_path)
    root = tree.getroot()
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": DOUBLE_READY_KEYFRAME, "qpos": " ".join(f"{value:.17g}" for value in qpos)},
    )
    _sort_attributes(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    mujoco.MjModel.from_xml_path(str(out_path))  # fail-closed load check
    if not keep_intermediate_single_scene:
        single_path.unlink(missing_ok=True)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two-player badminton rally scene.")
    parser.add_argument("--out", type=Path, default=None, help="Output XML path.")
    parser.add_argument(
        "--racket-attachment-contract",
        type=Path,
        default=None,
        help="Versioned exact-child racket attachment contract JSON (defaults to v4).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_double_play_scene(
        args.out,
        racket_attachment_contract=args.racket_attachment_contract,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
