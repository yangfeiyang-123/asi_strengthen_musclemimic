from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import musclemimic_models


def _load_path_helpers():
    if __package__ in {None, ""}:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
    paths = importlib.import_module("src.grip.paths")
    return paths.REPO_ROOT, paths.racket_xml_path, paths.scene_xml_path


REPO_ROOT, racket_xml_path, scene_xml_path = _load_path_helpers()

HAND_GRIP_SITES: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("thirdmc_r", "rh_palm_grip_site", (0.0, 0.0, 0.0)),
    ("distal_thumb_r", "rh_thumb_pad_site", (0.015, -0.018, -0.007)),
    ("2distph_r", "rh_index_pad_site", (0.003, -0.018, 0.0055)),
    ("3distph_r", "rh_middle_pad_site", (0.002, -0.019, 0.003)),
    ("4distph_r", "rh_ring_pad_site", (-0.004, -0.019, 0.003)),
    ("5distph_r", "rh_pinky_pad_site", (-0.005, -0.018, 0.0)),
)

RACKET_HANDLE_SITES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("handle_axis_start_site", (0.0, 0.02, 0.0)),
    ("handle_axis_end_site", (0.0, 0.16, 0.0)),
    ("racket_face_normal_site", (0.0, 0.09, 0.05)),
)

SITE_SIZE = (0.006, 0.006, 0.006)
HAND_SITE_RGBA = (0.1, 0.7, 1.0, 1.0)
RACKET_SITE_RGBA = (1.0, 0.3, 0.1, 1.0)
HANDLE_CONTACT_BIT = "16"
DEFAULT_HANDLE_PARAMS = REPO_ROOT / "configs" / "racket_handle_params.json"

RIGHT_HAND_HANDLE_CONTACT_GEOMS = {
    "1mcskin_coll",
    "distal_thumb_r_coll_2",
    "2mcskin_coll",
    "distph2_r_coll_2",
    "3mcskin_coll",
    "distph3_r_coll_2",
    "4mcskin_coll",
    "distph4_r_coll_2",
    "5mc_r_coll",
    "5distph_r_coll_2",
}
HANDLE_BEVEL_GEOM_NAMES = tuple(f"handle_bevel_{index:02d}" for index in range(8))
HANDLE_CONTACT_GEOMS = set(HANDLE_BEVEL_GEOM_NAMES)


def _add_site(body: mujoco.MjsBody, name: str, pos: tuple[float, float, float], rgba: tuple[float, float, float, float]) -> None:
    if any(site.name == name for site in body.sites):
        return
    body.add_site(name=name, pos=pos, size=SITE_SIZE, rgba=rgba)


def _add_hand_grip_sites(spec: mujoco.MjSpec) -> None:
    for body_name, site_name, pos in HAND_GRIP_SITES:
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"missing MyoFullBody right-hand body: {body_name}")
        _add_site(body, site_name, pos, HAND_SITE_RGBA)


def _add_racket_handle_sites(spec: mujoco.MjSpec) -> None:
    body = spec.body("racket")
    if body is None:
        raise ValueError("missing racket body: racket")
    for site_name, pos in RACKET_HANDLE_SITES:
        _add_site(body, site_name, pos, RACKET_SITE_RGBA)


def _strip_attachment_namespace(root: ET.Element) -> None:
    for elem in root.iter():
        value = elem.attrib.get("name")
        if value is not None and value.startswith("/"):
            elem.set("name", value[1:])


def _remove_external_asset_paths(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    for asset in root.findall("asset"):
        for child in list(asset):
            if child.tag == "mesh" or (child.tag == "texture" and "file" in child.attrib):
                asset.remove(child)


def _remove_mesh_geoms(root: ET.Element) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and (child.attrib.get("type") == "mesh" or "mesh" in child.attrib):
                parent.remove(child)


def _load_handle_params(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError(f"handle params root must be an object: {path}")
    return params


def _handle_geometry(params: dict[str, object]) -> dict[str, float]:
    geometry = params.get("handle_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("handle params missing handle_geometry object")
    return {
        "radius": _positive_float(geometry.get("equivalent_circular_radius_m"), "equivalent_circular_radius_m"),
        "across_flats": _positive_float(geometry.get("octagon_across_flats_m"), "octagon_across_flats_m"),
        "side_length": _positive_float(geometry.get("octagon_side_length_m"), "octagon_side_length_m"),
        "usable_start": _positive_float(geometry.get("usable_grip_start_y_m"), "usable_grip_start_y_m"),
        "usable_end": _positive_float(geometry.get("usable_grip_end_y_m"), "usable_grip_end_y_m"),
        "butt_cap_radius": _positive_float(geometry.get("butt_cap_radius_m"), "butt_cap_radius_m"),
    }


def _handle_contact_params(params: dict[str, object]) -> dict[str, str | float]:
    contact = params.get("contact")
    if not isinstance(contact, dict):
        raise ValueError("handle params missing contact object")
    solref = _number_list(contact.get("solref"), "contact.solref", 2)
    solimp = _number_list(contact.get("solimp"), "contact.solimp", 3)
    return {
        "friction": _friction_string(contact),
        "condim": str(_positive_int(contact.get("condim"), "contact.condim")),
        "solref": _float_string(solref),
        "solimp": _float_string(solimp),
        "margin": _format_float(_nonnegative_float(contact.get("margin_m"), "contact.margin_m")),
    }


def _positive_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a positive number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{context} must be a positive finite number, got {value!r}")
    return number


def _nonnegative_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a nonnegative number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{context} must be a nonnegative finite number, got {value!r}")
    return number


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be a finite number, got {value!r}")
    return number


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    return int(value)


def _number_list(value: object, context: str, expected_len: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_len:
        raise ValueError(f"{context} must be a list of {expected_len} numbers")
    return [_finite_float(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _friction_string(contact: dict[str, object]) -> str:
    values = [
        _positive_float(contact.get("tangential_friction"), "contact.tangential_friction"),
        _nonnegative_float(contact.get("torsional_friction"), "contact.torsional_friction"),
        _nonnegative_float(contact.get("rolling_friction"), "contact.rolling_friction"),
    ]
    return _float_string(values)


def _float_string(values: list[float]) -> str:
    return " ".join(_format_float(value) for value in values)


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def _configure_standard_handle(root: ET.Element, handle_params: dict[str, object]) -> None:
    geometry = _handle_geometry(handle_params)
    contact = _handle_contact_params(handle_params)
    racket_body = root.find(".//body[@name='racket']")
    if racket_body is None:
        raise ValueError("missing racket body in generated XML")

    _configure_butt_cap(racket_body, geometry)
    _configure_handle_fallback_capsule(racket_body, geometry, contact)
    _remove_existing_bevel_geoms(racket_body)
    _add_octagonal_handle_bevels(racket_body, geometry, contact)
    _add_standard_handle_sites(racket_body, handle_params)


def _configure_butt_cap(racket_body: ET.Element, geometry: dict[str, float]) -> None:
    butt_cap = racket_body.find("./geom[@name='butt_cap']")
    if butt_cap is not None:
        butt_cap.set("size", _format_float(geometry["butt_cap_radius"]))


def _configure_handle_fallback_capsule(
    racket_body: ET.Element,
    geometry: dict[str, float],
    contact: dict[str, str | float],
) -> None:
    handle = racket_body.find("./geom[@name='handle_grip']")
    if handle is None:
        raise ValueError("missing handle_grip geom")
    center_y = (geometry["usable_start"] + geometry["usable_end"]) * 0.5
    half_length = (geometry["usable_end"] - geometry["usable_start"]) * 0.5
    handle.set("type", "capsule")
    handle.attrib.pop("fromto", None)
    handle.set("pos", f"0 {_format_float(center_y)} 0")
    handle.set("quat", "0.707107 0.707107 0 0")
    handle.set("size", f"{_format_float(geometry['radius'])} {_format_float(half_length)}")
    handle.set("rgba", "0.08 0.08 0.08 0.35")
    handle.set("group", "4")
    handle.set("contype", "0")
    handle.set("conaffinity", "0")
    handle.set("condim", str(contact["condim"]))
    handle.set("friction", str(contact["friction"]))
    handle.set("solref", str(contact["solref"]))
    handle.set("solimp", str(contact["solimp"]))
    handle.set("margin", str(contact["margin"]))


def _remove_existing_bevel_geoms(racket_body: ET.Element) -> None:
    for geom in list(racket_body.findall("./geom")):
        if geom.attrib.get("name", "").startswith("handle_bevel_"):
            racket_body.remove(geom)


def _add_octagonal_handle_bevels(
    racket_body: ET.Element,
    geometry: dict[str, float],
    contact: dict[str, str | float],
) -> None:
    center_y = (geometry["usable_start"] + geometry["usable_end"]) * 0.5
    half_length = (geometry["usable_end"] - geometry["usable_start"]) * 0.5
    apothem = geometry["across_flats"] * 0.5
    side_half = geometry["side_length"] * 0.5
    thickness_half = 0.001
    for index in range(8):
        theta = index * math.pi / 4.0
        pos = (
            apothem * math.cos(theta),
            center_y,
            apothem * math.sin(theta),
        )
        geom = ET.Element(
            "geom",
            {
                "name": f"handle_bevel_{index:02d}",
                "type": "box",
                "pos": _vec_string(pos),
                "quat": _quat_y_string(math.pi * 0.5 - theta),
                "size": f"{_format_float(side_half)} {_format_float(half_length)} {_format_float(thickness_half)}",
                "rgba": "0.08 0.08 0.08 1",
            },
        )
        _apply_handle_contact_attributes(geom, contact)
        racket_body.append(geom)


def _apply_handle_contact_attributes(geom: ET.Element, contact: dict[str, str | float]) -> None:
    geom.set("contype", HANDLE_CONTACT_BIT)
    geom.set("conaffinity", "0")
    geom.set("condim", str(contact["condim"]))
    geom.set("friction", str(contact["friction"]))
    geom.set("solref", str(contact["solref"]))
    geom.set("solimp", str(contact["solimp"]))
    geom.set("margin", str(contact["margin"]))


def _add_standard_handle_sites(racket_body: ET.Element, handle_params: dict[str, object]) -> None:
    sites = handle_params.get("grip_target_sites")
    if not isinstance(sites, dict):
        raise ValueError("handle params missing grip_target_sites object")
    site_map = {
        "racket_grip_center_site": sites.get("grip_center"),
        "racket_grip_lower_site": sites.get("grip_lower"),
        "racket_grip_upper_site": sites.get("grip_upper"),
        "racket_bevel_top_site": sites.get("bevel_top"),
        "racket_bevel_bottom_site": sites.get("bevel_bottom"),
        "racket_bevel_left_site": sites.get("bevel_left"),
        "racket_bevel_right_site": sites.get("bevel_right"),
    }
    for site_name, value in site_map.items():
        pos = _number_list(value, site_name, 3)
        existing = racket_body.find(f"./site[@name='{site_name}']")
        if existing is None:
            existing = ET.SubElement(racket_body, "site", {"name": site_name})
        existing.set("pos", _float_string(pos))
        existing.set("size", "0.004")
        existing.set("rgba", "1 0.55 0.05 1")


def _vec_string(values: tuple[float, float, float]) -> str:
    return " ".join(_format_float(value) for value in values)


def _quat_y_string(angle: float) -> str:
    half = angle * 0.5
    return f"{_format_float(math.cos(half))} 0 {_format_float(math.sin(half))} 0"


def _is_right_hand_handle_contact_geom(name: str) -> bool:
    return name in RIGHT_HAND_HANDLE_CONTACT_GEOMS


def _configure_handle_contact_filter(root: ET.Element) -> None:
    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")
        if name in HANDLE_CONTACT_GEOMS:
            geom.set("contype", HANDLE_CONTACT_BIT)
            geom.set("conaffinity", "0")
        elif _is_right_hand_handle_contact_geom(name):
            geom.set("conaffinity", HANDLE_CONTACT_BIT)
            geom.set("condim", "4")
            geom.set("friction", "1.4 0.03 0.003")
            geom.set("solref", "0.004 1")
            geom.set("solimp", "0.90 0.97 0.002")
            geom.set("margin", "0.001")


def _sort_attributes(root: ET.Element) -> None:
    for elem in root.iter():
        if len(elem.attrib) > 1:
            attributes = sorted(elem.attrib.items())
            elem.attrib.clear()
            elem.attrib.update(attributes)


def _postprocess_attached_xml(path: Path, handle_params: dict[str, object]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    _strip_attachment_namespace(root)
    _remove_external_asset_paths(root)
    _remove_mesh_geoms(root)
    _configure_standard_handle(root, handle_params)
    _configure_handle_contact_filter(root)
    _sort_attributes(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_scene(
    output_xml: Path | str | None = None,
    handle_params: Path | str = DEFAULT_HANDLE_PARAMS,
) -> Path:
    out_path = Path(output_xml) if output_xml is not None else scene_xml_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle_params_data = _load_handle_params(Path(handle_params))

    myofullbody_path = Path(musclemimic_models.get_xml_path("myofullbody"))
    base_spec = mujoco.MjSpec.from_file(str(myofullbody_path))
    racket_spec = mujoco.MjSpec.from_file(str(racket_xml_path()))

    _add_hand_grip_sites(base_spec)
    _add_racket_handle_sites(racket_spec)

    racket_frame = base_spec.worldbody.add_frame(name="racket_mount_frame")
    base_spec.attach(racket_spec, frame=racket_frame)

    base_spec.compile()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "right_hand_racket_grip_scene.xml"
        base_spec.to_file(str(tmp_path))
        _postprocess_attached_xml(tmp_path, handle_params_data)
        mujoco.MjModel.from_xml_path(str(tmp_path))
        out_path.write_bytes(tmp_path.read_bytes())

    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the right-hand racket grip MuJoCo scene.")
    parser.add_argument("--out", type=Path, default=scene_xml_path(), help="Output XML path.")
    parser.add_argument(
        "--handle-params",
        type=Path,
        default=DEFAULT_HANDLE_PARAMS,
        help="Standard racket handle JSON parameter path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_scene(output_xml=args.out, handle_params=args.handle_params)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
