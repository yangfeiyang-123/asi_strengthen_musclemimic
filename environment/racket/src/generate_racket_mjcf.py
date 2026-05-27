"""Generate MuJoCo MJCF assets for the badminton racket design dossier.

The generated files use a procedural capsule/cylinder approximation rather than
private CAD geometry. Coordinates are SI units and follow the dossier convention:
+X across string bed, +Y butt-to-tip, +Z normal to the string-bed plane.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDLE_PARAMS = REPO_ROOT / "configs" / "racket_handle_params.json"


def fmt(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) >= 1e-9 else "0"


def v3(xs: Iterable[float]) -> str:
    return " ".join(fmt(float(x)) for x in xs)


def capsule(name: str, p0: Tuple[float, float, float], p1: Tuple[float, float, float], radius: float,
            rgba: str, contype: int = 0, conaffinity: int = 0, cls: str | None = None) -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return (f'      <geom name="{name}"{class_attr} type="capsule" fromto="{v3(p0)} {v3(p1)}" '
            f'size="{fmt(radius)}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"/>')


def sphere(name: str, pos: Tuple[float, float, float], radius: float, rgba: str,
           contype: int = 0, conaffinity: int = 0, cls: str | None = None) -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return (f'      <geom name="{name}"{class_attr} type="sphere" pos="{v3(pos)}" size="{fmt(radius)}" '
            f'rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"/>')


def box(name: str, pos: Tuple[float, float, float], quat: str, size: Tuple[float, float, float],
        rgba: str, contype: int = 0, conaffinity: int = 0, cls: str | None = None,
        indent: str = "      ") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return (f'{indent}<geom name="{name}"{class_attr} type="box" pos="{v3(pos)}" quat="{quat}" '
            f'size="{v3(size)}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"/>')


def quat_y(angle_rad: float) -> str:
    half = angle_rad * 0.5
    return f"{fmt(math.cos(half))} 0 {fmt(math.sin(half))} 0"


def load_standard_handle_params() -> dict:
    with DEFAULT_HANDLE_PARAMS.open("r", encoding="utf-8") as f:
        return json.load(f)


def standard_handle_geometry(params: dict) -> dict[str, float]:
    standard = params.get("standard_handle") or load_standard_handle_params()
    geometry = standard["handle_geometry"]
    contact = standard["contact"]
    return {
        "radius": float(geometry["equivalent_circular_radius_m"]),
        "across_flats": float(geometry["octagon_across_flats_m"]),
        "side_length": float(geometry["octagon_side_length_m"]),
        "usable_start": float(geometry["usable_grip_start_y_m"]),
        "usable_end": float(geometry["usable_grip_end_y_m"]),
        "butt_cap_radius": float(geometry["butt_cap_radius_m"]),
        "condim": int(contact["condim"]),
        "friction": " ".join(fmt(float(value)) for value in (
            contact["tangential_friction"],
            contact["torsional_friction"],
            contact["rolling_friction"],
        )),
        "solref": " ".join(fmt(float(value)) for value in contact["solref"]),
        "solimp": " ".join(fmt(float(value)) for value in contact["solimp"]),
        "margin": fmt(float(contact["margin_m"])),
    }


def standard_handle_geoms(params: dict, grip_rgba: str, indent: str = "      ") -> List[str]:
    h = standard_handle_geometry(params)
    grip_rgb = grip_rgba.split()[:3]
    fallback_rgba = " ".join([*grip_rgb, "0"])
    center_y = 0.5 * (h["usable_start"] + h["usable_end"])
    half_length = 0.5 * (h["usable_end"] - h["usable_start"])
    apothem = 0.5 * h["across_flats"]
    side_half = 0.5 * h["side_length"]
    thickness_half = 0.001

    lines = [
        sphere("butt_cap", (0.0, 0.0, 0.0), h["butt_cap_radius"], grip_rgba, 1, 1, "frame_contact").replace("      ", indent, 1),
        (
            f'{indent}<geom name="handle_grip" type="capsule" pos="0 {fmt(center_y)} 0" '
            f'quat="0.707107 0.707107 0 0" size="{fmt(h["radius"])} {fmt(half_length)}" '
            f'rgba="{fallback_rgba}" contype="0" conaffinity="0" condim="{h["condim"]}" '
            f'friction="{h["friction"]}" solref="{h["solref"]}" solimp="{h["solimp"]}" '
            f'margin="{h["margin"]}"/>'
        ),
    ]
    for index in range(8):
        theta = index * math.pi / 4.0
        lines.append(
            box(
                f"handle_bevel_{index:02d}",
                (apothem * math.cos(theta), center_y, apothem * math.sin(theta)),
                quat_y(math.pi * 0.5 - theta),
                (side_half, half_length, thickness_half),
                grip_rgba,
                1,
                1,
                "frame_contact",
                indent,
            )
        )
    return lines


def ellipsoid_head_frame(g: dict, rgba: str, contype: int = 1, conaffinity: int = 1, name_prefix="head_frame",
                         y_offset: float = 0.0) -> List[str]:
    cx_y = g["head_center_y"] - y_offset
    a = g["head_outer_half_width"]
    b = g["head_outer_half_length"]
    r = g["head_frame_radius"]
    segs = 40
    lines = []
    pts = []
    # Start at lower throat-ish part and go around the ellipse.
    for i in range(segs):
        th = 2 * math.pi * i / segs
        x = a * math.cos(th)
        y = cx_y + b * math.sin(th)
        pts.append((x, y, 0.0))
    for i in range(segs):
        p0 = pts[i]
        p1 = pts[(i + 1) % segs]
        lines.append(capsule(f"{name_prefix}_{i:02d}", p0, p1, r, rgba, contype, conaffinity, cls="frame_contact"))
    return lines


def string_geoms(g: dict, s: dict, rgba: str, y_offset: float = 0.0) -> List[str]:
    a = g["stringbed_half_width"]
    b = g["stringbed_half_length"]
    cy = g["stringbed_center_y"] - y_offset
    main_n = int(s["main_string_count"])
    cross_n = int(s["cross_string_count"])
    radius = float(s["string_diameter_m"]) / 2.0
    lines = []

    # Mains run along +Y. Keep endpoints slightly away from zero-length edge strings.
    for i in range(main_n):
        x = -0.94 * a + (1.88 * a) * i / (main_n - 1)
        y_half = b * math.sqrt(max(0.0, 1.0 - (x / a) ** 2))
        p0 = (x, cy - y_half, 0.00055)
        p1 = (x, cy + y_half, 0.00055)
        lines.append(capsule(f"string_main_{i:02d}", p0, p1, radius, rgba, 0, 0, cls="string_visual"))

    # Crosses run along +X. Offset in -Z to reduce visual z-fighting at intersections.
    for j in range(cross_n):
        dy = -0.94 * b + (1.88 * b) * j / (cross_n - 1)
        x_half = a * math.sqrt(max(0.0, 1.0 - (dy / b) ** 2))
        y = cy + dy
        p0 = (-x_half, y, -0.00055)
        p1 = (x_half, y, -0.00055)
        lines.append(capsule(f"string_cross_{j:02d}", p0, p1, radius, rgba, 0, 0, cls="string_visual"))
    return lines


def common_defaults() -> str:
    return """  <compiler angle=\"degree\" autolimits=\"true\"/>
  <option timestep=\"0.0005\" integrator=\"implicitfast\" gravity=\"0 0 -9.81\"/>
  <default>
    <geom condim=\"3\" friction=\"0.8 0.02 0.001\" solref=\"0.0015 1\" solimp=\"0.95 0.99 0.001\"/>
    <site rgba=\"1 0.3 0.1 1\"/>
    <default class=\"frame_contact\">
      <geom contype=\"1\" conaffinity=\"1\"/>
    </default>
    <default class=\"string_visual\">
      <geom contype=\"0\" conaffinity=\"0\"/>
    </default>
    <default class=\"stringbed_ground_contact\">
      <geom contype=\"1\" conaffinity=\"1\" group=\"1\" condim=\"4\" friction=\"1.1 0.05 0.003\" solref=\"0.004 1\" solimp=\"0.92 0.98 0.001\"/>
    </default>
  </default>
"""


def rigid_mjcf(params: dict) -> str:
    g = params["geometry_m"]
    m = params["mass_properties"]
    s = params["stringbed"]
    ap = params["appearance"]
    frame_rgba = v3(ap["frame_rgba"])
    shaft_rgba = v3(ap["shaft_rgba"])
    grip_rgba = v3(ap["grip_rgba"])
    string_rgba = v3(ap["string_rgba"])
    proxy_rgba = v3(ap["proxy_rgba"])

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<mujoco model="badminton_racket_rigid">')
    lines.append(common_defaults().rstrip())
    lines.append('  <worldbody>')
    lines.append('    <body name="racket" pos="0 0 1.2">')
    lines.append('      <freejoint name="racket_free"/>')
    lines.append(f'      <inertial pos="{v3(m["mujoco_inertial_pos"])}" mass="{fmt(m["mass_kg"])}" diaginertia="{v3(m["mujoco_diaginertia"])}"/>')
    lines.append('      <site name="grip_pose_site" pos="0 0.09 0" size="0.006"/>')
    lines.append('      <site name="butt_site" pos="0 0 0" size="0.004" rgba="0.1 1 0.1 1"/>')
    lines.append(f'      <site name="stringbed_center_site" pos="0 {fmt(g["stringbed_center_y"])} 0" size="0.006" rgba="1 0.1 0.1 1"/>')
    lines.append(f'      <site name="head_tip_site" pos="0 {fmt(g["overall_length"])} 0" size="0.004" rgba="1 1 0 1"/>')
    lines.extend(standard_handle_geoms(params, grip_rgba))
    lines.append(capsule("shaft", (0.0, g["shaft_start_y"], 0.0), (0.0, g["shaft_end_y"], 0.0), g["shaft_radius"], shaft_rgba, 1, 1, "frame_contact"))
    lines.append(capsule("throat_left", (0.0, g["throat_start_y"], 0.0), (-g["throat_half_width"], g["throat_end_y"], 0.0), g["head_frame_radius"]*0.75, frame_rgba, 1, 1, "frame_contact"))
    lines.append(capsule("throat_right", (0.0, g["throat_start_y"], 0.0), (g["throat_half_width"], g["throat_end_y"], 0.0), g["head_frame_radius"]*0.75, frame_rgba, 1, 1, "frame_contact"))
    lines.extend(ellipsoid_head_frame(g, frame_rgba, 1, 1))
    lines.extend(string_geoms(g, s, string_rgba))
    lines.append(f'      <geom name="stringbed_ground_contact_proxy" class="stringbed_ground_contact" type="box" pos="0 {fmt(g["stringbed_center_y"])} 0" size="{fmt(g["stringbed_half_width"] + 0.002)} {fmt(g["stringbed_half_length"] + 0.0035)} 0.003" rgba="0.1 0.45 1 0.035" condim="4" friction="1.1 0.05 0.003"/>')
    lines.append(f'      <geom name="stringbed_proxy_visual" type="box" pos="0 {fmt(g["stringbed_center_y"])} 0" size="{fmt(g["stringbed_half_width"])} {fmt(g["stringbed_half_length"])} {fmt(g["stringbed_proxy_thickness"])}" rgba="{proxy_rgba}" contype="0" conaffinity="0"/>')
    lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"


def flex_mjcf(params: dict) -> str:
    g = params["geometry_m"]
    s = params["stringbed"]
    ap = params["appearance"]
    f = params["flex_proxy"]
    frame_rgba = v3(ap["frame_rgba"])
    shaft_rgba = v3(ap["shaft_rgba"])
    grip_rgba = v3(ap["grip_rgba"])
    string_rgba = v3(ap["string_rgba"])
    proxy_rgba = v3(ap["proxy_rgba"])
    hinge_y = f["hinge_y_m"]

    # A two-body proxy: handle/lower shaft + head/upper shaft. Mass properties are approximate;
    # use the rigid model for exact swingweight, flex proxy for qualitative bending.
    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<mujoco model="badminton_racket_flex_proxy">')
    lines.append(common_defaults().rstrip())
    lines.append('  <worldbody>')
    lines.append('    <body name="racket_handle" pos="0 0 1.2">')
    lines.append('      <freejoint name="racket_handle_free"/>')
    lines.append('      <inertial pos="0 0.135 0" mass="0.035" diaginertia="0.00048 0.00004 0.00048"/>')
    lines.append('      <site name="grip_pose_site" pos="0 0.09 0" size="0.006"/>')
    lines.append('      <site name="butt_site" pos="0 0 0" size="0.004" rgba="0.1 1 0.1 1"/>')
    lines.extend(standard_handle_geoms(params, grip_rgba))
    lines.append(capsule("lower_shaft", (0.0, g["shaft_start_y"], 0.0), (0.0, hinge_y, 0.0), g["shaft_radius"], shaft_rgba, 1, 1, "frame_contact"))
    lines.append(f'      <body name="racket_head" pos="0 {fmt(hinge_y)} 0">')
    lines.append(f'        <joint name="shaft_flex_x" type="hinge" axis="1 0 0" stiffness="{fmt(f["out_of_plane_stiffness_nm_per_rad"])}" damping="{fmt(f["out_of_plane_damping_nm_s_per_rad"])}" limited="true" range="-{fmt(f["joint_limit_deg"])} {fmt(f["joint_limit_deg"])}"/>')
    lines.append(f'        <joint name="shaft_flex_z" type="hinge" axis="0 0 1" stiffness="{fmt(f["in_plane_stiffness_nm_per_rad"])}" damping="{fmt(f["in_plane_damping_nm_s_per_rad"])}" limited="true" range="-{fmt(f["joint_limit_deg"])} {fmt(f["joint_limit_deg"])}"/>')
    lines.append('        <inertial pos="0 0.190 0" mass="0.055" diaginertia="0.00175 0.00014 0.00170"/>')
    # Child local coordinates subtract hinge_y from global Y.
    yo = hinge_y
    lines.append(f'        <site name="stringbed_center_site" pos="0 {fmt(g["stringbed_center_y"]-yo)} 0" size="0.006" rgba="1 0.1 0.1 1"/>')
    lines.append(f'        <site name="head_tip_site" pos="0 {fmt(g["overall_length"]-yo)} 0" size="0.004" rgba="1 1 0 1"/>')
    lines.append(capsule("upper_shaft", (0.0, 0.0, 0.0), (0.0, g["shaft_end_y"]-yo, 0.0), g["shaft_radius"], shaft_rgba, 1, 1, "frame_contact").replace('      <geom', '        <geom'))
    lines.append(capsule("throat_left", (0.0, g["throat_start_y"]-yo, 0.0), (-g["throat_half_width"], g["throat_end_y"]-yo, 0.0), g["head_frame_radius"]*0.75, frame_rgba, 1, 1, "frame_contact").replace('      <geom', '        <geom'))
    lines.append(capsule("throat_right", (0.0, g["throat_start_y"]-yo, 0.0), (g["throat_half_width"], g["throat_end_y"]-yo, 0.0), g["head_frame_radius"]*0.75, frame_rgba, 1, 1, "frame_contact").replace('      <geom', '        <geom'))
    for line in ellipsoid_head_frame(g, frame_rgba, 1, 1, y_offset=yo):
        lines.append(line.replace('      <geom', '        <geom'))
    for line in string_geoms(g, s, string_rgba, y_offset=yo):
        lines.append(line.replace('      <geom', '        <geom'))
    lines.append(f'        <geom name="stringbed_proxy_visual" type="box" pos="0 {fmt(g["stringbed_center_y"]-yo)} 0" size="{fmt(g["stringbed_half_width"])} {fmt(g["stringbed_half_length"])} {fmt(g["stringbed_proxy_thickness"])}" rgba="{proxy_rgba}" contype="0" conaffinity="0"/>')
    lines.append('      </body>')
    lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"


def scene_mjcf() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="racket_stringbed_debug_scene">
  <include file="badminton_racket_rigid.xml"/>
  <statistic center="0 0 1.0" extent="1.0"/>
  <visual>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <worldbody>
    <light name="key" pos="0 -1 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.01" rgba="0.55 0.6 0.55 1"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with open(root / "params" / "racket_nominal.json", "r", encoding="utf-8") as f:
        params = json.load(f)
    (root / "assets" / "badminton_racket_rigid.xml").write_text(rigid_mjcf(params), encoding="utf-8")
    (root / "assets" / "badminton_racket_flex_proxy.xml").write_text(flex_mjcf(params), encoding="utf-8")
    (root / "assets" / "scene_racket_stringbed_debug.xml").write_text(scene_mjcf(), encoding="utf-8")


if __name__ == "__main__":
    main()
