"""Generate MuJoCo MJCF assets for a BWF-standard badminton court.

Run from the package root:

    python src/generate_court_mjcf.py

This writes:
    assets/badminton_court_bwf_visual.xml
    assets/badminton_court_bwf_collision_net.xml

The visual asset disables net cloth collision; the collision-net asset enables a
thin proxy wall and top-cord collision suitable for shuttlecock contact tests.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

try:
    from court_geometry import CourtParams
except ImportError:  # allow running from package root
    sys.path.append(str(Path(__file__).resolve().parent))
    from court_geometry import CourtParams


def fmt(x: float) -> str:
    return f"{x:.6g}"


def geom_box(name: str, pos, size, material: str, group=1,
             contype=0, conaffinity=0, extra: str = "") -> str:
    return (
        f'      <geom name="{name}" type="box" pos="{fmt(pos[0])} {fmt(pos[1])} {fmt(pos[2])}" '
        f'size="{fmt(size[0])} {fmt(size[1])} {fmt(size[2])}" material="{material}" '
        f'group="{group}" contype="{contype}" conaffinity="{conaffinity}"{extra}/>\n'
    )


def geom_capsule(name: str, p0, p1, radius: float, material: str, group=1,
                 contype=0, conaffinity=0, extra: str = "") -> str:
    return (
        f'      <geom name="{name}" type="capsule" fromto="'
        f'{fmt(p0[0])} {fmt(p0[1])} {fmt(p0[2])} {fmt(p1[0])} {fmt(p1[1])} {fmt(p1[2])}" '
        f'size="{fmt(radius)}" material="{material}" group="{group}" '
        f'contype="{contype}" conaffinity="{conaffinity}"{extra}/>\n'
    )


def top_z(c: CourtParams, y: float) -> float:
    return c.net_top_height(y)


def generate_mjcf(c: CourtParams, enable_net_collision: bool = False) -> str:
    floor_half_x = 7.70
    floor_half_y = 4.05
    floor_z = -0.005
    floor_thickness = 0.010
    line_z = 0.002
    line_t = 0.002
    net_pitch = 0.1525  # visual LOD; official mesh pitch is 15-20 mm.
    net_thick_x = 0.006

    net_contype = 2 if enable_net_collision else 0
    net_conaffinity = 1 if enable_net_collision else 0

    xml = []
    xml.append('<mujoco model="bwf_badminton_court">\n')
    xml.append('  <compiler angle="degree" autolimits="true"/>\n')
    xml.append('  <option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>\n')
    xml.append('  <asset>\n')
    xml.append('    <material name="mat_floor" rgba="0.05 0.23 0.12 1"/>\n')
    xml.append('    <material name="mat_line" rgba="1.0 1.0 0.92 1"/>\n')
    xml.append('    <material name="mat_net_cord" rgba="0.03 0.03 0.035 1"/>\n')
    xml.append('    <material name="mat_net_tape" rgba="1.0 1.0 1.0 1"/>\n')
    xml.append('    <material name="mat_post" rgba="0.55 0.55 0.58 1"/>\n')
    xml.append('    <material name="mat_net_proxy" rgba="0.02 0.02 0.02 0.14"/>\n')
    xml.append('  </asset>\n')
    xml.append('  <default>\n')
    xml.append('    <geom solref="0.003 1" solimp="0.95 0.99 0.001" friction="1.0 0.005 0.0001"/>\n')
    xml.append('  </default>\n')
    xml.append('  <worldbody>\n')
    xml.append('    <body name="court_static" pos="0 0 0">\n')
    xml.append(geom_box(
        "floor_collision", (0, 0, floor_z),
        (floor_half_x, floor_half_y, floor_thickness / 2.0),
        "mat_floor", group=0, contype=1, conaffinity=1,
        extra=' condim="3"'
    ))

    # Court lines. All lines are visual-only to avoid shuttle bounces on paint/tape.
    for rect in c.visual_line_rectangles():
        xml.append(geom_box(
            str(rect["name"]),
            (float(rect["x"]), float(rect["y"]), line_z),
            (float(rect["sx"]), float(rect["sy"]), line_t / 2.0),
            "mat_line",
            group=1,
            contype=0,
            conaffinity=0,
        ))

    # Official reference sites.
    sites = [
        ("net_midpoint_site", 0, 0, 0.0),
        ("post_official_neg_y_site", 0, -c.half_width_doubles, 0.0),
        ("post_official_pos_y_site", 0, c.half_width_doubles, 0.0),
        ("court_back_neg_x_site", -c.half_length, 0, 0.0),
        ("court_back_pos_x_site", c.half_length, 0, 0.0),
        ("singles_side_neg_y_site", 0, -c.half_width_singles, 0.0),
        ("singles_side_pos_y_site", 0, c.half_width_singles, 0.0),
    ]
    for name, x, y, z in sites:
        xml.append(f'      <site name="{name}" pos="{fmt(x)} {fmt(y)} {fmt(z)}" size="0.025" rgba="1 0.3 0.2 0.8"/>\n')

    # Posts outside-tangent to avoid support intrusion into legal court.
    post_y = c.post_center_abs_y
    for sign, suffix in [(-1, "neg_y"), (1, "pos_y")]:
        y = sign * post_y
        xml.append(geom_capsule(
            f"net_post_{suffix}",
            (0, y, 0.0),
            (0, y, c.net_top_height_posts),
            c.post_radius,
            "mat_post",
            group=2,
            contype=2 if enable_net_collision else 0,
            conaffinity=1 if enable_net_collision else 0,
        ))

    # Net top tape and top cord segments.
    nseg = 40
    y_min = -c.half_width_doubles
    y_max = c.half_width_doubles
    for i in range(nseg):
        y0 = y_min + (y_max - y_min) * i / nseg
        y1 = y_min + (y_max - y_min) * (i + 1) / nseg
        ym = 0.5 * (y0 + y1)
        z0 = top_z(c, y0)
        z1 = top_z(c, y1)
        zm = top_z(c, ym)
        seg_half_y = abs(y1 - y0) / 2.0
        xml.append(geom_box(
            f"net_top_tape_{i:02d}",
            (0, ym, zm - c.net_depth * 0.0 - 0.075 / 2.0),
            (0.006, seg_half_y, 0.075 / 2.0),
            "mat_net_tape",
            group=2,
            contype=0,
            conaffinity=0,
        ))
        xml.append(geom_capsule(
            f"net_top_cord_{i:02d}",
            (0, y0, z0),
            (0, y1, z1),
            0.005,
            "mat_net_tape",
            group=2,
            contype=net_contype,
            conaffinity=net_conaffinity,
        ))

    # Visual net cords, simplified LOD. Official mesh is 15-20 mm; this is a render/runtime proxy.
    y_values = []
    y = y_min
    while y <= y_max + 1e-9:
        y_values.append(round(y, 6))
        y += net_pitch
    if abs(y_values[-1] - y_max) > 1e-6:
        y_values.append(y_max)

    # Vertical cords.
    for j, y in enumerate(y_values):
        zt = top_z(c, y) - 0.075  # start below tape
        zb = c.net_bottom_height(y)
        xml.append(geom_capsule(
            f"net_vertical_cord_{j:03d}",
            (0, y, zb),
            (0, y, zt),
            0.0012,
            "mat_net_cord",
            group=2,
            contype=0,
            conaffinity=0,
        ))

    # Horizontal cords at roughly 1/8 net depth.
    h_levels = 8
    for k in range(1, h_levels + 1):
        frac = k / (h_levels + 1)
        for i in range(nseg):
            y0 = y_min + (y_max - y_min) * i / nseg
            y1 = y_min + (y_max - y_min) * (i + 1) / nseg
            z0 = c.net_bottom_height(y0) * (1 - frac) + (top_z(c, y0) - 0.075) * frac
            z1 = c.net_bottom_height(y1) * (1 - frac) + (top_z(c, y1) - 0.075) * frac
            xml.append(geom_capsule(
                f"net_horizontal_cord_{k:02d}_{i:02d}",
                (0, y0, z0),
                (0, y1, z1),
                0.0012,
                "mat_net_cord",
                group=2,
                contype=0,
                conaffinity=0,
            ))

    if enable_net_collision:
        # Thin wall proxy split into segments to follow sagged top.
        for i in range(nseg):
            y0 = y_min + (y_max - y_min) * i / nseg
            y1 = y_min + (y_max - y_min) * (i + 1) / nseg
            ym = 0.5 * (y0 + y1)
            z_top_mid = top_z(c, ym) - 0.075  # top tape/cord is separate
            z_bottom_mid = c.net_bottom_height(ym)
            z_mid = 0.5 * (z_top_mid + z_bottom_mid)
            z_half = 0.5 * (z_top_mid - z_bottom_mid)
            xml.append(geom_box(
                f"net_collision_proxy_{i:02d}",
                (0, ym, z_mid),
                (net_thick_x / 2.0, abs(y1 - y0) / 2.0, z_half),
                "mat_net_proxy",
                group=3,
                contype=net_contype,
                conaffinity=net_conaffinity,
                extra=' condim="3" priority="1"'
            ))

    xml.append('    </body>\n')
    xml.append('  </worldbody>\n')
    xml.append('</mujoco>\n')
    return "".join(xml)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    params_path = root / "params" / "court_bwf_nominal.json"
    c = CourtParams.from_json(params_path)
    out_dir = root / "assets"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "badminton_court_bwf_visual.xml").write_text(
        generate_mjcf(c, enable_net_collision=False), encoding="utf-8"
    )
    (out_dir / "badminton_court_bwf_collision_net.xml").write_text(
        generate_mjcf(c, enable_net_collision=True), encoding="utf-8"
    )
    print("Wrote court MJCF assets to", out_dir)


if __name__ == "__main__":
    main()
