"""Validate the BWF badminton court design package.

This script checks:
- official/derived dimensions in params JSON,
- key court line centres and extents,
- rally and service landing classifiers,
- generated XML parseability and presence of required geoms/sites.

Run from package root:

    python src/validate_court_params.py
"""

from __future__ import annotations

import json
import math
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

try:
    from court_geometry import CourtParams
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from court_geometry import CourtParams


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def check(cond: bool, name: str, failures: list[str]) -> None:
    if cond:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}")
        failures.append(name)


def check_optional_mujoco_compile(xml_paths: list[Path], failures: list[str]) -> None:
    """Compile MJCF assets with MuJoCo when the optional dependency is installed."""
    if importlib.util.find_spec("mujoco") is None:
        print("SKIP  MuJoCo compile validation (mujoco package not installed)")
        return

    import mujoco

    for xml_path in xml_paths:
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        except Exception as exc:
            print(f"FAIL  {xml_path.name} compiles with MuJoCo: {exc}")
            failures.append(f"{xml_path.name} compiles with MuJoCo")
        else:
            print(
                f"PASS  {xml_path.name} compiles with MuJoCo "
                f"(ngeom={model.ngeom}, nbody={model.nbody})"
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    params_path = root / "params" / "court_bwf_nominal.json"
    c = CourtParams.from_json(params_path)

    failures: list[str] = []

    check(approx(c.full_court_length, 13.40), "full court length 13.40 m", failures)
    check(approx(c.doubles_width, 6.10), "doubles width 6.10 m", failures)
    check(approx(c.singles_width, 5.18), "singles width 5.18 m", failures)
    check(approx(c.line_width, 0.040), "line width 40 mm", failures)
    check(approx(c.half_length, 6.70), "half length 6.70 m", failures)
    check(approx(c.half_width_doubles, 3.05), "doubles half-width 3.05 m", failures)
    check(approx(c.half_width_singles, 2.59), "singles half-width 2.59 m", failures)
    check(approx(c.short_service_near_edge_abs_x, 1.98), "short service near edge at |x|=1.98 m", failures)
    check(approx(c.doubles_long_service_outer_edge_abs_x, 5.94), "doubles long service outer edge at |x|=5.94 m", failures)
    check(approx(c.net_top_height(0), 1.524), "net centre height 1.524 m", failures)
    check(approx(c.net_top_height(c.half_width_doubles), 1.550), "net sideline height 1.550 m", failures)
    check(approx(c.net_bottom_height(0), 0.764), "net centre bottom height 0.764 m", failures)

    # Rally classifier.
    check(c.inside_rally(6.70, 3.05, "doubles"), "doubles rally includes outer boundary line", failures)
    check(not c.inside_rally(6.701, 0.0, "doubles"), "doubles rally excludes beyond back boundary", failures)
    check(c.inside_rally(0.0, 2.59, "singles"), "singles rally includes singles side line", failures)
    check(not c.inside_rally(0.0, 2.591, "singles"), "singles rally excludes doubles tramline", failures)

    # Service classifier.
    check(c.inside_service(1.98, 0.01, "doubles", "+x", "+y"), "service includes short service line", failures)
    check(c.inside_service(5.94, 2.0, "doubles", "+x", "+y"), "doubles service includes long service line", failures)
    check(not c.inside_service(5.941, 2.0, "doubles", "+x", "+y"), "doubles service excludes behind long service line", failures)
    check(c.inside_service(6.70, -2.0, "singles", "+x", "-y"), "singles service includes back boundary", failures)
    check(not c.inside_service(6.701, -2.0, "singles", "+x", "-y"), "singles service excludes beyond back boundary", failures)

    # XML parse and key elements.
    xml_paths = [
        root / "assets" / "badminton_court_bwf_visual.xml",
        root / "assets" / "badminton_court_bwf_collision_net.xml",
    ]

    for xml_path in xml_paths:
        asset_name = xml_path.name
        check(xml_path.exists(), f"{asset_name} exists", failures)
        tree = ET.parse(xml_path)
        root_xml = tree.getroot()
        check(root_xml.tag == "mujoco", f"{asset_name} parses as MJCF root", failures)
        names = {elem.attrib.get("name") for elem in root_xml.iter() if "name" in elem.attrib}
        for required in [
            "floor_collision",
            "back_boundary_pos_x",
            "back_boundary_neg_x",
            "short_service_line_pos_x",
            "short_service_line_neg_x",
            "doubles_long_service_line_pos_x",
            "doubles_long_service_line_neg_x",
            "net_post_pos_y",
            "net_post_neg_y",
            "net_midpoint_site",
        ]:
            check(required in names, f"{asset_name} contains {required}", failures)

        if "collision_net" in asset_name:
            check(any((elem.attrib.get("name", "").startswith("net_collision_proxy")
                       and elem.attrib.get("contype") == "2")
                      for elem in root_xml.iter("geom")),
                  f"{asset_name} has enabled net collision proxies", failures)
        else:
            check(not any((elem.attrib.get("name", "").startswith("net_collision_proxy"))
                          for elem in root_xml.iter("geom")),
                  f"{asset_name} has no net collision proxies", failures)

    check_optional_mujoco_compile(xml_paths, failures)

    if failures:
        print("\nValidation failed:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)

    print("\nAll court design checks passed.")


if __name__ == "__main__":
    main()
