from __future__ import annotations

import json
import math
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_racket_mjcf import flex_mjcf  # noqa: E402
import validate_racket_params  # noqa: E402


def parse_fromto(text: str) -> list[float]:
    return [float(x) for x in text.split()]


class FlexProxyGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = json.loads((ROOT / "params" / "racket_nominal.json").read_text(encoding="utf-8"))
        self.root = ET.fromstring(flex_mjcf(self.params))

    def test_upper_shaft_spans_from_hinge_to_nominal_shaft_end(self) -> None:
        geom = self.root.find(".//geom[@name='upper_shaft']")
        self.assertIsNotNone(geom)

        x0, y0, z0, x1, y1, z1 = parse_fromto(geom.attrib["fromto"])
        expected_length = (
            self.params["geometry_m"]["shaft_end_y"]
            - self.params["flex_proxy"]["hinge_y_m"]
        )

        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(z0, 0.0)
        self.assertAlmostEqual(x1, 0.0)
        self.assertAlmostEqual(y1, expected_length, places=6)
        self.assertAlmostEqual(z1, 0.0)

    def test_flex_joint_stiffness_matches_declared_bending_frequency(self) -> None:
        target_hz = self.params["flex_proxy"]["target_first_bending_frequency_hz"]
        body = self.root.find(".//body[@name='racket_head']")
        self.assertIsNotNone(body)
        inertial = body.find("inertial")
        self.assertIsNotNone(inertial)

        mass = float(inertial.attrib["mass"])
        _, y_com, _ = [float(x) for x in inertial.attrib["pos"].split()]
        ixx, _, izz = [float(x) for x in inertial.attrib["diaginertia"].split()]
        inertia_by_joint = {
            "shaft_flex_x": ixx + mass * y_com**2,
            "shaft_flex_z": izz + mass * y_com**2,
        }

        for joint_name, inertia in inertia_by_joint.items():
            joint = self.root.find(f".//joint[@name='{joint_name}']")
            self.assertIsNotNone(joint)
            stiffness = float(joint.attrib["stiffness"])
            hz = math.sqrt(stiffness / inertia) / (2.0 * math.pi)
            self.assertAlmostEqual(hz, target_hz, delta=1.0)


class ParameterValidationTest(unittest.TestCase):
    def test_static_center_force_is_validated(self) -> None:
        with patch("builtins.print") as print_mock:
            validate_racket_params.main()

        lines = [" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list]
        self.assertTrue(
            any("center_static_5mm_force_ok" in line and "PASS" in line for line in lines),
            "validation output should include an explicit PASS/FAIL for the 5 mm center force target",
        )


if __name__ == "__main__":
    unittest.main()
