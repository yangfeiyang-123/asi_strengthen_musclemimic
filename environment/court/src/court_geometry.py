"""BWF badminton court geometry helpers for MuJoCo scenes.

Coordinate convention
---------------------
x: court length. The net is at x=0. The two half-courts are x>0 and x<0.
y: court width. The centre service line is y=0.
z: up. Court surface is z=0.

This module uses "edge-correct" semantics:
official court dimensions are legal outer edges, and 40 mm lines are part of
the areas they define. Visual line geoms should be offset inward by half the
line width when the line is an outer boundary.

The small line-width ambiguities in real court construction are deliberately
centralized here. Use this module as the single source of truth for landing
classification and line placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

CourtMode = Literal["singles", "doubles"]
CourtHalf = Literal["+x", "-x"]
LateralHalf = Literal["+y", "-y"]


@dataclass(frozen=True)
class CourtParams:
    full_court_length: float = 13.40
    doubles_width: float = 6.10
    singles_width: float = 5.18
    line_width: float = 0.040
    short_service_line_from_net_near_edge: float = 1.98
    doubles_long_service_line_from_back_outer_edge: float = 0.76
    net_top_height_center: float = 1.524
    net_top_height_posts: float = 1.550
    net_depth: float = 0.760
    post_radius: float = 0.035

    @classmethod
    def from_json(cls, path: str | Path) -> "CourtParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        official = data["official_dimensions_m"]
        net = data["net_m"]
        return cls(
            full_court_length=official["full_court_length"],
            doubles_width=official["doubles_width"],
            singles_width=official["singles_width"],
            line_width=official["line_width"],
            short_service_line_from_net_near_edge=official[
                "short_service_line_from_net_near_edge"
            ],
            doubles_long_service_line_from_back_outer_edge=official[
                "doubles_long_service_line_from_back_outer_edge"
            ],
            net_top_height_center=net["top_height_center"],
            net_top_height_posts=net["top_height_over_doubles_sidelines"],
            net_depth=net["depth"],
            post_radius=net["simulation_post_radius"],
        )

    @property
    def half_line(self) -> float:
        return self.line_width / 2.0

    @property
    def half_length(self) -> float:
        return self.full_court_length / 2.0

    @property
    def half_width_doubles(self) -> float:
        return self.doubles_width / 2.0

    @property
    def half_width_singles(self) -> float:
        return self.singles_width / 2.0

    @property
    def short_service_near_edge_abs_x(self) -> float:
        return self.short_service_line_from_net_near_edge

    @property
    def short_service_center_abs_x(self) -> float:
        return self.short_service_near_edge_abs_x + self.half_line

    @property
    def doubles_long_service_outer_edge_abs_x(self) -> float:
        return self.half_length - self.doubles_long_service_line_from_back_outer_edge

    @property
    def doubles_long_service_center_abs_x(self) -> float:
        return self.doubles_long_service_outer_edge_abs_x - self.half_line

    @property
    def back_boundary_center_abs_x(self) -> float:
        return self.half_length - self.half_line

    @property
    def doubles_sideline_center_abs_y(self) -> float:
        return self.half_width_doubles - self.half_line

    @property
    def singles_sideline_center_abs_y(self) -> float:
        return self.half_width_singles - self.half_line

    @property
    def post_center_abs_y(self) -> float:
        """Simulation post centre with inner tangent on y=±doubles half-width."""
        return self.half_width_doubles + self.post_radius

    def net_top_height(self, y: float) -> float:
        """Parabolic net sag profile from centre height to sideline/post height."""
        ratio = min(abs(y) / self.half_width_doubles, 1.0)
        return self.net_top_height_center + (
            self.net_top_height_posts - self.net_top_height_center
        ) * ratio * ratio

    def net_bottom_height(self, y: float) -> float:
        return self.net_top_height(y) - self.net_depth

    def rally_bounds(self, mode: CourtMode) -> Tuple[float, float, float, float]:
        """Return legal rally landing bounds as (xmin, xmax, ymin, ymax).

        Lines are included. For singles, the active side boundaries are the
        outer edges of the singles side lines. For doubles, they are the outer
        edges of the doubles side lines.
        """
        half_y = self.half_width_singles if mode == "singles" else self.half_width_doubles
        return (-self.half_length, self.half_length, -half_y, half_y)

    def inside_rally(self, x: float, y: float, mode: CourtMode = "doubles",
                     eps: float = 1e-9) -> bool:
        xmin, xmax, ymin, ymax = self.rally_bounds(mode)
        return (xmin - eps <= x <= xmax + eps) and (ymin - eps <= y <= ymax + eps)

    def service_bounds(
        self,
        mode: CourtMode,
        court_half: CourtHalf,
        lateral_half: LateralHalf,
    ) -> Tuple[float, float, float, float]:
        """Return legal service landing bounds as (xmin, xmax, ymin, ymax).

        Parameters
        ----------
        mode:
            "singles" or "doubles".
        court_half:
            Target/receiver half-court: "+x" or "-x".
        lateral_half:
            Target service court lateral side: "+y" or "-y". The centre line
            is included in both halves.
        """
        half_y = self.half_width_singles if mode == "singles" else self.half_width_doubles

        # The short service line is included; near edge starts the service court.
        x_near = self.short_service_near_edge_abs_x
        if mode == "doubles":
            x_far = self.doubles_long_service_outer_edge_abs_x
        else:
            x_far = self.half_length

        if court_half == "+x":
            xmin, xmax = x_near, x_far
        else:
            xmin, xmax = -x_far, -x_near

        # Centre line is included in both service courts.
        if lateral_half == "+y":
            ymin, ymax = -self.half_line, half_y
        else:
            ymin, ymax = -half_y, self.half_line

        return xmin, xmax, ymin, ymax

    def inside_service(
        self,
        x: float,
        y: float,
        mode: CourtMode,
        court_half: CourtHalf,
        lateral_half: LateralHalf,
        eps: float = 1e-9,
    ) -> bool:
        xmin, xmax, ymin, ymax = self.service_bounds(mode, court_half, lateral_half)
        return (xmin - eps <= x <= xmax + eps) and (ymin - eps <= y <= ymax + eps)

    def visual_line_rectangles(self) -> List[Dict[str, float | str]]:
        """Return line rectangles suitable for MJCF box geoms.

        Each rectangle is represented by centre position (x, y), half extents
        (sx, sy), and semantic name. All z sizes are handled by the XML generator.
        """
        h = self.half_line
        L = self.half_length
        Wd = self.half_width_doubles
        Ws = self.half_width_singles
        rects: List[Dict[str, float | str]] = []

        # Outer doubles side lines.
        for s in (-1, 1):
            rects.append({
                "name": f"doubles_sideline_{'pos' if s > 0 else 'neg'}_y",
                "x": 0.0,
                "y": s * (Wd - h),
                "sx": L,
                "sy": h,
                "role": "rally_boundary_doubles",
            })

        # Back boundary lines; also long service line for singles.
        for s in (-1, 1):
            rects.append({
                "name": f"back_boundary_{'pos' if s > 0 else 'neg'}_x",
                "x": s * (L - h),
                "y": 0.0,
                "sx": h,
                "sy": Wd,
                "role": "rally_boundary_all_and_singles_long_service",
            })

        # Singles side lines.
        for s in (-1, 1):
            rects.append({
                "name": f"singles_sideline_{'pos' if s > 0 else 'neg'}_y",
                "x": 0.0,
                "y": s * (Ws - h),
                "sx": L,
                "sy": h,
                "role": "rally_boundary_singles",
            })

        # Short service lines on each half-court.
        x_short_center = self.short_service_center_abs_x
        for s in (-1, 1):
            rects.append({
                "name": f"short_service_line_{'pos' if s > 0 else 'neg'}_x",
                "x": s * x_short_center,
                "y": 0.0,
                "sx": h,
                "sy": Wd,
                "role": "service_near_boundary",
            })

        # Doubles long service lines.
        x_dlong_center = self.doubles_long_service_center_abs_x
        for s in (-1, 1):
            rects.append({
                "name": f"doubles_long_service_line_{'pos' if s > 0 else 'neg'}_x",
                "x": s * x_dlong_center,
                "y": 0.0,
                "sx": h,
                "sy": Wd,
                "role": "service_far_boundary_doubles",
            })

        # Centre service lines, one per side, from short service line to back boundary.
        x1 = self.short_service_near_edge_abs_x
        x2 = self.half_length
        sx = (x2 - x1) / 2.0
        xc = (x2 + x1) / 2.0
        for s in (-1, 1):
            rects.append({
                "name": f"centre_service_line_{'pos' if s > 0 else 'neg'}_x_half",
                "x": s * xc,
                "y": 0.0,
                "sx": sx,
                "sy": h,
                "role": "service_lateral_boundary",
            })

        return rects


def load_default() -> CourtParams:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "params" / "court_bwf_nominal.json",
        Path("params/court_bwf_nominal.json"),
        Path("../params/court_bwf_nominal.json"),
    ]
    for path in candidates:
        if path.exists():
            return CourtParams.from_json(path)
    return CourtParams()


if __name__ == "__main__":
    c = load_default()
    print("BWF court:")
    print(f"  full court: {c.full_court_length:.2f} m × {c.doubles_width:.2f} m")
    print(f"  singles width: {c.singles_width:.2f} m")
    print(f"  line width: {c.line_width:.3f} m")
    print(f"  net top: {c.net_top_height(0):.3f} m centre, "
          f"{c.net_top_height(c.half_width_doubles):.3f} m sidelines")
