"""Visualize and summarize the nominal badminton racket design."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

cache_root = Path(tempfile.gettempdir()) / "racket_design_matplotlib_cache"
cache_root.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
(cache_root / "mpl").mkdir(exist_ok=True)
(cache_root / "xdg").mkdir(exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle


def load_params() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "params" / "racket_nominal.json").read_text(encoding="utf-8"))


def ellipse_points(cx: float, cy: float, a: float, b: float, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0.0, 2.0 * math.pi, n)
    return cx + a * np.cos(th), cy + b * np.sin(th)


def draw_dimension(ax, start, end, text: str, offset=(0.0, 0.0), color="#374151") -> None:
    sx, sy = start
    ex, ey = end
    ox, oy = offset
    ax.annotate(
        "",
        xy=(ex + ox, ey + oy),
        xytext=(sx + ox, sy + oy),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=color, shrinkA=0, shrinkB=0),
    )
    ax.text((sx + ex) * 0.5 + ox, (sy + ey) * 0.5 + oy, text, ha="center", va="center", color=color, fontsize=8)


def draw_geometry(ax, params: dict) -> None:
    g = params["geometry_m"]
    s = params["stringbed"]
    m = params["mass_properties"]

    ax.set_title("Plan geometry, x-y plane", loc="left", fontsize=11, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("X lateral (m)")
    ax.set_ylabel("Y butt to tip (m)")

    frame = Ellipse(
        (0.0, g["head_center_y"]),
        width=2.0 * g["head_outer_half_width"],
        height=2.0 * g["head_outer_half_length"],
        fill=False,
        lw=4,
        ec="#111827",
        zorder=3,
    )
    stringbed = Ellipse(
        (0.0, g["stringbed_center_y"]),
        width=2.0 * g["stringbed_half_width"],
        height=2.0 * g["stringbed_half_length"],
        facecolor="#dbeafe",
        edgecolor="#2563eb",
        lw=1.5,
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(stringbed)
    ax.add_patch(frame)

    ax.plot([0, 0], [g["shaft_start_y"], g["shaft_end_y"]], color="#111827", lw=5, solid_capstyle="round", zorder=4)
    ax.plot([0, -g["throat_half_width"]], [g["throat_start_y"], g["throat_end_y"]], color="#111827", lw=4, zorder=4)
    ax.plot([0, g["throat_half_width"]], [g["throat_start_y"], g["throat_end_y"]], color="#111827", lw=4, zorder=4)
    ax.add_patch(Rectangle((-g["handle_radius"], 0.01), 2 * g["handle_radius"], g["handle_length"] - 0.01,
                           facecolor="#4b5563", edgecolor="#111827", lw=1.0, zorder=4))
    ax.add_patch(Circle((0.0, 0.0), g["butt_cap_radius"], facecolor="#6b7280", edgecolor="#111827", lw=1.0, zorder=4))

    a = g["stringbed_half_width"]
    b = g["stringbed_half_length"]
    cy = g["stringbed_center_y"]
    for i in range(int(s["main_string_count"])):
        x = -0.94 * a + (1.88 * a) * i / (int(s["main_string_count"]) - 1)
        yh = b * math.sqrt(max(0.0, 1.0 - (x / a) ** 2))
        ax.plot([x, x], [cy - yh, cy + yh], color="#f9fafb", lw=0.45, alpha=0.9, zorder=2)
    for j in range(int(s["cross_string_count"])):
        dy = -0.94 * b + (1.88 * b) * j / (int(s["cross_string_count"]) - 1)
        xh = a * math.sqrt(max(0.0, 1.0 - (dy / b) ** 2))
        ax.plot([-xh, xh], [cy + dy, cy + dy], color="#e5e7eb", lw=0.45, alpha=0.9, zorder=2)

    ax.scatter([0], [m["center_of_mass_from_butt_y_m"]], s=42, color="#dc2626", zorder=6)
    ax.text(0.012, m["center_of_mass_from_butt_y_m"], "COM 310 mm", color="#dc2626", va="center", fontsize=8)
    ax.scatter([0], [m["swingweight_axis_from_butt_y_m"]], s=36, color="#7c3aed", zorder=6)
    ax.text(0.012, m["swingweight_axis_from_butt_y_m"], "9 cm swing axis", color="#7c3aed", va="center", fontsize=8)
    ax.scatter([0], [g["stringbed_center_y"]], s=30, color="#2563eb", zorder=6)

    draw_dimension(ax, (0.125, 0.0), (0.125, g["overall_length"]), "675 mm length")
    draw_dimension(ax, (-g["head_outer_half_width"], g["head_center_y"] + 0.17),
                   (g["head_outer_half_width"], g["head_center_y"] + 0.17), "210 mm width")
    draw_dimension(ax, (-g["stringbed_half_width"], g["stringbed_center_y"] - 0.155),
                   (g["stringbed_half_width"], g["stringbed_center_y"] - 0.155), "188 mm stringbed")

    ax.set_xlim(-0.155, 0.17)
    ax.set_ylim(-0.035, 0.705)
    ax.grid(True, color="#e5e7eb", lw=0.6)


def draw_compliance(ax, params: dict) -> None:
    g = params["geometry_m"]
    reg = params["regulatory_limits_bwf"]
    rows = [
        ("Overall length", g["overall_length"], reg["max_overall_length_m"], "m"),
        ("Overall width", g["overall_width"], reg["max_overall_width_m"], "m"),
        ("Stringbed length", g["stringbed_length"], reg["max_stringed_area_length_m"], "m"),
        ("Stringbed width", g["stringbed_width"], reg["max_stringed_area_width_m"], "m"),
    ]

    ax.set_title("Rule-size margins", loc="left", fontsize=11, weight="bold")
    y = np.arange(len(rows))
    used = np.array([r[1] / r[2] for r in rows])
    colors = ["#059669" if u <= 1.0 else "#dc2626" for u in used]
    ax.barh(y, [1.0] * len(rows), color="#e5e7eb", height=0.58)
    ax.barh(y, used, color=colors, height=0.58)
    ax.axvline(1.0, color="#111827", lw=1.0)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.0, 1.08)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of limit")
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    for idx, (label, value, limit, unit) in enumerate(rows):
        margin = limit - value
        ax.text(0.02, idx, f"{value*1000:.0f}/{limit*1000:.0f} mm", va="center", ha="left", fontsize=8, color="#111827")
        ax.text(1.02, idx, f"+{margin*1000:.0f} mm", va="center", ha="left", fontsize=8, color="#059669")


def draw_mass_properties(ax, params: dict) -> None:
    g = params["geometry_m"]
    m = params["mass_properties"]
    mass = m["mass_kg"]
    com_y = m["center_of_mass_from_butt_y_m"]
    axis_y = m["swingweight_axis_from_butt_y_m"]
    ixx = m["principal_inertia_about_com_kg_m2"]["Ixx"]
    swing = ixx + mass * (com_y - axis_y) ** 2

    ax.set_title("Mass properties", loc="left", fontsize=11, weight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, g["overall_length"] + 0.02)
    ax.axis("off")
    ax.plot([0.42, 0.42], [0.0, g["overall_length"]], color="#9ca3af", lw=2.0)

    marks = [
        ("butt", 0.0, "#111827"),
        ("grip site", axis_y, "#7c3aed"),
        ("COM", com_y, "#dc2626"),
        ("stringbed center", g["stringbed_center_y"], "#2563eb"),
        ("tip", g["overall_length"], "#111827"),
    ]
    for label, yy, color in marks:
        ax.plot([0.35, 0.49], [yy, yy], color=color, lw=2)
        ax.text(0.53, yy, f"{label}: {yy*1000:.0f} mm", va="center", fontsize=8, color=color)

    text = (
        f"mass = {mass*1000:.1f} g\n"
        f"Ixx_COM = {ixx:.6f} kg m^2\n"
        f"SW@90mm = {swing*10000:.2f} kg cm^2\n"
        f"parallel axis: Ixx + m*(COM-axis)^2"
    )
    ax.text(0.05, 0.58, text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9fafb", edgecolor="#d1d5db"), fontsize=9)


def draw_stiffness(ax, fig, params: dict) -> None:
    g = params["geometry_m"]
    s = params["stringbed"]
    a = g["stringbed_half_width"]
    b = g["stringbed_half_length"]
    k0 = s["static_center_normal_stiffness_n_per_m"]
    edge_gain = s["normal_stiffness_edge_gain"]
    kmax = k0 * s["normal_stiffness_max_multiplier"]

    xs = np.linspace(-a, a, 280)
    ys = np.linspace(-b, b, 280)
    xx, yy = np.meshgrid(xs, ys)
    rho2 = (xx / a) ** 2 + (yy / b) ** 2
    stiffness = k0 * np.minimum(1.0 + edge_gain * np.clip(rho2, 0.0, 1.0), kmax / k0)
    stiffness = np.ma.masked_where(rho2 > 1.0, stiffness)

    ax.set_title("Stringbed normal stiffness proxy", loc="left", fontsize=11, weight="bold")
    im = ax.imshow(stiffness, extent=(-a, a, -b, b), origin="lower", cmap="viridis", aspect="equal")
    ax.contour(xx, yy, rho2, levels=[0.25, 0.50, 0.75, 1.0], colors="white", linewidths=0.7, alpha=0.65)
    ax.set_xlabel("local X from center (m)")
    ax.set_ylabel("local Y from center (m)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("N/m")

    center_force = k0 * 0.005
    edge_force = k0 * (1.0 + edge_gain) * 0.005
    ax.text(
        0.03,
        0.04,
        f"5 mm static force:\ncenter {center_force:.1f} N\nnear edge {edge_force:.1f} N",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
        fontsize=8,
    )


def main() -> None:
    params = load_params()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])
    draw_geometry(fig.add_subplot(gs[:, 0]), params)
    draw_compliance(fig.add_subplot(gs[0, 1]), params)
    sub = gs[1, 1].subgridspec(1, 2, width_ratios=[0.82, 1.18])
    draw_mass_properties(fig.add_subplot(sub[0, 0]), params)
    draw_stiffness(fig.add_subplot(sub[0, 1]), fig, params)

    fig.suptitle("Badminton racket design realism check", fontsize=14, weight="bold")
    png = out_dir / "racket_design_validation.png"
    svg = out_dir / "racket_design_validation.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
