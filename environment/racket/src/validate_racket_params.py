"""Sanity checks for the nominal badminton racket design parameters."""
from __future__ import annotations

import json
from pathlib import Path


def kg_m2_to_kg_cm2(x: float) -> float:
    return x * 10000.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    params = json.loads((root / "params" / "racket_nominal.json").read_text(encoding="utf-8"))
    g = params["geometry_m"]
    reg = params["regulatory_limits_bwf"]
    m = params["mass_properties"]
    s = params["stringbed"]
    vt = params["validation_targets"]

    mass = m["mass_kg"]
    com_y = m["center_of_mass_from_butt_y_m"]
    axis_y = m["swingweight_axis_from_butt_y_m"]
    Ixx = m["principal_inertia_about_com_kg_m2"]["Ixx"]
    swingweight = Ixx + mass * (com_y - axis_y) ** 2
    swingweight_kg_cm2 = kg_m2_to_kg_cm2(swingweight)
    center_static_5mm_force = s["static_center_normal_stiffness_n_per_m"] * 0.005

    checks = {
        "overall_length_ok": g["overall_length"] <= reg["max_overall_length_m"],
        "overall_width_ok": g["overall_width"] <= reg["max_overall_width_m"],
        "stringbed_length_ok": g["stringbed_length"] <= reg["max_stringed_area_length_m"],
        "stringbed_width_ok": g["stringbed_width"] <= reg["max_stringed_area_width_m"],
        "mass_target_ok": vt["mass_range_kg"][0] <= mass <= vt["mass_range_kg"][1],
        "swingweight_target_ok": vt["swingweight_kg_cm2"][0] <= swingweight_kg_cm2 <= vt["swingweight_kg_cm2"][1],
        "balance_target_ok": vt["balance_y_m"][0] <= com_y <= vt["balance_y_m"][1],
        "center_static_5mm_force_ok": vt["center_static_5mm_force_n"][0] <= center_static_5mm_force <= vt["center_static_5mm_force_n"][1],
    }

    print("Badminton racket parameter validation")
    print("---------------------------------------")
    print(f"overall length:        {g['overall_length']:.3f} m <= {reg['max_overall_length_m']:.3f} m")
    print(f"overall width:         {g['overall_width']:.3f} m <= {reg['max_overall_width_m']:.3f} m")
    print(f"stringbed length:      {g['stringbed_length']:.3f} m <= {reg['max_stringed_area_length_m']:.3f} m")
    print(f"stringbed width:       {g['stringbed_width']:.3f} m <= {reg['max_stringed_area_width_m']:.3f} m")
    print(f"mass:                  {mass*1000.0:.1f} g")
    print(f"balance from butt:     {com_y*1000.0:.1f} mm")
    print(f"swingweight @ 90 mm:   {swingweight_kg_cm2:.2f} kg cm^2")
    print(f"center 5 mm force:     {center_static_5mm_force:.2f} N")
    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  {name:<24} {'PASS' if ok else 'FAIL'}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
