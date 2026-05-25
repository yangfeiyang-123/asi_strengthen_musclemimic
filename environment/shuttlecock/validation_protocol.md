# Shuttlecock MuJoCo Validation Protocol

Run these tests after integrating `assets/shuttlecock_mujoco.xml`,
`src/shuttlecock_aero.py`, and the racket string-bed proxy.

## 1. Geometry and Contact Acceptance

- Feather count: 16.
- Feather length from base-top plane to tip: `0.065 m`; allowed rules range is `0.062-0.070 m`.
- Tip circle diameter: `0.065 m`; allowed rules range is `0.058-0.068 m`.
- Base/cork diameter: `0.027 m`; allowed rules range is `0.025-0.028 m`.
- Mass: `0.00519 kg`; allowed rules range is `0.00474-0.00550 kg`.
- `cork_collision` is the primary collision geom.
- `cork_contact_site` exists and is colocated with the cork collision sphere center.
- Feather and thread geoms are visual-only in the first RL version.

## 2. Stage 1: Free Flight

Initialize the shuttle at `z=18 m`, low or zero velocity, random orientation,
and enable `apply_shuttlecock_aero`.

Acceptance:

- Measured vertical speed approaches `6.5-6.9 m/s` before ground contact.
- Default `terminal_velocity_m_s=6.86` produces approximately `6.86 m/s` terminal speed.
- A `30 m/s`, `30 degree` launch lands around the current `9-10 m` horizontal scale.
- A launch with the nose initially `60-120 degrees` away from velocity aligns nose-first within about `0.05-0.20 s` for speeds above `15 m/s`.
- Sideways or reversed flight decays faster than nose-forward flight.
- Aerodynamic force opposes relative velocity.
- Pressure-center torque tends to align local `+Z` with relative velocity.

## 3. Stage 2: Ordinary Racket Impact

Use the racket string-bed proxy with:

```python
apply_stringbed_force(
    model,
    data,
    racket_body_name="racket",
    shuttle_body_name="shuttle",
    shuttle_contact_site_name="cork_contact_site",
)
```

Acceptance:

- Medium racket speeds produce outgoing velocity consistent with racket surface velocity and string-bed normal.
- Sweet-spot impacts are stable and repeatable.
- Edge impacts are less stable than center impacts but remain numerically bounded.
- No high-speed tunneling at `timestep <= 0.0005 s`.
- The impact path uses `cork_contact_site`, not the shuttle COM fallback.

## 4. Stage 3: High-Intensity Racket Impact

Use event rebound when active contact has high closing normal speed.

Acceptance:

- Fast smash or drive-like contacts do not miss the shuttle.
- Event rebound triggers only for active contact with closing normal speed above `min_speed_for_event_m_s`.
- Rebound velocity is bounded by `max_rebound_speed_m_s`.
- Aerodynamic torque restores nose-forward flight after impact.
- Force, torque, and event clipping diagnostics are logged during validation.

## 5. Parameter Randomization Smoke Test

Sample the configured randomization ranges:

- `mass_kg`
- `terminal_velocity_m_s`
- `center_of_pressure_offset_m`
- `angle_drag_gain`
- `angular_damping_nms_per_rad`
- `event_restitution_normal`
- `event_tangential_velocity_scale`
- `stringbed_center_stiffness_n_per_m`
- `wind_m_s`

Acceptance:

- The nominal model passes before randomization is widened.
- Randomized ordinary impacts do not produce NaN, infinite velocity, or unbounded force.
- Training should start with narrow randomization and widen only after nominal validation passes.
