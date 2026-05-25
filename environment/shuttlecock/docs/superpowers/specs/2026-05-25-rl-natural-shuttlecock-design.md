# RL Natural Shuttlecock Design

Date: 2026-05-25

## Context

The goal is a MuJoCo natural-feather badminton shuttlecock for muscle-driven RL.
It should look like a real shuttlecock and reproduce the flight and impact
behaviors that matter for learning badminton movement: tracking the shuttle,
swinging the racket, striking through the cork, and reacting to realistic
outgoing trajectories.

This is not a CFD, FEM, or brand-specific shuttle model. The first version uses
public regulation and aerodynamic references for defaults, exposes the important
parameters, and leaves room for later fitting from measured trajectories.

Existing project baseline:

- `assets/shuttlecock_mujoco.xml`
- `src/shuttlecock_aero.py`
- `params/shuttlecock_nominal.json`
- `validation_protocol.md`
- `/data3/yangfeiyang/WorkSpace/musclemimic/environment/racket/src/racket_stringbed.py`

## Design Position

Use a natural feather shuttlecock with:

- real-looking 16-feather geometry;
- MuJoCo rigid-body mass and inertia;
- visual-only feathers in the first version;
- cork/base as the primary collision and impact body;
- custom aerodynamic force for free flight;
- racket string-bed proxy for impact;
- event-style high-speed rebound as a numerical safety layer.

This gives a stable RL object while preserving the main real-world force paths:
gravity, aerodynamic drag, pressure-center restoring torque, string-bed impulse,
and cork-first impact.

## Geometry and Appearance

The MJCF should keep the current real-shuttle proportions:

- 16 feathers.
- Feather length: about `0.065 m`.
- Tip circle diameter: about `0.065 m`.
- Base/cork diameter: about `0.027 m`.
- Total length: about `0.090 m`.
- Mass nominal: `0.00519 kg`.

These values are inside the BWF shuttle constraints: 16 feathers, feather length
`62-70 mm`, and mass `4.74-5.50 g`.

The first RL version keeps all feather and thread geoms visual-only. The cork
collision sphere remains the main contact body. This deliberately avoids dense
high-speed contact between many small feather boxes and the racket or ground.

Future extension: a coarse feather-skirt collision proxy can be added for low
speed ground, net, or body interactions, but it should be a separate mode, not
the default training path.

## Coordinate Convention

The shuttle body keeps the existing convention:

- body origin: estimated center of mass;
- local `+Z`: cork/nose direction;
- local `-Z`: feather-skirt/tail direction;
- body `+Z` should align with the velocity direction in stable forward flight.

Add one explicit contact site if it is not already present:

```text
cork_contact_site
```

This site should represent the soft cork contact point used by the racket
string-bed model. The racket integration must use this site instead of falling
back to the shuttle COM, because COM impact loses the correct contact geometry
and torque behavior.

## Free-Flight Dynamics

MuJoCo provides gravity. `src/shuttlecock_aero.py` applies aerodynamic force.

The model remains:

```text
F_drag = -k_eff * |v_rel| * v_rel
k = m * g / vt^2
k_eff = k * (1 + angle_drag_gain * sin(alpha)^2)
```

where:

- `vt` is the terminal velocity;
- `alpha` is the angle between the nose axis and relative velocity;
- `angle_drag_gain` increases drag when the shuttle is sideways or reversed.

Drag is applied at a center of pressure behind the COM:

```text
CP = COM - center_of_pressure_offset * nose_axis
```

This produces the real shuttle behavior that the cork turns toward the velocity
direction after launch.

Nominal values:

```text
mass_kg                         0.00519
terminal_velocity_m_s           6.86
center_of_pressure_offset_m     0.035
angle_drag_gain                 0.20
angular_damping_nms_per_rad     2.0e-5
```

Required safeguards:

- cap aerodynamic force and torque;
- keep timestep at `0.0005 s` for high-speed shots;
- expose diagnostics for force magnitude, torque magnitude, speed, angle of
  attack, and whether clipping occurred.

## Racket Impact Dynamics

The racket model already defines a string-bed proxy:

- racket local `+Z`: string-bed normal;
- string-bed center around local `y = 0.532`;
- elliptical string-bed region;
- center normal stiffness around `9600 N/m`;
- spring-damper force plus optional event rebound.

The shuttle impact path should call the racket code with:

```python
apply_stringbed_force(
    model,
    data,
    racket_body_name="racket",
    shuttle_body_name="shuttle",
    shuttle_contact_site_name="cork_contact_site",
)
```

The normal force is not constant. It depends on penetration, relative velocity,
and impact location:

```text
F_normal = k(rho) * penetration - c * relative_normal_velocity
```

Edge hits use higher effective stiffness than sweet-spot hits. Tangential
damping and friction reduce sliding/tangential relative velocity.

For high-speed impacts, the simulation may miss a very short contact interval.
When the contact is active and closing speed exceeds the event threshold, use
event rebound:

```text
v_rel_after_n = -e_normal * v_rel_before_n
v_rel_after_t = tangential_velocity_scale * v_rel_before_t
v_shuttle_after = v_racket_surface + v_rel_after
```

Recommended first-version impact parameters:

```text
stringbed_center_stiffness_n_per_m  8000-12000 randomizable, 9600 nominal
normal_damping_n_s_per_m            3.0 nominal
tangential_damping_n_s_per_m        0.15 nominal
tangential_mu                       0.08 nominal
event_restitution_normal            0.45-0.60
event_tangential_velocity_scale     0.80-0.90
min_speed_for_event_m_s             5.0
```

This model is intended to match outgoing velocity direction, speed scale, and
shot stability, not individual string deformation.

## Simulation Step Order

The integration order should be explicit:

```python
data.qfrc_applied[:] = 0.0
apply_shuttlecock_aero(model, data, shuttle_cfg)
contact_info = apply_stringbed_force(
    model,
    data,
    racket_body_name="racket",
    shuttle_body_name="shuttle",
    shuttle_contact_site_name="cork_contact_site",
    geom=racket_geom,
    params=stringbed_params,
)
if should_use_event_rebound(contact_info, stringbed_params):
    apply_event_rebound_to_shuttle_state(...)
mujoco.mj_step(model, data)
```

The first implementation should provide one helper that directly updates the
shuttle freejoint linear velocity after computing the post-impact velocity. The
environment should call that helper immediately before `mujoco.mj_step`. Do not
also add a second environment-level rebound path, because two rebound handlers
could double-apply the impact.

## Domain Randomization

Randomize parameters that represent real shuttle and string-bed variation:

```text
mass_kg                         0.00474 - 0.00550
terminal_velocity_m_s           6.50 - 6.90
center_of_pressure_offset_m     0.025 - 0.045
angle_drag_gain                 0.00 - 0.50
angular_damping_nms_per_rad     5e-6 - 8e-5
event_restitution_normal        0.45 - 0.60
event_tangential_velocity_scale  0.80 - 0.90
stringbed_center_stiffness      8000 - 12000 N/m
wind                            optional small random vector
```

Randomization should be narrow at first. Widen it only after the nominal model
passes the validation stages.

## Validation Protocol

Validation is staged so ordinary rallies can become stable before extreme shots.

### Stage 1: Free Flight

Acceptance:

- 18 m free fall approaches `6.5-6.9 m/s` vertical terminal speed.
- 30 m/s at 30 degrees lands around the current expected `9-10 m` scale.
- Starting with the nose `60-120 degrees` away from velocity, the shuttle aligns
  nose-first within roughly `0.05-0.20 s` for speeds above `15 m/s`.
- Sideways or reversed flight decays faster than nose-forward flight.
- Aerodynamic force always opposes relative velocity.
- Pressure-center torque tends to align the nose with relative velocity.

### Stage 2: Ordinary Impact

Acceptance:

- Medium racket speeds produce outgoing velocity consistent with racket normal
  and racket surface velocity.
- Sweet-spot impacts are stable and repeatable.
- Edge impacts are less stable but do not explode numerically.
- No high-speed tunneling at `timestep <= 0.0005 s`.
- Cork site, not shuttle COM, is used as the impact contact point.

### Stage 3: High-Intensity Impact

Acceptance:

- Fast smash/drive-like contacts do not miss the shuttle.
- Event rebound triggers only for high-speed active contact.
- Maximum force, torque, and outgoing speed are clipped or bounded.
- After impact, aerodynamic torque restores nose-forward flight.

## Known Limits

- No per-feather collision in the first version.
- No feather bending, damage, humidity, or brand-specific material response.
- No full string FEM or individual string contact.
- Cut/slice/spin behavior is approximate through tangential damping and velocity
  scaling; it should not be treated as a strong first-version claim.
- If measured trajectory data becomes available, fit `terminal_velocity`,
  `center_of_pressure_offset`, `angle_drag_gain`, damping, restitution, and
  string-bed stiffness rather than hard-coding shot trajectories.

## References

- Local shuttle design dossier: `badminton_shuttlecock_design_dossier.md`.
- Local validation protocol: `validation_protocol.md`.
- Local racket design dossier:
  `/data3/yangfeiyang/WorkSpace/musclemimic/environment/racket/badminton_racket_design_dossier.md`.
- BWF Laws of Badminton shuttle constraints:
  https://system.bwfbadminton.com/documents/folder_1_81/Regulations/Laws/Part%20II%20Section%201A%20-%20Laws%20of%20Badminton%20-%20May%202017.pdf
