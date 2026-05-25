# Validation Protocol

## Geometry

Check that:

- Overall length <= 0.680 m.
- Overall width <= 0.230 m.
- Stringbed length <= 0.280 m.
- Stringbed width <= 0.220 m.

Run:

```bash
python src/validate_racket_params.py
```

## Mass properties

Target:

- Mass: 0.085–0.095 kg.
- Balance from butt: 0.285–0.320 m.
- Swingweight about local +X axis at y=0.090 m: 90–97 kg cm^2.

Formula:

```text
I_s = Ixx_COM + m * (y_com - 0.090)^2
```

## Static stringbed stiffness

At the center of the stringbed:

- Displace shuttle cork proxy by 5 mm into the plane.
- Target reaction force: 45–55 N.
- Nominal model: 9600 N/m × 0.005 m = 48 N.

## Dynamic impact

Perform center and off-center impacts at incoming speeds:

```text
5, 10, 20, 40 m/s
```

Record:

- Max penetration.
- Max normal force.
- Outgoing velocity.
- Effective contact duration.
- Racket applied force/torque.
- Shuttle attitude/spin change.

For high-speed missed contacts, use event-based rebound or reduce timestep to 0.0005 s or below.

## Flex proxy

When using `badminton_racket_flex_proxy.xml`:

- Check that `shaft_flex_x` and `shaft_flex_z` remain within ±8 deg during normal strokes.
- Tune stiffness upward if the head visibly lags too much.
- Tune damping upward if the head oscillates unrealistically after impact.
