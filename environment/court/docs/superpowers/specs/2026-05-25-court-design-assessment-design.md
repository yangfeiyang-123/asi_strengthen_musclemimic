# Court Directory Design Assessment

Date: 2026-05-25
Scope: `/data3/yangfeiyang/WorkSpace/musclemimic/environment/court`

## Summary

The current court directory design is reasonable. It has a clear generation pipeline:

```text
params/court_bwf_nominal.json
  -> src/court_geometry.py
  -> src/generate_court_mjcf.py
  -> assets/badminton_court_bwf_visual.xml
  -> assets/badminton_court_bwf_collision_net.xml
```

The strongest part of the design is that `court_geometry.py` centralizes the official BWF dimensions, edge-correct line semantics, rally bounds, service bounds, and net height profile. The MJCF generator consumes those helpers instead of re-encoding most court geometry independently.

The directory is not fully integration-ready yet. The main gaps are repository hygiene, generated-asset drift prevention, and real MuJoCo compile validation in an environment with `mujoco` installed.

Recommended path: keep the current architecture and tighten the edges. Do not perform a large refactor now.

## Evidence Reviewed

Reviewed files:

```text
README.md
badminton_court_design_dossier.md
docs/validation_protocol.md
docs/codex_tasks.md
params/court_bwf_nominal.json
src/court_geometry.py
src/generate_court_mjcf.py
src/validate_court_params.py
assets/badminton_court_bwf_visual.xml
assets/badminton_court_bwf_collision_net.xml
```

Validation run:

```bash
python src/validate_court_params.py
```

Result: all static court design checks passed.

MuJoCo runtime compile validation was not run because the current Python environment does not have the `mujoco` package installed.

## Architecture Assessment

The package responsibilities are mostly well separated:

- `params/` stores nominal official dimensions and simulation settings.
- `src/court_geometry.py` acts as the single source of truth for derived geometry and rule helpers.
- `src/generate_court_mjcf.py` converts geometry into MJCF assets.
- `src/validate_court_params.py` checks dimensions, rule helpers, XML parseability, and key MJCF elements.
- `assets/` stores the generated visual-only and collision-net XML outputs.
- `docs/` records validation protocol and future integration tasks.

This structure is appropriate for a MuJoCo asset package. The current module size is also acceptable: the geometry helper is focused, the generator is asset-specific, and the validator is direct enough to understand.

The main architecture weakness is that generated XML is stored beside the generator without an explicit drift policy. A future edit can change `params` or `src` while leaving `assets` stale. The package should either treat `assets` as committed generated artifacts with a regeneration check, or treat them as build outputs and exclude them from source control. For integration with other MuJoCo scenes, committed generated assets are probably more convenient, but they need drift detection.

## MuJoCo And Physics Assessment

The physical modeling choices are reasonable for training and evaluation:

- Court lines are visual-only with `contype="0"` and `conaffinity="0"`, which avoids artificial bumps when the shuttle, robot feet, or racket contact the floor.
- The floor is the only default collision surface and extends beyond the legal court for practical movement margin.
- The visual asset disables net collision by default, which is a sensible training default.
- The collision-net asset separates contact testing from normal visual use.
- The net post center is offset outside the legal doubles sideline, avoiding support intrusion into the court.
- The parabolic net sag profile matches the documented center and sideline heights.

The biggest physical caveat is the collision-net model. It uses thin rigid proxy geoms, which is acceptable for blocking/contact diagnostics but not a realistic flexible net. For learning tasks, net crossing and net contact should likely be event-based first, with the collision-net asset used for evaluation or targeted diagnostics.

## Data Flow

Expected flow:

1. Edit official and simulation parameters in `params/court_bwf_nominal.json`.
2. Load parameters through `CourtParams.from_json`.
3. Use `CourtParams` methods for line placement, rally classification, service classification, and net profile queries.
4. Generate MJCF assets from `src/generate_court_mjcf.py`.
5. Validate static geometry and XML structure with `src/validate_court_params.py`.
6. In downstream environments, use `badminton_court_bwf_visual.xml` by default and switch to `badminton_court_bwf_collision_net.xml` only for net-contact testing.

The data flow is coherent. The key implementation rule is that downstream tasks should use `court_geometry.py` for landing legality instead of duplicating court bounds.

## Issues And Optimization Suggestions

1. Remove generated Python cache files from the package.

   The directory currently contains `src/__pycache__/court_geometry.cpython-313.pyc`. This should not be tracked and should be covered by `.gitignore`.

2. Decide and document the generated-asset policy.

   Recommended policy: commit `assets/*.xml` because they are useful for MuJoCo scene includes, but require a drift check in validation or CI. The drift check should regenerate XML into a temporary location or compare regenerated output against committed assets.

3. Add real MuJoCo compile validation when the dependency is available.

   XML parseability with `xml.etree.ElementTree` is useful but weaker than `mujoco.MjModel.from_xml_path`. The validation protocol already describes this; it should become an optional or CI test in an environment with MuJoCo installed.

4. Add focused pytest tests.

   Recommended tests:

   ```text
   tests/test_court_geometry.py
   tests/test_court_xml.py
   tests/test_generated_assets.py
   ```

   These should cover line-inclusion semantics, singles/doubles differences, service bounds, XML key elements, collision group separation, and generated asset drift.

5. Keep collision-net as a diagnostic asset, not the default training surface.

   The rigid proxy net is useful for checking contact but can create unrealistic dynamics. The default should remain visual-only plus event-based net crossing/contact logic.

6. Add integration adapters rather than changing court coordinates.

   The current coordinate system is good: net at `x=0`, court centerline at `y=0`, ground at `z=0`. If SMPL, racket, or player-role conventions differ, adapters should map to this coordinate system instead of changing the court package.

## Visual Assessment

The companion visualization shows two useful views:

- a package responsibility map showing `params -> geometry -> generator -> assets`;
- a top-down court diagram showing net position, court boundaries, singles lines, service lines, and the visual-only line semantics.

The diagrams support the same conclusion as the code review: the conceptual design is sound, and the remaining work is boundary hardening rather than redesign.

## Accepted Approach

Accepted approach: keep the architecture, tighten the edges.

Rejected alternatives:

- Minimal acceptance: too weak because it ignores generated-asset drift and MuJoCo compile risk.
- Full package refactor now: more work than necessary for the current asset scope.

## Completion Criteria For Future Hardening

The court package can be considered integration-ready when:

- cache files are absent and ignored;
- `python src/validate_court_params.py` passes;
- generated assets are confirmed current with source parameters and generator code;
- MuJoCo can compile both XML assets in a dependency-ready environment;
- tests cover geometry semantics and XML collision/visual separation;
- downstream shuttlecock, racket, and SMPL code call the court helper instead of duplicating court bounds.
