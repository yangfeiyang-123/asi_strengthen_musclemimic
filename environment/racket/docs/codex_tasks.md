# Codex Task List

1. Load `params/racket_nominal.json` and verify units and coordinate frame.
2. Run `python src/validate_racket_params.py`.
3. Open `assets/badminton_racket_rigid.xml` and confirm all key sites:
   - `grip_pose_site`
   - `butt_site`
   - `stringbed_center_site`
   - `head_tip_site`
4. Add the racket to the target MuJoCo scene.
5. Attach `grip_pose_site` to the robot end-effector or SMPL hand.
6. Ensure the shuttlecock model exposes a cork/base site, preferably `shuttle_cork_site`.
7. Call `apply_stringbed_force(...)` once per MuJoCo step before `mj_step`.
8. For high-speed impacts, add event detection and use `stringbed_rebound_velocity(...)` to correct missed collisions.
9. Calibrate `center_normal_stiffness`, `normal_damping`, and `event_restitution_normal` against real or desired outgoing shuttle speed.
10. If shaft bending is needed, switch to `badminton_racket_flex_proxy.xml` and apply stringbed force to body `racket_head`.
