# Stage5 Smooth Filtered Retarget Discontinuity Report

Scope: 10 `forehand_clear/stage5_10demo_smooth_filtered` motions. This is the final training-oriented version generated after the IK-only smooth pass did not reduce the largest left-hand site spikes.

## Summary

- AMASS input remains smooth at 60Hz: worst root translation speed is 0.753 m/s.
- Baseline retarget worst site speed: 23.506 m/s.
- IK-only smooth retarget worst site speed: 23.791 m/s.
- Final filtered retarget worst site speed: 4.477 m/s.
- Max raw qpos step changed baseline -> IK-only -> filtered: 0.338 -> 0.349 -> 0.040.
- The final filtered cache applies post-retarget Gaussian low-pass filtering and scalar joint velocity limiting, then recomputes qvel and all body/site FK fields from the filtered qpos.

## Per-Motion Metrics

| video | AMASS fps | AMASS frames | AMASS root max speed (m/s) | baseline site max (m/s) | IK-only site max (m/s) | filtered site max (m/s) | filtered worst site | baseline qpos step | IK-only qpos step | filtered qpos step |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 60.0 | 154 | 0.749 | 4.845 | 4.642 | 3.883 | right_ankle_mimic@45 | 0.189 | 0.192 | 0.040 |
| 2 | 60.0 | 155 | 0.441 | 5.325 | 5.048 | 3.060 | right_hand_mimic@199 | 0.224 | 0.223 | 0.040 |
| 3 | 60.0 | 173 | 0.496 | 14.164 | 14.308 | 3.190 | right_hand_mimic@221 | 0.228 | 0.232 | 0.040 |
| 4 | 60.0 | 176 | 0.614 | 11.651 | 11.913 | 3.439 | left_hand_mimic@215 | 0.230 | 0.227 | 0.040 |
| 5 | 60.0 | 179 | 0.753 | 21.223 | 21.679 | 4.477 | left_hand_mimic@195 | 0.282 | 0.302 | 0.040 |
| 6 | 60.0 | 168 | 0.648 | 18.222 | 18.375 | 4.355 | left_hand_mimic@203 | 0.277 | 0.292 | 0.040 |
| 7 | 60.0 | 190 | 0.689 | 16.160 | 16.365 | 3.814 | left_hand_mimic@246 | 0.230 | 0.234 | 0.040 |
| 8 | 60.0 | 185 | 0.710 | 20.248 | 20.433 | 3.719 | left_hand_mimic@215 | 0.300 | 0.306 | 0.040 |
| 9 | 60.0 | 146 | 0.570 | 14.565 | 14.712 | 3.371 | right_hand_mimic@192 | 0.221 | 0.224 | 0.040 |
| 10 | 60.0 | 173 | 0.602 | 23.506 | 23.791 | 3.810 | left_hand_mimic@218 | 0.338 | 0.349 | 0.040 |

## Final Artifacts

- Final manifest: `BadmintonMimic/manifests/stage5_10demo_smooth_filtered_list.txt`
- Final AMASS namespace: `BadmintonMimic/data/amass_npz/forehand_clear/stage5_10demo_smooth_filtered`
- Final MyoFullBody cache: `caches/AMASS/MyoFullBody/gmr/forehand_clear/stage5_10demo_smooth_filtered`
- Final 60fps videos: `visualize/msk_retarget/forehand_clear_stage5_10demo_smooth_filtered_video*_lower_body_full_poses.mp4`
- Filtering script: `BadmintonMimic/scripts/filter_retarget_cache.py`

## Recommendation

Use the `stage5_10demo_smooth_filtered` cache/manifest for training when the priority is smooth and trainable motion. Keep the earlier `stage5_10demo` and `stage5_10demo_smooth` outputs only for comparison/debugging.
