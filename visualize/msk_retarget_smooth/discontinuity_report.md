# Smooth Retarget Discontinuity Report

Scope: latest 10 `forehand_clear/stage5_10demo_smooth` AMASS inputs and their MyoFullBody retarget caches. Baseline is the previous `stage5_10demo` retarget cache.

## Summary
- AMASS input root translation is smooth at 60Hz: worst root speed 0.753 m/s; no evidence of large source translation jumps.
- Retarget site-speed worst case changed from 23.506 m/s to 23.791 m/s.
- Retarget max raw qpos frame step changed from 0.338 to 0.349.
- The smooth pass uses higher IK damping, per-joint velocity limits inside GMR, and lower arm/hand end-effector weights. It keeps the source SMPL/AMASS files untouched and writes a separate cache namespace.

## Per-Motion Metrics

| video | AMASS fps | AMASS frames | AMASS max root speed (m/s) | AMASS max pose step (rad) | baseline max site speed (m/s) | smooth max site speed (m/s) | baseline worst site | smooth worst site | baseline max qpos step | smooth max qpos step |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 60.0 | 154 | 0.749 | 5.792 | 4.845 | 4.642 | right_hand_mimic@190 | right_hand_mimic@194 | 0.189 | 0.192 |
| 2 | 60.0 | 155 | 0.441 | 6.111 | 5.325 | 5.048 | right_hand_mimic@182 | right_hand_mimic@183 | 0.224 | 0.223 |
| 3 | 60.0 | 173 | 0.496 | 6.141 | 14.164 | 14.308 | left_hand_mimic@196 | left_hand_mimic@196 | 0.228 | 0.232 |
| 4 | 60.0 | 176 | 0.614 | 6.186 | 11.651 | 11.913 | left_hand_mimic@206 | left_hand_mimic@206 | 0.230 | 0.227 |
| 5 | 60.0 | 179 | 0.753 | 6.130 | 21.223 | 21.679 | left_hand_mimic@195 | left_hand_mimic@195 | 0.282 | 0.302 |
| 6 | 60.0 | 168 | 0.648 | 6.104 | 18.222 | 18.375 | left_hand_mimic@200 | left_hand_mimic@200 | 0.277 | 0.292 |
| 7 | 60.0 | 190 | 0.689 | 5.746 | 16.160 | 16.365 | left_hand_mimic@245 | left_hand_mimic@245 | 0.230 | 0.234 |
| 8 | 60.0 | 185 | 0.710 | 6.227 | 20.248 | 20.433 | left_hand_mimic@214 | left_hand_mimic@214 | 0.300 | 0.306 |
| 9 | 60.0 | 146 | 0.570 | 6.068 | 14.565 | 14.712 | left_hand_mimic@168 | left_hand_mimic@168 | 0.221 | 0.224 |
| 10 | 60.0 | 173 | 0.602 | 6.169 | 23.506 | 23.791 | left_hand_mimic@210 | left_hand_mimic@210 | 0.338 | 0.349 |

## Interpretation

- The source SMPL/AMASS motion is not the main discontinuity source by root translation checks; the large artifacts originate during MyoFullBody retargeting.
- The smooth run strongly reduces the left-hand site spikes that appeared in several baseline videos, but raw joint-space spikes remain in some clips. For training, this cache is safer than the baseline, yet video 5/8/10 still deserve visual inspection before long training runs.
- If stricter smoothness is needed, the next implementation should add a post-retarget qpos low-pass / velocity-clamp pass before FK cache extension, then regenerate site data from the filtered qpos.

## Generated Artifacts

- Smooth AMASS namespace: `BadmintonMimic/data/amass_npz/forehand_clear/stage5_10demo_smooth`
- Smooth manifest: `BadmintonMimic/manifests/stage5_10demo_smooth_list.txt`
- Smooth cache: `caches/AMASS/MyoFullBody/gmr/forehand_clear/stage5_10demo_smooth`
- Smooth videos: `visualize/msk_retarget_smooth/*.mp4`
