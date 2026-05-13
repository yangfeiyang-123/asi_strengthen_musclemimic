# Stage5 10-demo Discontinuity Check

Generated for latest 60Hz AMASS inputs and MyoFullBody GMR caches.

## Summary

- Retarget-before AMASS root translation and SMPL rotations do not show hard discontinuities by the thresholds used here.
- Retarget-after caches show large site-position jumps mostly at left_hand_mimic for video3-video10.
- The corresponding largest robot joint step is usually shoulder_elv_l, so the visible bad motion is concentrated in the left arm/hand chain.
- pos_error remains moderate (< 0.09 m max), so the bad visual motion is more a high-speed/jerky solution than a complete IK tracking failure.

## Metrics

| video | input max root speed m/s | input max SMPL joint deg/frame | raw mesh max vertex speed m/s | cache spike site | cache frame | site speed m/s | top robot joint | qpos step rad |
|---:|---:|---:|---:|---|---:|---:|---|---:|
| 1 | 0.75 | 16.4 | 8.40 | right_hand_mimic | 190->191 | 4.85 | elbow_flex_r | 0.144 |
| 2 | 0.44 | 19.7 | 9.23 | right_hand_mimic | 182->183 | 5.32 | elbow_flex_r | 0.220 |
| 3 | 0.50 | 18.9 | 17.55 | left_hand_mimic | 196->197 | 14.16 | shoulder_elv_l | 0.228 |
| 4 | 0.61 | 19.7 | 14.67 | left_hand_mimic | 206->207 | 11.65 | shoulder_elv_l | 0.188 |
| 5 | 0.75 | 18.5 | 28.60 | left_hand_mimic | 195->196 | 21.22 | shoulder_elv_l | 0.282 |
| 6 | 0.65 | 17.8 | 22.59 | left_hand_mimic | 200->201 | 18.22 | shoulder_elv_l | 0.271 |
| 7 | 0.69 | 14.6 | 22.33 | left_hand_mimic | 245->246 | 16.16 | shoulder_elv_l | 0.230 |
| 8 | 0.71 | 17.9 | 27.88 | left_hand_mimic | 214->215 | 20.25 | shoulder_elv_l | 0.300 |
| 9 | 0.57 | 13.5 | 17.11 | left_hand_mimic | 168->169 | 14.57 | shoulder_elv_l | 0.221 |
| 10 | 0.60 | 19.7 | 29.90 | left_hand_mimic | 210->211 | 23.51 | shoulder_elv_l | 0.338 |

## Interpretation

Input-side checks do not indicate missing-frame jumps: frame_ids are continuous for all 10 pkl files. The visible failures are concentrated after retargeting, where high-speed hand/arm site motion appears despite smooth root motion. The raw WHAM mesh also has high vertex speeds around those moments, so the retargeting likely amplifies already aggressive upper-body motion rather than creating a root trajectory discontinuity.
