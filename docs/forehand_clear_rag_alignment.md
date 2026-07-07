# Forehand-Clear BadmintonRAG Alignment

This adapter exports MyoFullBody rollout `.npz` files to the BadmintonRAG simulation CSV contract for `forehand_clear`.

## Output Contract

The generated CSV uses one row per time point and includes:

- Metadata: `sample_id`, `split`, `action_type`, `outcome_label`, `time`
- Events: `event_impact`, `event_acceleration_start`, `event_follow_through_end`
- Joint signals in degrees: `joint_trunk_rotation`, `joint_forearm_pronation`, `joint_shoulder_internal_rotation`, `joint_elbow_flexion`, `joint_wrist_extension`
- Muscle activations normalized to `[0, 1]`: `muscle_external_oblique`, `muscle_anterior_deltoid`, `muscle_triceps_brachii`, `muscle_forearm_pronator_group`
- Individual MyoFullBody muscle activations, when available, as `muscle_myo_<actuator_name>` columns. Legacy exports without `muscle_names` use `muscle_actuator_000`, `muscle_actuator_001`, etc.

## Default Mapping

| BadmintonRAG column | MyoFullBody source |
| --- | --- |
| `joint_trunk_rotation` | `axial_rotation` |
| `joint_forearm_pronation` | `pro_sup_r` |
| `joint_shoulder_internal_rotation` | `shoulder_rot_r` |
| `joint_elbow_flexion` | `elbow_flex_r` |
| `joint_wrist_extension` | `flexion_r` |
| `muscle_external_oblique` | mean of `rect_abd_r`, `rect_abd_l` |
| `muscle_anterior_deltoid` | `DELT1` |
| `muscle_triceps_brachii` | mean of `TRIlong`, `TRIlat`, `TRImed` |
| `muscle_forearm_pronator_group` | mean of `PT`, `PQ` |

`muscle_external_oblique` uses `rect_abd_*` as the stable abdominal proxy because the current MyoFullBody actuator set does not expose an explicit external-oblique label.

The semantic columns above are kept for the current BadmintonRAG rules. By default, the exporter also writes every individual muscle activation column so future evaluation rules can use finer muscle timing without re-exporting the rollout.

## Export

For a single rollout file, the default split marks episode 0 as `correct` and all later episodes as `eval`:

```bash
uv run python -m BadmintonMimic.scripts.export_forehand_clear_rag_csv \
  --input trajectory_data/myofullbody_episodes_mujoco_20260502_230942.npz \
  --output outputs/rag_alignment/forehand_clear_simulation.csv \
  --outcome-label low_speed
```

For explicit episode splits:

```bash
uv run python -m BadmintonMimic.scripts.export_forehand_clear_rag_csv \
  --input trajectory_data/myofullbody_episodes_mujoco_20260502_230942.npz \
  --correct-episodes 0,1 \
  --eval-episodes 2,3,4 \
  --output outputs/rag_alignment/forehand_clear_simulation.csv
```

For separate reference and evaluation files:

```bash
uv run python -m BadmintonMimic.scripts.export_forehand_clear_rag_csv \
  --correct-input path/to/reference_rollout.npz \
  --eval-input path/to/new_rollout.npz \
  --output outputs/rag_alignment/forehand_clear_simulation.csv
```

If measured impact timing is available, pass it explicitly:

```bash
uv run python -m BadmintonMimic.scripts.export_forehand_clear_rag_csv \
  --input path/to/rollout.npz \
  --impact-frame 42 \
  --output outputs/rag_alignment/forehand_clear_simulation.csv
```

Without `--impact-time` or `--impact-frame`, the adapter infers impact from the peak combined release velocity of shoulder rotation, elbow flexion, forearm pronation, and wrist extension.

To produce only the compact semantic muscle groups, add:

```bash
--semantic-muscles-only
```

## Validate In BadmintonRAG

```bash
cd /data3/yangfeiyang/WorkSpace/BadmintonRAG
python -m rag_project.diagnostics.data_contract \
  --csv-dataset /data3/yangfeiyang/WorkSpace/musclemimic/outputs/rag_alignment/forehand_clear_simulation.csv
```

Then run the BadmintonRAG batch diagnosis with the same CSV path.
