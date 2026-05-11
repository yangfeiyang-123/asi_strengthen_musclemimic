# CSV Marker Mocap to MoSh++ Pipeline

This package implements a practical pipeline:

`CSV mocap marker data -> cleaned marker NPZ -> labeled C3D -> official MoSh++ -> SMPL/SMPL-H/SMPL-X parameters -> AMASS-style NPZ`.

It does not reimplement a MoSh-style optimizer. MoSh++ remains an external dependency and must be supplied with a valid marker-to-body correspondence layout.

## Why C3D

Official MoSh++ workflows are built around labeled marker-based mocap, commonly stored as C3D, plus marker-to-body correspondences. Converting the CSV to a clean, validated C3D makes the data usable by standard mocap tools and by MoSh++ examples.

## Dependencies

Required for CSV processing and reports:

- numpy
- pandas
- pyyaml
- matplotlib

Required for C3D read/write:

- ezc3d

Optional for body-model rendering:

- smplx
- human_body_prior
- trimesh / pyrender

External requirements:

- official MoSh++ checkout
- SMPL, SMPL-H, or SMPL-X model files downloaded by the user

## MoSh++ Setup

Do not copy MoSh++ into this project. Install or clone it elsewhere and pass the path with `--moshpp_dir`.

MoSh++ commonly requires Python 3.7 and `chumpy`; many users install it inside the SOMA conda environment. Body model files must be downloaded from the official model sources and placed in a `body_models` directory. This project can still parse CSV, clean markers, write C3D, and validate C3D without MoSh++ installed.

## Full Run

From the project root:

```bash
python scripts/parse_mocap_csv.py --csv clip1/mocap_clip_1_440.csv --out outputs/markers_raw.npz --name_map outputs/marker_name_map.yaml
python scripts/inspect_markers.py --npz outputs/markers_raw.npz --out_dir outputs/inspect
python scripts/preprocess_markers.py --in_npz outputs/markers_raw.npz --out_npz outputs/markers_clean.npz --max_gap 10
python scripts/normalize_marker_names.py --markers_npz outputs/markers_clean.npz --out_map configs/marker_name_map.yaml --out_npz outputs/markers_clean_safe_names.npz
python scripts/csv_to_c3d.py --in_npz outputs/markers_clean_safe_names.npz --out_c3d outputs/c3d/mocap_clip_1_440.c3d --units mm
python scripts/validate_c3d.py --c3d outputs/c3d/mocap_clip_1_440.c3d --ref_npz outputs/markers_clean_safe_names.npz --out_json outputs/c3d/c3d_validation_report.json
python scripts/prepare_moshpp_config.py --c3d outputs/c3d/mocap_clip_1_440.c3d --marker_name_map configs/marker_name_map.yaml --moshpp_dir /path/to/moshpp --body_model_dir /path/to/body_models --model_type smplh --gender neutral --out_dir outputs/moshpp_config
python scripts/run_moshpp.py --c3d outputs/c3d/mocap_clip_1_440.c3d --config_dir outputs/moshpp_config --moshpp_dir /path/to/moshpp --conda_env soma --out_dir outputs/moshpp_run
python scripts/convert_moshpp_output.py --moshpp_out outputs/moshpp_run --out_npz outputs/amass_style/mocap_clip_1_440_amass_style.npz
python scripts/visualize_c3d_markers.py --c3d outputs/c3d/mocap_clip_1_440.c3d --out_mp4 outputs/vis/c3d_markers_preview.mp4
python scripts/visualize_moshpp_result.py --amass_npz outputs/amass_style/mocap_clip_1_440_amass_style.npz --c3d outputs/c3d/mocap_clip_1_440.c3d --body_model_dir /path/to/body_models --out_mp4 outputs/vis/moshpp_fit_preview.mp4
```

## Important Limits

Converting to C3D does not guarantee good SMPL fitting. Stable marker labels and correct marker-to-SMPL surface correspondences are the critical part.

Chinese marker names and special characters are normalized to safe English labels for C3D and MoSh++. Long missing tracks are kept invalid; short gaps are interpolated by default. Fast badminton swings can cause right-hand marker loss and speed outliers, so inspect the missing heatmap and speed report before fitting. If `marker_layout.yaml` lacks valid vertex IDs, MoSh++ may fail or produce unstable results.
