import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT_SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "convert_wham_to_amass.py"
RETARGET_SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "run_retarget.py"
CONFIG_SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "build_config_from_manifests.py"
RENDER_SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "render_retarget_cache.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_wham_force_fps_overrides_input_metadata(tmp_path):
    input_path = tmp_path / "wham.pkl"
    output_path = tmp_path / "motion_poses.npz"
    n_frames = 4
    wham_data = {
        "pose_world": np.zeros((n_frames, 156), dtype=np.float32),
        "trans_world": np.zeros((n_frames, 3), dtype=np.float32),
        "betas": np.zeros(10, dtype=np.float32),
        "fps": 60.0,
    }
    with input_path.open("wb") as f:
        pickle.dump(wham_data, f)

    subprocess.run(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--fps",
            "30",
            "--force-fps",
            "--gender",
            "neutral",
        ],
        check=True,
    )

    converted = np.load(output_path, allow_pickle=True)
    assert float(converted["mocap_framerate"]) == 30.0
    assert float(converted["mocap_frame_rate"]) == 30.0


def test_run_retarget_rejects_manifest_motion_with_wrong_fps(tmp_path):
    run_retarget = _load_module(RETARGET_SCRIPT, "run_retarget_for_test")
    motion_path = tmp_path / "badminton" / "train" / "clip_poses.npz"
    motion_path.parent.mkdir(parents=True)
    np.savez(
        motion_path,
        poses=np.zeros((4, 156), dtype=np.float32),
        trans=np.zeros((4, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(60.0, dtype=np.float32),
        mocap_frame_rate=np.asarray(60.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="expected 30"):
        run_retarget._validate_motion_fps(["badminton/train/clip_poses"], tmp_path, 30)


def test_run_retarget_accepts_fps_alias_for_target_fps():
    run_retarget = _load_module(RETARGET_SCRIPT, "run_retarget_parser_for_test")

    args = run_retarget._build_parser().parse_args(["--split", "train", "--fps", "60"])

    assert args.target_fps == 60


def test_run_retarget_accepts_manifest_motion_with_matching_60hz_fps(tmp_path):
    run_retarget = _load_module(RETARGET_SCRIPT, "run_retarget_validate_60_for_test")
    motion_path = tmp_path / "badminton" / "train" / "clip_poses.npz"
    motion_path.parent.mkdir(parents=True)
    np.savez(
        motion_path,
        poses=np.zeros((4, 156), dtype=np.float32),
        trans=np.zeros((4, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(60.0, dtype=np.float32),
        mocap_frame_rate=np.asarray(60.0, dtype=np.float32),
    )

    run_retarget._validate_motion_fps(["badminton/train/clip_poses"], tmp_path, 60)


def test_build_config_uses_requested_target_fps(tmp_path):
    build_config = _load_module(CONFIG_SCRIPT, "build_config_for_test")
    output = tmp_path / "conf.yaml"

    build_config.build_config(
        ["badminton/train/clip_poses"],
        ["badminton/val/clip_poses"],
        output,
        num_envs=16,
        total_timesteps=1000,
        target_fps=60,
    )

    text = output.read_text()
    assert text.count("target_fps: 60") == 2
    assert "target_fps: 30" not in text


def test_render_defaults_to_cache_frequency_divided_by_stride():
    render_cache = _load_module(RENDER_SCRIPT, "render_cache_for_test")

    assert render_cache._resolve_output_fps(cache_frequency=100.0, stride=4, requested_fps=None) == 25.0
    assert render_cache._resolve_output_fps(cache_frequency=100.0, stride=1, requested_fps=None) == 100.0
    assert render_cache._resolve_output_fps(cache_frequency=100.0, stride=1, requested_fps=60.0) == 60.0


def test_render_sample_fps_selects_frames_by_time():
    render_cache = _load_module(RENDER_SCRIPT, "render_cache_sample_for_test")

    frame_ids = render_cache._select_frame_ids(n_frames=100, cache_frequency=100.0, stride=1, sample_fps=30.0)

    assert len(frame_ids) == 30
    assert frame_ids[:5] == [0, 3, 7, 10, 13]
    assert frame_ids[-1] == 97
