import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RETARGET_SCRIPT = REPO_ROOT / "musclemimic" / "badminton" / "scripts" / "run_retarget.py"
RENDER_SCRIPT = REPO_ROOT / "musclemimic" / "badminton" / "scripts" / "render_retarget_cache.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_retarget_accepts_unified_dataset_roots():
    run_retarget = _load_module(RETARGET_SCRIPT, "run_retarget_dataset_layout_for_test")

    args = run_retarget._build_parser().parse_args(
        [
            "--manifest",
            "datasets/forehandLift/manifests/ForehandNetLift/best_list.txt",
            "--amass-root",
            "datasets/forehandLift/muscle_trajectory/amass_npz",
            "--gmr-cache-root",
            "datasets/forehandLift/muscle_trajectory/gmr_cache",
        ]
    )

    assert args.amass_root == Path("datasets/forehandLift/muscle_trajectory/amass_npz")
    assert args.gmr_cache_root == Path("datasets/forehandLift/muscle_trajectory/gmr_cache")


def test_run_retarget_configures_dataset_root_environment(monkeypatch, tmp_path):
    run_retarget = _load_module(RETARGET_SCRIPT, "run_retarget_env_layout_for_test")
    amass_root = tmp_path / "datasets" / "forehandLift" / "muscle_trajectory" / "amass_npz"
    gmr_cache_root = tmp_path / "datasets" / "forehandLift" / "muscle_trajectory" / "gmr_cache"

    for key in (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "MUJOCO_GL",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "MUSCLEMIMIC_AMASS_PATH",
        "AMASS_PATH",
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
        "CONVERTED_AMASS_PATH",
        "MUSCLEMIMIC_GMR_CACHE_PATH",
        "MUSCLEMIMIC_SMPL_MODEL_PATH",
        "SMPL_MODEL_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    run_retarget._configure_env(
        project_root=tmp_path / "musclemimic" / "badminton",
        repo_root=tmp_path,
        amass_root=amass_root,
        gmr_cache_root=gmr_cache_root,
    )

    assert run_retarget.os.environ["MUSCLEMIMIC_AMASS_PATH"] == str(amass_root)
    assert run_retarget.os.environ["AMASS_PATH"] == str(amass_root)
    assert run_retarget.os.environ["MUSCLEMIMIC_GMR_CACHE_PATH"] == str(gmr_cache_root)


def test_render_cache_accepts_direct_gmr_cache_root():
    render_retarget_cache = _load_module(RENDER_SCRIPT, "render_cache_dataset_layout_for_test")

    args = render_retarget_cache._build_parser().parse_args(
        [
            "--motion",
            "ForehandNetLift/best/video01_best_stage7_smpl",
            "--cache-root",
            "datasets/forehandLift/muscle_trajectory/gmr_cache",
        ]
    )

    assert args.cache_root == Path("datasets/forehandLift/muscle_trajectory/gmr_cache")
    assert render_retarget_cache._resolve_cache_path(
        args.cache_root,
        "ForehandNetLift/best/video01_best_stage7_smpl",
    ) == Path("datasets/forehandLift/muscle_trajectory/gmr_cache/ForehandNetLift/best/video01_best_stage7_smpl.npz")
