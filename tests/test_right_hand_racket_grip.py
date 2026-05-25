import math
import tomllib

from src.grip.paths import REPO_ROOT, racket_xml_path, target_config_path
from src.grip.target_config import GripTargetConfig, load_grip_target_config


def test_package_discovery_includes_local_src_package():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    includes = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "src*" in includes


def test_repo_paths_resolve_existing_racket_asset():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / ".git").exists()
    assert racket_xml_path().is_file()
    assert racket_xml_path().name == "badminton_racket_rigid.xml"


def test_target_config_default_path_is_repo_level_configs():
    path = target_config_path()
    assert path == REPO_ROOT / "configs" / "right_hand_racket_grip_targets.json"
    assert path.parent.is_dir()


def test_load_default_grip_targets():
    config = load_grip_target_config()
    assert isinstance(config, GripTargetConfig)
    assert config.handle_radius_m == 0.014
    assert set(config.target_points_racket_local) == {"palm", "thumb", "index", "middle", "ring", "pinky"}


def test_target_point_conversion_uses_racket_local_cylinder():
    config = load_grip_target_config()
    palm = config.target_xyz("palm")
    assert palm[1] == 0.085
    assert math.isclose(palm[0], -0.014, abs_tol=1e-9)
    assert math.isclose(palm[2], 0.0, abs_tol=1e-9)
