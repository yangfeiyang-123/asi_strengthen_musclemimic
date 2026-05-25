from src.grip.paths import REPO_ROOT, racket_xml_path, target_config_path


def test_repo_paths_resolve_existing_racket_asset():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / ".git").exists()
    assert racket_xml_path().is_file()
    assert racket_xml_path().name == "badminton_racket_rigid.xml"


def test_target_config_default_path_is_repo_level_configs():
    path = target_config_path()
    assert path == REPO_ROOT / "configs" / "right_hand_racket_grip_targets.json"
