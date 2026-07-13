from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from musclemimic.badminton.data_qc import TRAIN_MOTIONS, VAL_MOTIONS
from musclemimic.runner.engine import validate_training_source_preflight


def _config(tmp_path):
    prefix = "forehandClear_standard/muscle_trajectory/raw_smooth_v1"
    dataset_conf = {
        "retargeting_method": "gmr",
        "clear_cache": False,
        "gmr_config": {
            "target_fps": 60,
            "solver": "daqp",
            "damping": 1.0,
            "use_velocity_limit": True,
            "ik_config_path": "smooth_ik.json",
        },
    }
    return OmegaConf.create(
        {
            "experiment": {
                "training_action": "forehandClear_standard",
                "training_source": {
                    "source_mode": "existing_ppo",
                    "variant": "raw_smooth_v1",
                    "source_namespace": "temp/raw_smooth_v1",
                    "cache_namespace": "muscle_trajectory/raw_smooth_v1",
                    "source_recipe": "recipe.json",
                    "release_manifest": (
                        "datasets/forehandClear_standard/manifests/"
                        "raw_smooth_v1/release_manifest.json"
                    ),
                    "source_fps": 60,
                    "cache_fps": 100,
                },
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            **dataset_conf,
                            "rel_dataset_path": [
                                f"{prefix}/{motion}" for motion in TRAIN_MOTIONS
                            ],
                        }
                    }
                },
                "validation": {
                    "amass_dataset_conf": {
                        **dataset_conf,
                        "rel_dataset_path": [
                            f"{prefix}/{motion}" for motion in VAL_MOTIONS
                        ],
                    }
                },
            }
        }
    )


def _prepare_paths(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    release = (
        datasets
        / "forehandClear_standard"
        / "manifests"
        / "raw_smooth_v1"
        / "release_manifest.json"
    )
    release.parent.mkdir(parents=True)
    release.write_text('{"release_sha256":"' + "a" * 64 + '"}\n', encoding="utf-8")
    (tmp_path / "recipe.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "smooth_ik.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("MUSCLEMIMIC_GMR_CACHE_PATH", str(datasets))
    return release


def test_runtime_preflight_binds_release_and_warning_free_qc_before_training(
    tmp_path, monkeypatch
):
    release = _prepare_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "musclemimic.badminton.scripts.data_release.validate_release_manifest",
        lambda *_args, **_kwargs: {
            "passed": True,
            "release_sha256": "a" * 64,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "musclemimic.badminton.data_qc.inspect_canonical_dataset",
        lambda *_args, **_kwargs: {
            "schema_version": "forehand_clear_data_qc_v3",
            "passed": True,
            "clean_passed": True,
            "warnings": [],
            "hard_errors": [],
        },
    )
    monkeypatch.setattr(
        "musclemimic.badminton.scripts.finalize_raw_smooth_visual_qc.validate_report",
        lambda *_args, **_kwargs: {
            "passed": True,
            "report_sha256": "b" * 64,
            "errors": [],
        },
    )
    config = _config(tmp_path)
    out = tmp_path / "hydra-output"
    report = validate_training_source_preflight(
        config,
        launch_dir=tmp_path,
        result_dir=out,
    )

    assert report["passed"] is True
    assert report["release_manifest_content_sha256"]
    assert report["visual_qc_report_content_sha256"] == "b" * 64
    assert report["release_manifest"] == str(release.resolve())
    assert len(config.experiment.training_source.preflight_binding_sha256) == 64
    persisted = json.loads((out / "training_source_preflight.json").read_text())
    assert persisted == report


def test_runtime_preflight_rejects_qc_warning_and_wrong_namespace(tmp_path, monkeypatch):
    _prepare_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "musclemimic.badminton.scripts.data_release.validate_release_manifest",
        lambda *_args, **_kwargs: {
            "passed": True,
            "release_sha256": "a" * 64,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "musclemimic.badminton.data_qc.inspect_canonical_dataset",
        lambda *_args, **_kwargs: {
            "passed": True,
            "clean_passed": False,
            "warnings": ["one discontinuity"],
            "hard_errors": [],
        },
    )
    with pytest.raises(ValueError, match="strict data QC failed"):
        validate_training_source_preflight(
            _config(tmp_path), launch_dir=tmp_path, result_dir=tmp_path / "out"
        )

    wrong = _config(tmp_path)
    wrong.experiment.training_source.cache_namespace = "muscle_trajectory/raw"
    with pytest.raises(ValueError, match="cache_namespace"):
        validate_training_source_preflight(
            wrong, launch_dir=tmp_path, result_dir=tmp_path / "out"
        )


def test_runtime_preflight_rejects_stale_release_manifest(tmp_path, monkeypatch):
    _prepare_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "musclemimic.badminton.scripts.data_release.validate_release_manifest",
        lambda *_args, **_kwargs: {
            "passed": False,
            "release_sha256": "a" * 64,
            "errors": ["cache content hash changed"],
        },
    )
    with pytest.raises(ValueError, match="release validation failed.*content hash"):
        validate_training_source_preflight(
            _config(tmp_path), launch_dir=tmp_path, result_dir=tmp_path / "out"
        )


def test_runtime_preflight_does_not_claim_other_source_modes(tmp_path):
    config = OmegaConf.create(
        {"experiment": {"training_source": {"source_mode": "reference_bundle"}}}
    )
    assert (
        validate_training_source_preflight(
            config, launch_dir=tmp_path, result_dir=tmp_path / "out"
        )
        is None
    )
    generic = OmegaConf.create(
        {
            "experiment": {
                "training_action": None,
                "training_source": {"source_mode": "existing_ppo"},
            }
        }
    )
    assert (
        validate_training_source_preflight(
            generic, launch_dir=tmp_path, result_dir=tmp_path / "out"
        )
        is None
    )
