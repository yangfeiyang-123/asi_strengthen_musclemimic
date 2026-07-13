from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from musclemimic.badminton.data_qc import TRAIN_MOTIONS, VAL_MOTIONS
from musclemimic.badminton.scripts import data_release
from musclemimic.badminton.scripts import materialize_raw_smooth_cache_override as cache_override
from musclemimic.badminton.scripts import prepare_raw_smooth_sources as prepare

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_RECIPE = (
    REPO_ROOT / "musclemimic/badminton/scripts/raw_smooth_v1_recipe.json"
)
PRODUCTION_OVERRIDE_RECIPE = (
    REPO_ROOT / "musclemimic/badminton/scripts/raw_smooth_v1_cache_overrides.json"
)


def _write_source(path: Path, frames: int, *, video1_reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trans = np.stack(
        [np.arange(frames, dtype=np.float32) * 0.01, np.zeros(frames), np.ones(frames)],
        axis=-1,
    )
    if video1_reset:
        trans[29:] += np.asarray([4.0, -2.0, 0.5], dtype=np.float32)
    poses = np.zeros((frames, 72), dtype=np.float32)
    np.savez(
        path,
        poses=poses,
        root_orient=poses[:, :3],
        pose_body=poses[:, 3:66],
        left_hand_pose=np.zeros((frames, 45), dtype=np.float32),
        right_hand_pose=np.zeros((frames, 45), dtype=np.float32),
        trans=trans,
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(60.0, dtype=np.float32),
        mocap_frame_rate=np.asarray(60.0, dtype=np.float32),
    )


def _write_cache(path: Path, frames: int, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.zeros((frames, 89), dtype=np.float32)
    qpos[:, 0] = np.arange(frames, dtype=np.float32) * 0.001 + offset
    np.savez(
        path,
        qpos=qpos,
        qvel=np.zeros((frames, 88), dtype=np.float32),
        frequency=np.asarray(100.0, dtype=np.float32),
    )


def _write_split_manifests(root: Path) -> None:
    manifest_dir = root / "manifests/raw_smooth_v1"
    manifest_dir.mkdir(parents=True)
    prefix = "forehandClear_standard/muscle_trajectory/raw_smooth_v1"
    (manifest_dir / "train_list.txt").write_text(
        "".join(f"{prefix}/{motion}\n" for motion in TRAIN_MOTIONS), encoding="utf-8"
    )
    (manifest_dir / "val_list.txt").write_text(
        "".join(f"{prefix}/{motion}\n" for motion in VAL_MOTIONS), encoding="utf-8"
    )


def _build_fixture(tmp_path: Path, monkeypatch):
    root = tmp_path / "forehandClear_standard"
    source_dir = root / "wham/initiall_wham"
    source_recipe = copy.deepcopy(json.loads(PRODUCTION_RECIPE.read_text(encoding="utf-8")))
    frame_counts = {
        "6月2日-3": 132,
        "6月2日-5": 128,
        "6月2日-6": 120,
        "6月2日-7": 119,
        "video1": 154,
    }
    original_hashes: dict[str, str] = {}
    for identity in source_recipe["identity"]:
        motion = identity["motion"]
        source = source_dir / f"{motion}__original.npz"
        _write_source(
            source,
            frame_counts.get(motion, 120),
            video1_reset=motion == "video1",
        )
        digest = prepare.sha256_file(source)
        identity["source_sha256"] = digest
        original_hashes[motion] = digest

    override_recipe = copy.deepcopy(
        json.loads(PRODUCTION_OVERRIDE_RECIPE.read_text(encoding="utf-8"))
    )
    cache_hashes: dict[str, str] = {}
    cache_frames = {
        "6月2日-1": 183,
        "6月2日-2": 195,
        "6月2日-3": 217,
        "6月2日-4": 185,
        "6月2日-5": 210,
        "6月2日-7": 195,
    }
    source_overrides: dict[str, dict] = {}
    for index, entry in enumerate(override_recipe["overrides"]):
        motion = entry["motion"]
        protected = root / entry["source_cache_path"]
        _write_cache(protected, cache_frames[motion], offset=float(index))
        digest = prepare.sha256_file(protected)
        entry["source_cache_sha256"] = digest
        cache_hashes[motion] = digest
        if entry["source_output_override"] is not None:
            contract = entry["source_output_override"]
            contract["protected_source_sha256"] = original_hashes[motion]
            source_overrides[motion] = copy.deepcopy(contract)
    source_recipe["release_source_overrides"] = source_overrides

    source_recipe_path = tmp_path / "source_recipe.json"
    source_recipe_path.write_text(
        json.dumps(source_recipe, ensure_ascii=False), encoding="utf-8"
    )
    override_recipe_path = tmp_path / "cache_overrides.json"
    override_recipe_path.write_text(
        json.dumps(override_recipe, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(cache_override, "_PINNED_CACHE_HASHES", cache_hashes)
    monkeypatch.setattr(cache_override, "_PINNED_SOURCE_OVERRIDES", source_overrides)

    def fake_filter(source_path, *, entry, recipe_digest, runtime):
        del recipe_digest, runtime
        with np.load(source_path, allow_pickle=True) as source:
            qpos = np.asarray(source["qpos"], dtype=np.float32)
            frequency = np.array(source["frequency"], copy=True)
        start = int(entry["source_cache_crop"]["start"])
        stop = entry["source_cache_crop"]["stop"]
        qpos = np.array(qpos[start:stop], copy=True)
        qpos[:, 7] += np.float32(0.001)
        return {
            "qpos": qpos,
            "qvel": np.zeros((len(qpos), 88), dtype=np.float32),
            "frequency": frequency,
        }

    monkeypatch.setattr(cache_override, "_build_filter_runtime", lambda: object())
    monkeypatch.setattr(cache_override, "_filter_cache_payload", fake_filter)
    ik_path = tmp_path / "smooth_ik.json"
    ik_path.write_text('{"mapping":"test"}\n', encoding="utf-8")
    return root, source_recipe_path, override_recipe_path, ik_path, cache_hashes


def _prepare_and_materialize(root, source_recipe_path, override_recipe_path):
    prepare.prepare_sources(root, source_recipe_path)
    for motion in cache_override.PINNED_MOTIONS:
        analysis = root / f"muscle_trajectory/raw_smooth_v1/{motion}_analysis.npz"
        analysis.parent.mkdir(parents=True, exist_ok=True)
        analysis.write_bytes(b"stale")
    return cache_override.materialize_overrides(
        root,
        override_recipe_path,
        source_recipe_path=source_recipe_path,
        replace=True,
    )


def test_release_and_materializer_imports_do_not_mutate_gpu_environment(monkeypatch):
    names = ("JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(cache_override)
    importlib.reload(data_release)
    assert all(name not in os.environ for name in names)

    sentinels = {
        "JAX_PLATFORMS": "gpu-sentinel",
        "JAX_PLATFORM_NAME": "gpu-platform-sentinel",
        "CUDA_VISIBLE_DEVICES": "3,7",
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)
    importlib.reload(cache_override)
    importlib.reload(data_release)
    assert {name: os.environ.get(name) for name in sentinels} == sentinels


def test_production_recipe_pins_six_cache_fallbacks_and_two_source_overrides():
    recipe = cache_override.load_override_recipe(PRODUCTION_OVERRIDE_RECIPE)
    assert [row["motion"] for row in recipe["overrides"]] == list(
        cache_override.PINNED_MOTIONS
    )
    assert all(row["filter"] == cache_override.PINNED_FILTER for row in recipe["overrides"])
    by_motion = {row["motion"]: row for row in recipe["overrides"]}
    assert by_motion["6月2日-3"]["source_cache_crop"] == {"start": 15, "stop": None}
    assert by_motion["6月2日-3"]["source_output_override"]["source_crop"] == {
        "start": 9,
        "stop": None,
    }
    assert by_motion["6月2日-5"]["source_cache_crop"] == {"start": 0, "stop": None}
    assert (
        by_motion["6月2日-5"]["source_output_override"]["operation"]
        == "deterministic_full_source_restore"
    )


def test_materializer_is_deterministic_restores_sources_and_protects_namespaces(
    tmp_path, monkeypatch
):
    root, source_recipe, override_recipe, _ik, cache_hashes = _build_fixture(
        tmp_path, monkeypatch
    )
    protected_sentinels = {}
    for variant in ("raw", "optimized", "initial"):
        sentinel = root / f"muscle_trajectory/{variant}/sentinel.bin"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(variant.encode())
        protected_sentinels[variant] = prepare.sha256_file(sentinel)

    result = _prepare_and_materialize(root, source_recipe, override_recipe)
    assert result["override_count"] == 6
    assert result["source_override_count"] == 2
    assert result["protected_input_hashes_unchanged"] is True
    assert all(row["removed_stale_analysis"] is True for row in result["motions"])
    assert all(
        not (root / f"muscle_trajectory/raw_smooth_v1/{motion}_analysis.npz").exists()
        for motion in cache_override.PINNED_MOTIONS
    )
    with np.load(root / "temp/raw_smooth_v1/6月2日-3.npz") as source:
        assert source["poses"].shape[0] == 123
    with np.load(root / "temp/raw_smooth_v1/6月2日-5.npz") as source:
        assert source["poses"].shape[0] == 128
    expected_cache_frames = {
        "6月2日-1": 183,
        "6月2日-2": 195,
        "6月2日-3": 202,
        "6月2日-4": 185,
        "6月2日-5": 210,
        "6月2日-7": 195,
    }
    first_hashes = {}
    first_sidecars = {}
    for motion, frames in expected_cache_frames.items():
        output = root / f"muscle_trajectory/raw_smooth_v1/{motion}.npz"
        with np.load(output) as cache:
            assert cache["qpos"].shape == (frames, 89)
            assert cache["qvel"].shape == (frames, 88)
        assert prepare.sha256_file(root / f"muscle_trajectory/raw/{motion}.npz") == cache_hashes[motion]
        first_hashes[motion] = prepare.sha256_file(output)
        first_sidecars[motion] = prepare.sha256_file(
            root / f"muscle_trajectory/raw_smooth_v1/{motion}.cache_provenance.json"
        )
    repeated = cache_override.materialize_overrides(
        root,
        override_recipe,
        source_recipe_path=source_recipe,
        replace=True,
    )
    assert all(row["removed_stale_analysis"] is False for row in repeated["motions"])
    assert first_hashes == {
        motion: prepare.sha256_file(root / f"muscle_trajectory/raw_smooth_v1/{motion}.npz")
        for motion in expected_cache_frames
    }
    assert first_sidecars == {
        motion: prepare.sha256_file(
            root / f"muscle_trajectory/raw_smooth_v1/{motion}.cache_provenance.json"
        )
        for motion in expected_cache_frames
    }
    for variant, digest in protected_sentinels.items():
        assert prepare.sha256_file(
            root / f"muscle_trajectory/{variant}/sentinel.bin"
        ) == digest


def test_materializer_rejects_recipe_path_escape_and_protected_hash_change(
    tmp_path, monkeypatch
):
    root, source_recipe, override_recipe, _ik, _hashes = _build_fixture(
        tmp_path, monkeypatch
    )
    wrong = json.loads(override_recipe.read_text(encoding="utf-8"))
    wrong["overrides"][0]["source_cache_path"] = "muscle_trajectory/optimized/6月2日-1.npz"
    wrong_path = tmp_path / "wrong.json"
    wrong_path.write_text(json.dumps(wrong, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source_cache_path must be pinned"):
        cache_override.load_override_recipe(wrong_path)

    protected = root / "muscle_trajectory/raw/6月2日-2.npz"
    with protected.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="protected fallback cache hash mismatch"):
        cache_override.materialize_overrides(
            root,
            override_recipe,
            source_recipe_path=source_recipe,
            dry_run=True,
        )


def test_release_manifest_truthfully_binds_21_default_and_6_overrides(
    tmp_path, monkeypatch
):
    root, source_recipe, override_recipe, ik_path, _hashes = _build_fixture(
        tmp_path, monkeypatch
    )
    _prepare_and_materialize(root, source_recipe, override_recipe)
    _write_split_manifests(root)
    source_dir = root / "temp/raw_smooth_v1"
    overridden = set(cache_override.PINNED_MOTIONS)
    for motion in (*TRAIN_MOTIONS, *VAL_MOTIONS):
        if motion in overridden:
            continue
        with np.load(source_dir / f"{motion}.npz") as source:
            source_frames = int(source["poses"].shape[0])
        _write_cache(
            root / f"muscle_trajectory/raw_smooth_v1/{motion}.npz",
            round(source_frames * 100.0 / 60.0),
        )

    manifest = data_release.build_release_manifest(
        root,
        recipe_path=source_recipe,
        ik_config_path=ik_path,
        cache_override_recipe_path=override_recipe,
    )
    assert manifest["schema_version"] == "forehand_clear_raw_smooth_release_v3"
    assert manifest["split"] == {
        "train_count": 22,
        "validation_count": 5,
        "overlap_count": 0,
    }
    smooth = manifest["default_smooth_retarget_contract"]
    assert smooth["motion_count"] == 21
    assert smooth["excluded_motions"] == sorted(overridden)
    assert smooth["frequency_transition"] == "60->100"
    assert smooth["ik_config_sha256"] == hashlib.sha256(ik_path.read_bytes()).hexdigest()
    contract = manifest["cache_override_contract"]
    assert contract["motion_count"] == 6
    assert contract["motions"] == sorted(overridden)
    generations = {row["motion"]: row for row in manifest["motions"]}
    assert all(
        generations[motion]["cache_generation"]["mode"]
        == "filtered_protected_raw_cache_fallback"
        for motion in overridden
    )
    assert sum(
        row["cache_generation"]["mode"] == "default_smooth_retarget"
        for row in manifest["motions"]
    ) == 21
    assert generations["6月2日-3"]["source"]["frames"] == 123
    assert generations["6月2日-3"]["cache"]["frames"] == 202
    assert generations["6月2日-5"]["source"]["frames"] == 128
    assert generations["6月2日-5"]["cache"]["frames"] == 210
    assert (
        generations["6月2日-5"]["source_generation"]["final_operation"]["type"]
        == "restore_full_original"
    )

    release_path = root / "manifests/raw_smooth_v1/release_manifest.json"
    data_release.write_release_manifest(release_path, manifest)
    validation = data_release.validate_release_manifest(
        root,
        release_path,
        recipe_path=source_recipe,
        ik_config_path=ik_path,
        cache_override_recipe_path=override_recipe,
    )
    assert validation["schema_version"].endswith("_v3")
    assert validation["passed"] is True
    assert validation["release_sha256"] == manifest["release_sha256"]
    with (source_dir / "video10.npz").open("ab") as handle:
        handle.write(b"tamper")
    failed = data_release.validate_release_manifest(
        root,
        release_path,
        recipe_path=source_recipe,
        ik_config_path=ik_path,
        cache_override_recipe_path=override_recipe,
    )
    assert failed["passed"] is False
