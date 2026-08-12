from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from musclemimic.distill import stage2_direct_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[2]


def _source_manifest(path: Path, *, split: str, fingerprint: str) -> dict:
    return {
        "_source_path": str(path),
        "schema_version": "distill_dataset_manifest_v2",
        "manifest_fingerprint": fingerprint,
        "run_uid": f"shared-{split}",
        "teacher_checkpoint": {"sha256": "1" * 64},
        "teacher_promotion": {"binding_sha256": "2" * 64},
        "collections": [
            {
                "collection_id": f"teacher_{split}",
                "contract": {
                    "collection_id": f"teacher_{split}",
                    "split": split,
                    "motion_paths": [f"motion/{split}"],
                },
            }
        ],
        "shards": [],
    }


def test_family_plan_is_three_seed_componentized_and_canonical(tmp_path, monkeypatch):
    shared_path = tmp_path / "shared.json"
    shared_path.write_text("{}", encoding="utf-8")
    train_root = tmp_path / "shared_train"
    val_root = tmp_path / "shared_val"
    train_root.mkdir()
    val_root.mkdir()
    teacher = {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": "/teacher",
        "resolved_path": "/teacher",
        "sha256": "1" * 64,
        "num_files": 1,
        "num_bytes": 1,
        "files": [],
    }
    shared = {
        "binding_sha256": "3" * 64,
        "teacher": {"checkpoint": teacher},
    }
    sources = {
        "train": _source_manifest(train_root, split="train", fingerprint="4" * 64),
        "validation": _source_manifest(val_root, split="val", fingerprint="5" * 64),
    }
    split = {
        "train": {
            "collection_ids": ["teacher_train"],
            "motion_paths": ["motion/train"],
            "motion_uids": [1],
            "motion_set_fingerprint": "6" * 64,
        },
        "val": {
            "collection_ids": ["teacher_val"],
            "motion_paths": ["motion/val"],
            "motion_uids": [2],
            "motion_set_fingerprint": "7" * 64,
        },
    }
    monkeypatch.setattr(
        lifecycle,
        "_validate_shared_sources",
        lambda **_kwargs: (shared, sources, split),
    )
    monkeypatch.setattr(lifecycle, "checkpoint_content_fingerprint", lambda _path: teacher)
    monkeypatch.setattr(
        lifecycle,
        "validate_stage2_teacher_promotion",
        lambda *_args, **_kwargs: {
            "binding_sha256": "2" * 64,
        },
    )

    payload, steps = lifecycle.build_stage2_direct_family_plan(
        lifecycle.Stage2DirectFamilyConfig(
            action="forehand_clear",
            shared_inputs=str(shared_path),
            source_train_dataset_dir=str(train_root),
            source_val_dataset_dir=str(val_root),
            teacher_checkpoint="/teacher",
            teacher_promotion_manifest="/promotion.json",
            output_dir=str(tmp_path / "out"),
            physical_gpu=2,
            cache_key_prefix="s2a",
        )
    )

    assert payload["exact_seeds"] == [0, 1, 2]
    assert len(steps) == 19
    assert len({payload["per_seed"][str(seed)]["direct_dataset"] for seed in (0, 1, 2)}) == 3
    assert all("run_distill_experiment" not in " ".join(step.command) for step in steps)
    for seed in (0, 1, 2):
        bc = next(step for step in steps if step.name == f"s2a_seed{seed}_bc")
        dagger = next(step for step in steps if step.name == f"s2a_seed{seed}_dagger_3round")
        ppo = next(step for step in steps if step.name == f"s2a_seed{seed}_fresh_ppo")
        compare = next(step for step in steps if step.name == f"s2a_seed{seed}_heldout_compare")
        assert bc.command[:2] == (
            "scripts/run_fullbody_training.sh",
            "--distill-bc",
        )
        assert dagger.command[:2] == (
            "scripts/run_fullbody_training.sh",
            "--distill-dagger",
        )
        assert dagger.command[dagger.command.index("--num_iters") + 1] == "3"
        assert ppo.command[0] == "scripts/run_fullbody_training.sh"
        assert "experiment.auto_resume=false" in ppo.command
        assert "experiment.reset_optimizer_on_resume=true" in ppo.command
        assert compare.command[:2] == (
            "scripts/run_fullbody_training.sh",
            "--distill-compare",
        )
        assert compare.command[compare.command.index("--dataset_dir") + 1] == str(val_root.resolve())


def test_family_plan_uses_server_specific_jax_cache_root(tmp_path, monkeypatch):
    cache_root = tmp_path / "server-jax-cache"
    monkeypatch.setenv("MUSCLEMIMIC_JAX_CACHE_ROOT", str(cache_root))
    config = lifecycle.Stage2DirectFamilyConfig(
        action="forehand_clear",
        shared_inputs=str(tmp_path / "shared.json"),
        source_train_dataset_dir=str(tmp_path / "train"),
        source_val_dataset_dir=str(tmp_path / "val"),
        teacher_checkpoint="/teacher",
        teacher_promotion_manifest="/promotion.json",
        output_dir=str(tmp_path / "out"),
        physical_gpu=2,
        cache_key_prefix="portable-s2a",
    )

    environment = lifecycle._train_environment(
        config=config,
        seed=1,
        phase="bc",
        log_path=tmp_path / "bc.log",
    )

    assert environment["JAX_COMPILATION_CACHE_DIR"] == str(cache_root / "portable-s2a_forehand_clear_s1_bc")


def test_derive_direct_dataset_copies_train_only(tmp_path, monkeypatch):
    shared = tmp_path / "shared.json"
    shared.write_text("{}", encoding="utf-8")
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.mkdir()
    validation.mkdir()
    (train / "train_payload.bin").write_bytes(b"train")
    (validation / "val_payload.bin").write_bytes(b"validation")
    train_manifest = _source_manifest(train, split="train", fingerprint="4" * 64)
    val_manifest = _source_manifest(validation, split="val", fingerprint="5" * 64)
    split = {
        "train": {"motion_paths": ["motion/train"]},
        "val": {"motion_paths": ["motion/val"]},
    }
    monkeypatch.setattr(
        lifecycle,
        "_validate_shared_sources",
        lambda **_kwargs: (
            {"binding_sha256": "3" * 64},
            {"train": train_manifest, "validation": val_manifest},
            split,
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "validate_dataset_manifest",
        lambda path, **_kwargs: {
            "run_uid": "shared-train",
            "manifest_fingerprint": "4" * 64,
        },
    )

    output = tmp_path / "seed_0" / "direct_dataset"
    lifecycle.derive_direct_dataset(
        action="forehand_clear",
        seed=0,
        shared_inputs=shared,
        source_train_dataset_dir=train,
        source_val_dataset_dir=validation,
        output_dataset_dir=output,
    )

    assert (output / "train_payload.bin").read_bytes() == b"train"
    assert not (output / "val_payload.bin").exists()
    assert (validation / "val_payload.bin").read_bytes() == b"validation"


def _seed_record(seed: int, *, return_delta: float = 1.0) -> dict:
    return {
        "binding_sha256": f"{seed + 1}" * 64,
        "accepted": True,
        "shared_inputs": {"binding_sha256": "a" * 64},
        "contract_core": {"fixed": True},
        "contract_core_sha256": "b" * 64,
        "teacher": {
            "checkpoint": {"sha256": "c" * 64},
            "promotion": {"binding_sha256": "d" * 64},
        },
        "fresh_ppo": {"git_sha": "e" * 40},
        "dataset": {
            "source_train_manifest_fingerprint": "f" * 64,
            "source_validation_manifest_fingerprint": "0" * 64,
            "train_split": {"motion_set_fingerprint": "1" * 64},
            "heldout_split": {"motion_set_fingerprint": "2" * 64},
        },
        "checkpoints": {"ppo": {"sha256": f"{seed + 3}" * 64}},
        "dagger": {
            "improvement_vs_bc": {
                "mean_episode_return_delta": return_delta,
                "completion_rate_delta": 0.01,
                "early_termination_rate_reduction": 0.01,
                "err_rpos_reduction": 0.01,
                "err_racket_pos_reduction": 0.01,
                "err_racket_rot_reduction": 0.01,
            }
        },
        "heldout": {
            "failure_rates": {
                "bc_early_termination_rate": 0.10,
                "dagger_early_termination_rate": 0.09,
                "ppo_early_termination_rate": 0.05,
                "ppo_completion_failure_rate": 0.04,
            }
        },
    }


def test_family_promotion_is_seed_paired_and_rejects_no_dagger_improvement(tmp_path, monkeypatch):
    shared = tmp_path / "shared.json"
    shared.write_text("{}", encoding="utf-8")
    paths = {seed: tmp_path / f"seed_{seed}.json" for seed in (0, 1, 2)}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    records = {seed: _seed_record(seed) for seed in (0, 1, 2)}
    monkeypatch.setattr(
        lifecycle,
        "validate_stage2_direct_seed_evidence",
        lambda path, **_kwargs: records[int(Path(path).stem.rsplit("_", 1)[-1])],
    )

    promotion = lifecycle.build_stage2_direct_family_promotion(
        action="forehand_clear", shared_inputs=shared, seed_evidence=paths
    )
    assert promotion["schema_version"] == "stage2_direct_family_promotion_v1"
    assert promotion["selected_deployment_seed"] == 0
    assert promotion["paired_heldout_gate"]["n_seeds"] == 3
    assert promotion["paired_heldout_gate"]["checks"]["dagger_mean_return_improves_over_bc"]

    records.update({seed: _seed_record(seed, return_delta=-1.0) for seed in (0, 1, 2)})
    with pytest.raises(ValueError, match="paired held-out promotion gate failed"):
        lifecycle.build_stage2_direct_family_promotion(
            action="forehand_clear", shared_inputs=shared, seed_evidence=paths
        )


@pytest.mark.parametrize(
    ("mode", "module", "read_only"),
    (
        ("--distill-dagger", "fullbody.distill_run_dagger", False),
        ("--distill-compare", "fullbody.distill_compare", True),
    ),
)
def test_canonical_launcher_routes_s2a_modes_without_starting_training(
    tmp_path, mode: str, module: str, read_only: bool
):
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUSCLEMIMIC_JAX_CACHE_KEY": "s2a_launcher_test",
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax-cache"),
            "MUSCLEMIMIC_TRAIN_LOG": str(tmp_path / "launch.log"),
            "MUSCLEMIMIC_DRY_RUN": "1",
        }
    )
    completed = subprocess.run(
        [str(ROOT / "scripts/run_fullbody_training.sh"), mode, "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout + completed.stderr
    assert f"mode={mode.removeprefix('--')}" in output
    assert module in output
    assert "dry-run complete" in output
    assert "fullbody/experiment.py" not in output
    assert ("workload=read-only-evaluation" in output) is read_only
