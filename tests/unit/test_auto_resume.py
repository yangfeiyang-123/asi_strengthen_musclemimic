"""Unit tests for preemption-robust auto-resume functionality."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.badminton.promotion_artifact import checkpoint_identity
from musclemimic.runner.checkpointing import (
    bind_explicit_parent_checkpoint,
    build_parent_checkpoint_lineage,
    config_hash,
    configured_parent_checkpoint_lineage,
    find_latest_checkpoint,
    infer_training_action,
    resolve_checkpoint_dir,
    resolve_training_root,
    validate_checkpoint_body_action_contract,
    validate_checkpoint_compatibility,
    write_manifest,
)
from musclemimic.runner.engine import validate_auto_resume_config
from musclemimic.synergy.multistage_contract import (
    EXACT_RUNTIME_COMPATIBILITY,
    PORTABLE_COMPATIBILITY,
    BodySynergyContractV2,
    build_full_354_action_manifest,
)


class TestConfigHash:
    """Tests for config_hash function."""

    def test_same_config_produces_same_hash(self):
        """Same config should produce same hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "batch_size": 32})
        cfg2 = OmegaConf.create({"lr": 0.001, "batch_size": 32})
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_config_produces_different_hash(self):
        """Different config should produce different hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "batch_size": 32})
        cfg2 = OmegaConf.create({"lr": 0.002, "batch_size": 32})
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_excludes_volatile_fields(self):
        """Volatile fields like resume_from shouldn't affect hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "resume_from": None})
        cfg2 = OmegaConf.create({"lr": 0.001, "resume_from": "/path/to/ckpt"})
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_excludes_checkpoint_dir(self):
        """checkpoint_dir shouldn't affect hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "checkpoint_dir": "/path/a"})
        cfg2 = OmegaConf.create({"lr": 0.001, "checkpoint_dir": "/path/b"})
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_excludes_resume_lr_override_injected_from_checkpoint(self):
        """Restoring an exact LR must not change promotion/run identity."""
        original = OmegaConf.create({"lr": 0.001, "anneal_lr": True})
        resumed = OmegaConf.create(
            {
                "lr": 0.001,
                "anneal_lr": True,
                "resume_lr_override": 0.00019999999494757503,
            }
        )
        assert config_hash(original) == config_hash(resumed)

    def test_excludes_completed_run_extension_flag(self):
        """Extending an unchanged run must retain its immutable run identity."""
        original = OmegaConf.create({"lr": 0.001, "total_timesteps": 320})
        extended = OmegaConf.create(
            {
                "lr": 0.001,
                "total_timesteps": 320,
                "extend_completed_run": True,
            }
        )
        assert config_hash(original) == config_hash(extended)

    def test_excludes_auto_resume_fields(self):
        """auto_resume, run_id, checkpoint_root shouldn't affect hash."""
        cfg1 = OmegaConf.create(
            {
                "lr": 0.001,
                "auto_resume": True,
                "run_id": None,
                "checkpoint_root": None,
            }
        )
        cfg2 = OmegaConf.create(
            {
                "lr": 0.001,
                "auto_resume": False,
                "run_id": "my-run",
                "checkpoint_root": "/stable/root",
            }
        )
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_hash_is_12_chars(self):
        """Hash should be 12 characters."""
        cfg = OmegaConf.create({"lr": 0.001})
        h = config_hash(cfg)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_independence(self):
        """Key order shouldn't affect hash."""
        cfg1 = OmegaConf.create({"a": 1, "b": 2, "c": 3})
        cfg2 = OmegaConf.create({"c": 3, "b": 2, "a": 1})
        assert config_hash(cfg1) == config_hash(cfg2)


def _create_complete_checkpoint(checkpoint_dir: Path) -> None:
    """Create a checkpoint with metadata to mark it as complete."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Create Orbax metadata marker file
    (checkpoint_dir / "_CHECKPOINT_METADATA").touch()
    metadata_dir = checkpoint_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / "metadata").touch()


def _create_parent_checkpoint(root: Path, name: str, payload: bytes) -> Path:
    run = root / name
    checkpoint = run / "checkpoint_7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("complete", encoding="utf-8")
    (checkpoint / "weights.bin").write_bytes(payload)
    metadata = checkpoint / "metadata" / "metadata"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps(
            {
                "update_number": 7,
                "global_timestep": 700,
                "target_global_timestep": 1000,
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps({"config_hash": f"parent-{name}"}),
        encoding="utf-8",
    )
    return checkpoint


def _direct_body_contract(model_hash: str, *, reverse_names: bool = False) -> dict:
    names = tuple(f"muscle_{index:03d}" for index in range(354))
    if reverse_names:
        names = tuple(reversed(names))
    manifest = build_full_354_action_manifest(
        actuator_names=names,
        ctrlrange=np.asarray([[0.0, 1.0]] * 354, dtype=np.float64),
        runtime_model_hash=model_hash,
        source_binding={"kind": "test_direct_body_action"},
    )
    return BodySynergyContractV2.from_action_manifest(manifest).to_manifest()


def _create_body_checkpoint_leaf(
    checkpoint: Path,
    contract: dict | None,
    *,
    update_number: int,
) -> None:
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    (checkpoint / "train_state").mkdir()
    (checkpoint / "config").mkdir()
    experiment = {}
    if contract is not None:
        experiment["body_synergy_contract"] = contract
    (checkpoint / "config" / "metadata").write_text(
        json.dumps({"experiment": experiment}),
        encoding="utf-8",
    )
    (checkpoint / "metadata").mkdir()
    (checkpoint / "metadata" / "metadata").write_text(
        json.dumps({"update_number": update_number}),
        encoding="utf-8",
    )


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""

    def test_returns_none_for_nonexistent_dir(self):
        """Should return None if directory doesn't exist."""
        result = find_latest_checkpoint("/nonexistent/path")
        assert result is None

    def test_returns_none_for_empty_dir(self):
        """Should return None if directory has no checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(tmpdir)
            assert result is None

    def test_finds_single_checkpoint(self):
        """Should find single complete checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint_100"
            _create_complete_checkpoint(ckpt_path)
            result = find_latest_checkpoint(tmpdir)
            assert result == str(ckpt_path)

    def test_finds_latest_checkpoint(self):
        """Should find latest complete checkpoint by step number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [100, 500, 300, 200]:
                _create_complete_checkpoint(Path(tmpdir) / f"checkpoint_{step}")
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir) / "checkpoint_500")

    def test_ignores_incomplete_checkpoints(self):
        """Should ignore checkpoints without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create incomplete checkpoint (no metadata)
            incomplete = Path(tmpdir) / "checkpoint_500"
            incomplete.mkdir()
            # Create complete checkpoint with lower step
            _create_complete_checkpoint(Path(tmpdir) / "checkpoint_100")
            result = find_latest_checkpoint(tmpdir)
            # Should pick the complete one, not the higher incomplete one
            assert result == str(Path(tmpdir) / "checkpoint_100")

    def test_ignores_non_checkpoint_dirs(self):
        """Should ignore directories that don't match checkpoint_* pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_complete_checkpoint(Path(tmpdir) / "checkpoint_100")
            (Path(tmpdir) / "other_dir").mkdir()
            (Path(tmpdir) / "checkpoint_invalid").mkdir()  # No number
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir) / "checkpoint_100")


class TestWriteManifest:
    """Tests for write_manifest function."""

    def test_writes_manifest(self):
        """Should write manifest.json with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"lr": 0.001, "batch_size": 32})
            write_manifest(tmpdir, cfg, "abc123def456")

            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["config_hash"] == "abc123def456"
            assert "created_at" in manifest
            assert "experiment_config" in manifest
            assert manifest["experiment_config"]["lr"] == 0.001

    def test_writes_body_synergy_contract_at_top_level(self, tmp_path):
        contract = {
            "schema_version": "body_synergy_contract_v2",
            "mode": "full_354",
            "contract_fingerprint": "a" * 64,
        }
        cfg = OmegaConf.create(
            {
                "lr": 0.001,
                "body_synergy_contract": contract,
            }
        )

        write_manifest(tmp_path, cfg, "body-contract-run")

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["body_synergy_contract"] == contract
        assert manifest["experiment_config"]["body_synergy_contract"] == contract

    def test_writes_continuity_training_contract_at_top_level(self, tmp_path):
        contract = {
            "schema_version": "continuity_training_runtime_contract_v1",
            "release_fingerprint": "b" * 64,
            "loss_spec_fingerprint": "c" * 64,
            "binding_sha256": "d" * 64,
        }
        cfg = OmegaConf.create(
            {
                "lr": 0.001,
                "continuity_training_contract": contract,
            }
        )

        write_manifest(tmp_path, cfg, "continuity-contract-run")

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["continuity_training_contract"] == contract
        assert manifest["experiment_config"]["continuity_training_contract"] == contract

    def test_writes_continuity_smoke_contract_at_top_level(self, tmp_path):
        contract = {
            "schema_version": "continuity_smoke_runtime_contract_v1",
            "artifact_fingerprint": "e" * 64,
            "binding_sha256": "f" * 64,
        }
        cfg = OmegaConf.create(
            {
                "lr": 0.001,
                "continuity_smoke_contract": contract,
            }
        )

        write_manifest(tmp_path, cfg, "continuity-smoke-contract-run")

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["continuity_smoke_contract"] == contract
        assert manifest["experiment_config"]["continuity_smoke_contract"] == contract

    def test_existing_manifest_rejects_different_run_identity(self):
        """An empty fixed run directory cannot be rebound to a new config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = OmegaConf.create({"lr": 0.001})
            write_manifest(tmpdir, cfg1, "hash1")

            cfg2 = OmegaConf.create({"lr": 0.999})
            with pytest.raises(ValueError, match="different config hash"):
                write_manifest(tmpdir, cfg2, "hash2")

            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["config_hash"] == "hash1"
            assert manifest["experiment_config"]["lr"] == 0.001


class TestValidateCheckpointCompatibility:
    """Tests for validate_checkpoint_compatibility function."""

    def test_returns_true_for_no_manifest(self):
        """Should return True if no manifest exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_checkpoint_compatibility(tmpdir, "any_hash")
            assert result is True

    def test_returns_true_for_matching_hash(self):
        """Should return True if hash matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"lr": 0.001})
            write_manifest(tmpdir, cfg, "matching_hash")
            result = validate_checkpoint_compatibility(tmpdir, "matching_hash")
            assert result is True

    def test_returns_false_for_mismatched_hash(self, capsys):
        """Should return False and print warning if hash mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"lr": 0.001})
            write_manifest(tmpdir, cfg, "original_hash")
            result = validate_checkpoint_compatibility(tmpdir, "different_hash")
            assert result is False

            captured = capsys.readouterr()
            assert "WARNING: Config hash mismatch" in captured.out
            assert "original_hash" in captured.out
            assert "different_hash" in captured.out

    def test_strict_production_policy_fails_fast_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_manifest(tmpdir, OmegaConf.create({"lr": 0.001}), "original_hash")
            assert (
                validate_auto_resume_config(
                    tmpdir,
                    "different_hash",
                    strict=False,
                )
                is False
            )
            with pytest.raises(ValueError, match="strict training run"):
                validate_auto_resume_config(
                    tmpdir,
                    "different_hash",
                    strict=True,
                )

    def test_fixed_run_id_rejects_auto_resume_across_parent_content(self, tmp_path):
        parent_a = _create_parent_checkpoint(tmp_path, "parent-a", b"policy-a")
        parent_b = _create_parent_checkpoint(tmp_path, "parent-b", b"policy-b")

        def _config(parent: Path):
            return OmegaConf.create(
                {
                    "run_id": "fixed-stage2-run",
                    "auto_resume": True,
                    "strict_auto_resume_config_hash": True,
                    "resume_from": str(parent),
                    "parent_checkpoint_lineage": {
                        "required": True,
                        "role": "stage1_promoted",
                    },
                    "lr": 1.0e-4,
                }
            )

        config_a = _config(parent_a)
        lineage_a = bind_explicit_parent_checkpoint(config_a, launch_dir=tmp_path)
        hash_a = config_hash(config_a)
        local_run = tmp_path / "children" / "fixed-stage2-run"
        write_manifest(local_run, config_a, hash_a)

        assert lineage_a == configured_parent_checkpoint_lineage(config_a)
        manifest = json.loads((local_run / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parent_checkpoint_lineage"] == lineage_a
        assert len(manifest["run_identity"]["binding_sha256"]) == 64
        assert validate_auto_resume_config(
            local_run,
            hash_a,
            strict=True,
            expected_parent_lineage=lineage_a,
        )

        copied_run = tmp_path / "relocated" / "different-directory-name"
        shutil.copytree(parent_a.parent, copied_run)
        copied_config = _config(copied_run / parent_a.name)
        copied_lineage = bind_explicit_parent_checkpoint(
            copied_config,
            launch_dir=tmp_path,
        )
        assert copied_lineage == lineage_a

        config_b = _config(parent_b)
        lineage_b = bind_explicit_parent_checkpoint(config_b, launch_dir=tmp_path)
        hash_b = config_hash(config_b)
        assert config_a.run_id == config_b.run_id == "fixed-stage2-run"
        assert lineage_a != lineage_b
        assert hash_a != hash_b
        with pytest.raises(ValueError, match="parent checkpoint lineage mismatch"):
            validate_auto_resume_config(
                local_run,
                hash_b,
                strict=True,
                expected_parent_lineage=lineage_b,
            )
        # No checkpoint is needed for the manifest itself to remain immutable.
        with pytest.raises(ValueError, match="different config hash|different parent"):
            write_manifest(local_run, config_b, hash_b)

        child_checkpoint = local_run / "checkpoint_11"
        child_checkpoint.mkdir()
        (child_checkpoint / "_CHECKPOINT_METADATA").write_text(
            "complete",
            encoding="utf-8",
        )
        (child_checkpoint / "weights.bin").write_bytes(b"stage2-policy")
        child_metadata = child_checkpoint / "metadata" / "metadata"
        child_metadata.parent.mkdir()
        child_metadata.write_text(
            json.dumps(
                {
                    "update_number": 11,
                    "global_timestep": 1100,
                    "target_global_timestep": 2000,
                }
            ),
            encoding="utf-8",
        )
        extension = OmegaConf.create(
            {
                "run_id": "fixed-stage2-extension",
                "resume_from": str(child_checkpoint),
                "parent_checkpoint_lineage": {
                    "required": True,
                    "role": "stage2_080m_checkpoint",
                },
            }
        )
        extension_lineage = bind_explicit_parent_checkpoint(
            extension,
            launch_dir=tmp_path,
        )
        assert extension_lineage["parent_checkpoint_lineage"] == lineage_a

    def test_parent_bound_auto_resume_rejects_old_manifest_without_lineage(self, tmp_path):
        parent = _create_parent_checkpoint(tmp_path, "parent", b"policy")
        config = OmegaConf.create(
            {
                "run_id": "fixed-stage1r-run",
                "resume_from": str(parent),
                "parent_checkpoint_lineage": {
                    "required": True,
                    "role": "stage1_promoted",
                },
            }
        )
        lineage = bind_explicit_parent_checkpoint(config, launch_dir=tmp_path)
        current_hash = config_hash(config)
        old_run = tmp_path / "old-local-run"
        old_run.mkdir()
        (old_run / "manifest.json").write_text(
            json.dumps({"config_hash": current_hash}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="parent checkpoint lineage mismatch"):
            validate_auto_resume_config(
                old_run,
                current_hash,
                strict=True,
                expected_parent_lineage=lineage,
            )


class TestBodyActionCheckpointCompatibility:
    def test_same_run_requires_exact_runtime_contract(self, tmp_path):
        checkpoint = tmp_path / "run" / "checkpoint_1"
        saved = _direct_body_contract("a" * 64)
        _create_body_checkpoint_leaf(checkpoint, saved, update_number=1)
        (checkpoint.parent / "manifest.json").write_text(
            json.dumps(
                {
                    "body_synergy_contract": saved,
                    "experiment_config": {"body_synergy_contract": saved},
                }
            ),
            encoding="utf-8",
        )

        validate_checkpoint_body_action_contract(
            checkpoint,
            saved,
            compatibility=EXACT_RUNTIME_COMPATIBILITY,
        )
        rebound = _direct_body_contract("b" * 64)
        with pytest.raises(ValueError, match="exact runtime bindings"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                rebound,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
            )

    def test_cross_stage_parent_allows_only_portable_rebinding(self, tmp_path):
        checkpoint = tmp_path / "run" / "checkpoint_2"
        saved = _direct_body_contract("a" * 64)
        _create_body_checkpoint_leaf(checkpoint, saved, update_number=2)
        (checkpoint.parent / "manifest.json").write_text(
            json.dumps(
                {
                    "body_synergy_contract": saved,
                    "experiment_config": {"body_synergy_contract": saved},
                }
            ),
            encoding="utf-8",
        )

        validate_checkpoint_body_action_contract(
            checkpoint,
            _direct_body_contract("b" * 64),
            compatibility=PORTABLE_COMPATIBILITY,
        )
        with pytest.raises(ValueError, match="portable decoder cores"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                _direct_body_contract("b" * 64, reverse_names=True),
                compatibility=PORTABLE_COMPATIBILITY,
            )

    def test_symlinked_nonstandard_leaf_keeps_alias_run_binding(self, tmp_path):
        saved = _direct_body_contract("a" * 64)
        external_leaf = tmp_path / "exports" / "orbax_leaf"
        _create_body_checkpoint_leaf(
            external_leaf,
            saved,
            update_number=12,
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "body_synergy_contract": saved,
                    "experiment_config": {"body_synergy_contract": saved},
                }
            ),
            encoding="utf-8",
        )
        alias = run_dir / "checkpoint_12"
        alias.symlink_to(external_leaf, target_is_directory=True)

        validate_checkpoint_body_action_contract(
            alias,
            saved,
            compatibility=EXACT_RUNTIME_COMPATIBILITY,
        )
        validate_checkpoint_body_action_contract(
            run_dir,
            saved,
            compatibility=EXACT_RUNTIME_COMPATIBILITY,
        )

        rebound = _direct_body_contract("b" * 64)
        (external_leaf / "config" / "metadata").write_text(
            json.dumps({"experiment": {"body_synergy_contract": rebound}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="leaf body action contract differs"):
            validate_checkpoint_body_action_contract(
                alias,
                saved,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
            )

    def test_modern_contract_refuses_unbound_or_inconsistent_manifest(self, tmp_path):
        checkpoint = tmp_path / "run" / "checkpoint_3"
        checkpoint.mkdir(parents=True)
        current = _direct_body_contract("a" * 64)
        manifest_path = checkpoint.parent / "manifest.json"
        manifest_path.write_text(json.dumps({"experiment_config": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="shape-only restore"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                current,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
            )

        manifest_path.write_text(
            json.dumps(
                {
                    "body_synergy_contract": current,
                    "experiment_config": {"body_synergy_contract": _direct_body_contract("b" * 64)},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="differs from experiment config"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                current,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
            )

        manifest_path.write_text(
            '{"body_synergy_contract": {}, "body_synergy_contract": {}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate JSON key"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                current,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
            )

    def test_modern_manifest_rejects_old_leaf_without_body_contract(self, tmp_path):
        current = _direct_body_contract("a" * 64)
        checkpoint = tmp_path / "run" / "checkpoint_8"
        _create_body_checkpoint_leaf(checkpoint, None, update_number=8)
        (checkpoint.parent / "manifest.json").write_text(
            json.dumps(
                {
                    "body_synergy_contract": current,
                    "experiment_config": {"body_synergy_contract": current},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="leaf config has no body"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                current,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
            )

    def test_legacy_direct_parent_requires_content_bound_attestation(self, tmp_path):
        checkpoint = _create_parent_checkpoint(tmp_path, "legacy-direct", b"weights")
        contract = _direct_body_contract("a" * 64)
        lineage = build_parent_checkpoint_lineage(
            checkpoint_identity(checkpoint),
            role="stage1_repair_parent_640m",
        )
        attestation = tmp_path / "legacy-attestation.json"
        attestation.write_text(
            json.dumps(
                {
                    "body_synergy_contract": contract,
                    "experiment_config": {"body_synergy_contract": contract},
                    "parent_checkpoint_lineage": lineage,
                }
            ),
            encoding="utf-8",
        )

        validate_checkpoint_body_action_contract(
            checkpoint,
            contract,
            compatibility=PORTABLE_COMPATIBILITY,
            legacy_attestation=attestation,
        )
        with pytest.raises(ValueError, match="allowed only for an explicit portable parent"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                contract,
                compatibility=EXACT_RUNTIME_COMPATIBILITY,
                legacy_attestation=attestation,
            )

        (checkpoint / "weights.bin").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="parent checkpoint binding mismatch"):
            validate_checkpoint_body_action_contract(
                checkpoint,
                contract,
                compatibility=PORTABLE_COMPATIBILITY,
                legacy_attestation=attestation,
            )


class TestResolveCheckpointDir:
    """Tests for resolve_checkpoint_dir function."""

    def test_auto_resume_uses_launch_dir(self):
        """With auto_resume=true, relative paths resolve to launch_dir."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
        )
        assert result == "/project/checkpoints/abc123"

    def test_auto_resume_false_uses_result_dir(self):
        """With auto_resume=false, relative paths resolve to result_dir."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=False,
        )
        # Returns base without experiment_id; caller adds unique suffix
        assert result == "/outputs/2026-01-07/run1/checkpoints"

    def test_checkpoint_root_overrides(self):
        """checkpoint_root takes precedence over configured_ckpt_dir."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
            checkpoint_root="custom_ckpts",
        )
        assert result == "/project/custom_ckpts/abc123"

    def test_absolute_paths_unchanged(self):
        """Absolute paths are not modified."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="/absolute/checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
        )
        assert result == "/absolute/checkpoints/abc123"

    def test_absolute_checkpoint_root(self):
        """Absolute checkpoint_root is not modified."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
            checkpoint_root="/stable/root",
        )
        assert result == "/stable/root/abc123"

    def test_different_experiment_ids_different_dirs(self):
        """Different experiment_ids produce different directories."""
        result1 = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/run1",
            experiment_id="exp_a",
            auto_resume=True,
        )
        result2 = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/run1",
            experiment_id="exp_b",
            auto_resume=True,
        )
        assert result1 != result2
        assert result1 == "/project/checkpoints/exp_a"
        assert result2 == "/project/checkpoints/exp_b"

    def test_training_root_routes_checkpoint_under_action_training_dir(self):
        """Action training roots should own checkpoint artifacts."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/run1",
            experiment_id="exp_a",
            auto_resume=True,
            checkpoint_root="/legacy/checkpoints",
            training_root="/project/datasets/backhand_clear/training",
        )
        assert result == "/project/datasets/backhand_clear/training/checkpoints/exp_a"


class TestResolveTrainingRoot:
    """Tests for action-scoped training artifact roots."""

    def test_explicit_training_action_maps_to_dataset_training_dir(self):
        cfg = OmegaConf.create({"training_action": "backhand_clear"})
        assert resolve_training_root(cfg, "/repo") == "/repo/datasets/backhand_clear/training"

    def test_explicit_training_root_takes_precedence(self):
        cfg = OmegaConf.create(
            {
                "training_action": "backhand_clear",
                "training_root": "datasets/custom_action/training",
            }
        )
        assert resolve_training_root(cfg, "/repo") == "/repo/datasets/custom_action/training"

    def test_infers_action_from_dataset_paths(self):
        cfg = OmegaConf.create(
            {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "amass_root": "/repo/datasets/backhand_light/muscle_trajectory/amass_npz"
                        }
                    }
                }
            }
        )
        assert infer_training_action(cfg) == "backhand_light"
        assert resolve_training_root(cfg, "/repo") == "/repo/datasets/backhand_light/training"
