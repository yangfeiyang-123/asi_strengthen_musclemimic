"""Tests for bounded Orbax checkpoint I/O memory."""

from __future__ import annotations

import pytest

from musclemimic.algorithms.common.checkpoint_manager import OrbaxCheckpointManager


def test_orbax_checkpoint_io_uses_bounded_defaults(tmp_path):
    manager = OrbaxCheckpointManager(str(tmp_path / "checkpoints"), async_save=False)
    try:
        handler = manager._train_state_handler
        assert handler._impl._save_concurrent_bytes == 8_000_000_000
        assert handler._impl._restore_concurrent_bytes == 8_000_000_000
    finally:
        manager.close()


def test_orbax_checkpoint_io_limits_allow_positive_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB", "3")
    monkeypatch.setenv("MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB", "5")
    manager = OrbaxCheckpointManager(str(tmp_path / "checkpoints"), async_save=False)
    try:
        handler = manager._train_state_handler
        assert handler._impl._save_concurrent_bytes == 3_000_000_000
        assert handler._impl._restore_concurrent_bytes == 5_000_000_000
    finally:
        manager.close()


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_orbax_checkpoint_io_rejects_invalid_env_overrides(monkeypatch, tmp_path, value):
    monkeypatch.setenv("MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB", value)
    with pytest.raises(ValueError, match="must be a positive integer"):
        OrbaxCheckpointManager(str(tmp_path / "checkpoints"), async_save=False)
