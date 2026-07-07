import numpy as np
import pytest

from musclemimic.badminton.scripts.extend_retarget_cache_with_stand_tail import build_stand_tail_qpos


def _sample_qpos(n_frames: int = 6, nq: int = 10) -> np.ndarray:
    qpos = np.zeros((n_frames, nq), dtype=np.float64)
    qpos[:, 0] = np.linspace(0.0, 0.5, n_frames)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.arange(n_frames, dtype=np.float64)[:, None]
    return qpos


def test_build_stand_tail_appends_settle_and_hold_frames() -> None:
    qpos = _sample_qpos()

    extended, settle_frames, hold_frames = build_stand_tail_qpos(
        qpos,
        frequency=10.0,
        hold_seconds=0.4,
        settle_seconds=0.2,
        anchor_window_seconds=0.3,
    )

    assert settle_frames == 2
    assert hold_frames == 4
    assert extended.shape[0] == qpos.shape[0] + settle_frames + hold_frames
    np.testing.assert_allclose(extended[: qpos.shape[0]], qpos)
    np.testing.assert_allclose(extended[-hold_frames:], np.repeat(extended[-1][None, :], hold_frames, axis=0))
    np.testing.assert_allclose(np.linalg.norm(extended[:, 3:7], axis=1), 1.0)


def test_build_stand_tail_uses_last_window_median_for_scalar_anchor() -> None:
    qpos = _sample_qpos(n_frames=5, nq=9)
    qpos[-3:, 7:] = np.array([[1.0, 7.0], [3.0, 9.0], [20.0, 40.0]])

    extended, _, hold_frames = build_stand_tail_qpos(
        qpos,
        frequency=10.0,
        hold_seconds=0.2,
        settle_seconds=0.0,
        anchor_window_seconds=0.3,
    )

    anchor = extended[-1]
    np.testing.assert_allclose(anchor[:7], qpos[-1, :7])
    np.testing.assert_allclose(anchor[7:], [3.0, 9.0])
    np.testing.assert_allclose(extended[-hold_frames:], np.repeat(anchor[None, :], hold_frames, axis=0))


def test_build_stand_tail_rejects_invalid_qpos_shape() -> None:
    with pytest.raises(ValueError, match="qpos"):
        build_stand_tail_qpos(
            np.zeros((1, 7)),
            frequency=10.0,
            hold_seconds=1.0,
            settle_seconds=0.5,
            anchor_window_seconds=0.25,
        )
