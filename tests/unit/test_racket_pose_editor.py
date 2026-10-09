from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from environment.overall_environment.src.racket_attachment import (
    DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH,
    canonical_contract_fingerprint,
    load_racket_attachment_contract,
)
from musclemimic.badminton.scripts.racket_pose_editor import (
    build_adjusted_contract_document,
    load_trajectory_bytes,
    load_trajectory_path,
    normalize_wxyz,
    rotate_about_racket_local_axis,
    translate_position_m,
    validate_position_m,
    write_adjusted_contract,
)


def _apply_wxyz(quaternion: np.ndarray, vector: list[float]) -> np.ndarray:
    w, x, y, z = quaternion
    return Rotation.from_quat([x, y, z, w]).apply(vector)


def test_normalize_wxyz_is_unit_and_sign_canonical() -> None:
    actual = normalize_wxyz([-2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(actual, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="norm"):
        normalize_wxyz([0.0, 0.0, 0.0, 0.0])


def test_local_handle_axis_nudge_rotates_racket_face() -> None:
    adjusted = rotate_about_racket_local_axis([1.0, 0.0, 0.0, 0.0], 1, 90.0)
    np.testing.assert_allclose(
        _apply_wxyz(adjusted, [0.0, 0.0, 1.0]),
        [1.0, 0.0, 0.0],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        _apply_wxyz(adjusted, [0.0, 1.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1.0e-9,
    )


def test_translation_changes_one_coordinate_without_touching_orientation() -> None:
    position = np.array([-0.03, -0.09, 0.02])
    quaternion = normalize_wxyz([0.42, 0.49, 0.05, -0.76])
    quaternion_before = quaternion.copy()

    translated = translate_position_m(position, 1, 0.002)

    np.testing.assert_allclose(translated, [-0.03, -0.088, 0.02])
    np.testing.assert_array_equal(quaternion, quaternion_before)
    np.testing.assert_array_equal(position, [-0.03, -0.09, 0.02])
    with pytest.raises(ValueError, match="three finite"):
        validate_position_m([0.0, np.nan, 0.0])


def test_translated_contract_preserves_racket_orientation(tmp_path) -> None:
    source = load_racket_attachment_contract()
    translated_position = translate_position_m(source.relative_position_m, 2, 0.005)
    output = tmp_path / "translated_racket.json"

    adjusted = write_adjusted_contract(
        source,
        output,
        position_m=translated_position,
        quaternion_wxyz=source.relative_quaternion_wxyz,
        contract_id="translated_racket",
    )

    np.testing.assert_allclose(adjusted.relative_position_m, translated_position)
    source_rotation = Rotation.from_quat(
        np.asarray(source.relative_quaternion_wxyz)[[1, 2, 3, 0]]
    )
    adjusted_rotation = Rotation.from_quat(
        np.asarray(adjusted.relative_quaternion_wxyz)[[1, 2, 3, 0]]
    )
    assert (source_rotation.inv() * adjusted_rotation).magnitude() < 1.0e-8


def test_adjusted_contract_round_trips_with_new_fingerprint(tmp_path) -> None:
    source = load_racket_attachment_contract()
    quaternion = rotate_about_racket_local_axis(
        source.relative_quaternion_wxyz,
        1,
        7.5,
    )
    output = tmp_path / "forehand_clear_rigid_test.json"
    adjusted = write_adjusted_contract(
        source,
        output,
        position_m=source.relative_position_m,
        quaternion_wxyz=quaternion,
        contract_id="forehand_clear_rigid_test",
    )

    assert adjusted.contract_id == "forehand_clear_rigid_test"
    assert adjusted.fingerprint != source.fingerprint
    np.testing.assert_allclose(adjusted.relative_quaternion_wxyz, quaternion, atol=1.0e-8)
    assert load_racket_attachment_contract(output) == adjusted


def test_contract_document_fingerprint_is_canonical() -> None:
    source = load_racket_attachment_contract()
    document = build_adjusted_contract_document(
        source,
        position_m=source.relative_position_m,
        quaternion_wxyz=source.relative_quaternion_wxyz,
        contract_id="same_pose_new_id",
    )
    assert document["fingerprint"] == canonical_contract_fingerprint(document)


def test_writer_refuses_to_overwrite_source_contract() -> None:
    source = load_racket_attachment_contract()
    assert source.source_path == DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH.resolve()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_adjusted_contract(
            source,
            source.source_path,
            position_m=source.relative_position_m,
            quaternion_wxyz=source.relative_quaternion_wxyz,
            contract_id="forbidden",
        )


def test_trajectory_can_be_selected_by_path_or_browser_upload(tmp_path) -> None:
    path = tmp_path / "swing.npz"
    qpos = np.arange(18, dtype=np.float64).reshape(3, 6)
    np.savez(path, qpos=qpos, frequency=np.asarray(100.0))

    from_path = load_trajectory_path(path)
    from_upload = load_trajectory_bytes(path.read_bytes(), name="chosen_swing.npz")

    np.testing.assert_array_equal(from_path.qpos, qpos)
    np.testing.assert_array_equal(from_upload.qpos, qpos)
    assert from_path.frequency_hz == 100.0
    assert from_upload.name == "chosen_swing"
    assert from_path.sha256 == from_upload.sha256
