import mujoco
import numpy as np
import pytest

from loco_mujoco.smpl.retargeting import _apply_gmr_ground_penetration_correction


def _free_sphere_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 0.1"/>
            <body>
              <freejoint/>
              <geom name="sphere" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _two_frame_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = np.repeat(model.qpos0[None, :], 2, axis=0)
    qpos[:, 2] = [0.05, 0.20]
    return qpos


def test_global_grounding_preserves_root_height_difference():
    model = _free_sphere_model()
    qpos = _two_frame_qpos(model)

    corrected, report = _apply_gmr_ground_penetration_correction(qpos, model, "global")

    np.testing.assert_allclose(np.diff(corrected[:, 2]), np.diff(qpos[:, 2]), atol=1e-9)
    np.testing.assert_allclose(corrected[:, 2], [0.10, 0.25], atol=1e-6)
    assert report["penetrating_frames_before"] == 1
    assert report["deepest_penetration_after_m"] >= -1e-9


def test_legacy_per_frame_grounding_only_raises_penetrating_frames():
    model = _free_sphere_model()
    corrected, report = _apply_gmr_ground_penetration_correction(_two_frame_qpos(model), model, "per_frame")

    np.testing.assert_allclose(corrected[:, 2], [0.10, 0.20], atol=1e-6)
    assert report["global_vertical_offset_m"] == 0.0


def test_grounding_rejects_unknown_mode():
    model = _free_sphere_model()
    with pytest.raises(ValueError, match="grounding_mode"):
        _apply_gmr_ground_penetration_correction(_two_frame_qpos(model), model, "unknown")
