from __future__ import annotations

import mujoco
import numpy as np

from musclemimic.viewer.viser_utils import build_body_meshes


def test_build_body_meshes_supports_current_trimesh_ellipsoids() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hand">
              <geom name="palm" type="ellipsoid" size="0.01 0.02 0.03"
                    contype="0" conaffinity="0" group="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    mesh = build_body_meshes(model, include_collision=False)[hand]

    np.testing.assert_allclose(mesh.extents, [0.02, 0.04, 0.06], atol=1.0e-9)
