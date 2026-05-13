import mujoco

from musclemimic.runner.export_metadata import model_actuator_names


def test_model_actuator_names_reads_current_mujoco_actuator_order():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="body">
              <joint name="hinge" type="hinge" axis="0 0 1"/>
              <geom name="geom" type="capsule" size="0.01 0.1"/>
            </body>
          </worldbody>
          <actuator>
            <motor name="biceps_r" joint="hinge"/>
            <motor name="triceps_r" joint="hinge"/>
          </actuator>
        </mujoco>
        """
    )

    assert model_actuator_names(model) == ["biceps_r", "triceps_r"]
