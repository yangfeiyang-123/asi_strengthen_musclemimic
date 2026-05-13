from __future__ import annotations

import mujoco


def model_actuator_names(model) -> list[str]:
    """Return actuator names in the exact MuJoCo control order."""
    names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(name or f"actuator_{i:03d}")
    return names
