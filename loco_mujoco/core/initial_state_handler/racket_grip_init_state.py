from typing import Any, Tuple, Union
from types import ModuleType

import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils import assert_backend_is_supported
from loco_mujoco.core.initial_state_handler.traj_init_state import TrajInitialStateHandler


class RacketGripInitialStateHandler(TrajInitialStateHandler):
    """Trajectory init handler that also closes the right hand around the racket.

    The retargeted trajectories carry no finger joints, so :class:`TrajInitialStateHandler`
    leaves the model's finger DOF at zero (a flat, open hand). For racket-holding
    environments with the fingers enabled we override the right-hand finger qpos
    with the optimized forehand grip pose right after the trajectory state is set,
    so every episode starts already gripping the racket. Environments without
    finger joints (``disable_fingers=True``) are unaffected — the override index
    set is empty.
    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        assert_backend_is_supported(backend)

        data, carry = super().reset(env, model, data, carry, backend)

        addrs = getattr(env, "grip_finger_qpos_addrs", None)
        targets = getattr(env, "grip_finger_targets", None)
        if addrs is None or len(addrs) == 0:
            return data, carry

        if backend == np:
            data.qpos[addrs] = targets
        else:
            data = data.replace(qpos=data.qpos.at[backend.asarray(addrs)].set(backend.asarray(targets)))

        return data, carry
