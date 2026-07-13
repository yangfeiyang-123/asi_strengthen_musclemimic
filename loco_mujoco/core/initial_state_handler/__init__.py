from .base import InitialStateHandler
from .default import DefaultInitialStateHandler
from .traj_init_state import TrajInitialStateHandler
from .racket_grip_init_state import RacketGripInitialStateHandler
from .finger_perturb_init_state import FingerPerturbInitialStateHandler

# register the initial state handlers
DefaultInitialStateHandler.register()
TrajInitialStateHandler.register()
RacketGripInitialStateHandler.register()
FingerPerturbInitialStateHandler.register()
