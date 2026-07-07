from .myofullbody import MyoFullBody, MjxMyoFullBody
from .myofullbody_racket import MyoFullBodyRacket, MjxMyoFullBodyRacket
from .bimanual import MyoBimanualArm, MjxMyoBimanualArm


# register muscle environments
MyoBimanualArm.register()
MjxMyoBimanualArm.register()
MyoFullBody.register()
MjxMyoFullBody.register()
MyoFullBodyRacket.register()
MjxMyoFullBodyRacket.register()
