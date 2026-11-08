from dataclasses import dataclass

import numpy as np


@dataclass
class ControlOutput:
    """What every controller returns from compute()."""
    tau: np.ndarray      # desired joint torques (29), NOT clipped: the simulation loop clips
    f: np.ndarray        # planned contact forces (3 per foot sphere, world frame)
    q_ref: np.ndarray    # joint reference used this step (29), for logging the tracking error
