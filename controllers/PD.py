"""Joint-space PD controller with optional model-based feedforward."""
import numpy as np

from controllers.base import ControlOutput

# PD gains per actuated joint (same order as qvel[6:] / ctrl)
DEFAULT_KP = np.array([200, 200, 200,     # left hip pitch, roll, yaw
                       200, 200, 200,     # left knee, ankle pitch, roll
                       200, 200, 200,     # right hip pitch, roll, yaw
                       200, 200, 200,     # right knee, ankle pitch, roll
                       80, 80, 80,        # waist yaw, roll, pitch
                       30, 30, 30,        # left shoulder pitch, roll, yaw
                       30, 30, 30,        # left elbow, wrist roll, pitch
                       30,                # left wrist yaw
                       30, 30, 30,        # right shoulder pitch, roll, yaw
                       30, 30, 30,        # right elbow, wrist roll, pitch
                       30], dtype=float)  # right wrist yaw


class PDController:
    """tau = feedforward + Kp (q_ref - q) - Kd q_dot

    feedforward:
        'none'    -> pure PD
        'gravity' -> h_j                         (naive: ignores that the feet carry the weight)
        'contact' -> h_j - J_cj^T f,  f = pinv(J_cb^T) h_b   (contact forces balance the base)
    """
    FEEDFORWARD = ('none', 'gravity', 'contact')

    def __init__(self, robot, feedforward='contact', kp=DEFAULT_KP):
        if feedforward not in self.FEEDFORWARD:
            raise ValueError(f"feedforward must be one of {self.FEEDFORWARD}")
        self.robot = robot
        self.feedforward = feedforward
        self.kp = np.asarray(kp, dtype=float)
        M = robot.mass_matrix()
        self.kd = 2 * np.sqrt(self.kp * np.diag(M)[6:])   # critical damping per joint
        self.q_ref = robot.q_key

    def compute(self, t):
        r, d = self.robot, self.robot.d
        h = d.qfrc_bias                                   # Coriolis + gravity
        f = np.zeros(3 * len(r.spheres))

        if self.feedforward == 'none':
            tau_ff = np.zeros(r.m.nu)
        elif self.feedforward == 'gravity':
            tau_ff = h[6:].copy()
        else:
            J_c = r.contact_jacobian()
            f = np.linalg.pinv(J_c[:, :6].T) @ h[:6]      # base rows:  J_cb^T f = h_b
            tau_ff = h[6:] - J_c[:, 6:].T @ f             # joint rows: tau = h_j - J_cj^T f

        q_err = self.q_ref - d.qpos[7:]
        tau = tau_ff + self.kp * q_err - self.kd * d.qvel[6:]
        return ControlOutput(tau=tau, f=f, q_ref=self.q_ref)
