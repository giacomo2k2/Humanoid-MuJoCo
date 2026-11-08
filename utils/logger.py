"""Per-step metrics, saved to CSV."""
import mujoco
import numpy as np


class Logger:
    COLUMNS = ['time', 'rms_leg_joint_err_rad',
               'fz_left_cmd_N', 'fz_right_cmd_N',
               'fz_left_N', 'fz_right_N', 'fz_total_N', 'mg_N',
               'saturation_count']

    def __init__(self, robot):
        self.robot = robot
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def log(self, out):
        """Call after mj_step2, with the ControlOutput used for that step."""
        r, d = self.robot, self.robot.d
        leg_err = out.q_ref[:12] - d.qpos[7:19]                 # legs = first 12 joints
        fz_left_cmd, fz_right_cmd = r.split_fz(out.f)           # planned by the controller
        fz_left, fz_right = r.simulated_foot_fz()               # what MuJoCo actually applied
        saturated = (out.tau <= r.ctrl_lo + 1e-6) | (out.tau >= r.ctrl_hi - 1e-6)
        self.rows.append((d.time, np.sqrt(np.mean(leg_err ** 2)),
                          fz_left_cmd, fz_right_cmd,
                          fz_left, fz_right, fz_left + fz_right, r.mg,
                          int(saturated.sum())))

    def save(self, path):
        np.savetxt(path, np.array(self.rows), delimiter=',', fmt='%.6g',
                   header=','.join(self.COLUMNS), comments='')


class AngularMomentumLogger:
    """Whole-body angular momentum about the COM, world frame, every step.

    mj_subtreeVel fills d.subtree_angmom; the pelvis row is the root subtree = the whole robot,
    taken about its own COM (d.subtree_com[pelvis]). Same log(out) interface as Logger, so it
    can be passed to run(logger=...).
    """
    COLUMNS = ['time', 'Lx', 'Ly', 'Lz']

    def __init__(self, robot):
        self.robot = robot
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def log(self, out=None):
        m, d = self.robot.m, self.robot.d
        mujoco.mj_subtreeVel(m, d)                 # MuJoCo computes it only on request
        self.rows.append((d.time, *d.subtree_angmom[self.robot.pelvis]))

    def save(self, path):
        np.savetxt(path, np.array(self.rows), delimiter=',', fmt='%.6g',
                   header=','.join(self.COLUMNS), comments='')
