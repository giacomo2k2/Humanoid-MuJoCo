"""Whole-body QP controller (proxsuite).

decision variables   x = [q_dd (nv=35) | tau (nu=29) | f (24)]  -> 88

equality constraints  A x = b
  1) M q_dd + h = S^T tau + J_c^T f                      dynamics (35)
  2) J_feet q_dd + Jdot_feet q_d = -kd_foot v_feet       feet stuck to the ground (2 x 6D = 12)

inequality constraints  l <= C x <= u
  3) f_z >= f_min                                        (8)
  4) |f_x| <= mu f_z,  |f_y| <= mu f_z                   linearized friction cone (32)
  5) tau_min <= tau <= tau_max                           (29)

cost  sum_t  w_t || A_t x - b_t ||^2                     (proxsuite form: 1/2 x^T H x + g^T x)
  6) COM position        A_t = [J_com 0 0]   b_t = Kp (c_ref - c) - Kd v_c - Jdot_com q_d
  7) pelvis orientation  A_t = [J_ori 0 0]   b_t = Kp e_R - Kd w - Jdot_ori q_d
  8) posture             A_t = [S 0 0]       b_t = Kp (q_ref - q) - Kd q_d_joints
  9) force regularization   ||f||^2
 10) torque regularization  ||tau||^2
"""
import mujoco
import numpy as np
import proxsuite

from controllers.base import ControlOutput

INF = 1e20


def _zero_offset(t):
    return np.zeros(3)


class QPController:
    def __init__(self, robot, com_offset=_zero_offset,
                 mu=0.42, f_min=5.0,
                 kp_com=100.0, kd_com=20.0, w_com=100.0,
                 kp_ori=100.0, kd_ori=20.0, w_ori=50.0,
                 kp_post=100.0, kd_post=20.0, w_post=1.0,
                 w_f=1e-4, w_tau=1e-5, kd_foot=10.0):
        self.robot = robot
        m, d = robot.m, robot.d
        self.com_offset = com_offset            # t -> 3D offset added to the initial COM

        # gains / weights
        self.kp_com, self.kd_com, self.w_com = kp_com, kd_com, w_com
        self.kp_ori, self.kd_ori, self.w_ori = kp_ori, kd_ori, w_ori
        self.kp_post, self.kd_post, self.w_post = kp_post, kd_post, w_post
        self.w_f, self.w_tau, self.kd_foot = w_f, w_tau, kd_foot

        # dimensions and variable blocks
        self.nv, self.nu = m.nv, m.nu
        self.nc = len(robot.spheres)
        self.nf = 3 * self.nc
        self.n_var = self.nv + self.nu + self.nf
        self.n_eq = self.nv + 12
        self.n_in = self.nc + 4 * self.nc + self.nu
        self.IQ = slice(0, self.nv)
        self.IT = slice(self.nv, self.nv + self.nu)
        self.IF = slice(self.nv + self.nu, self.n_var)
        self.S = np.hstack([np.zeros((self.nu, 6)), np.eye(self.nu)])   # tau acts on joint rows only

        # references: captured ONCE from the initial pose
        self.com0 = d.subtree_com[robot.pelvis].copy()
        self.quat_ref = d.xquat[robot.pelvis].copy()
        self.q_ref = robot.q_key

        self.C, self.l, self.u = self._inequalities(mu, f_min)

        self.qp = proxsuite.proxqp.dense.QP(self.n_var, self.n_eq, self.n_in)
        self.qp.settings.eps_abs = 1e-5
        self.qp.settings.eps_rel = 0.0
        self.qp.settings.max_iter = 1000
        self.qp.settings.verbose = False
        self.qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        self._initialized = False
        self._J_com_prev = None

    # ------------------------------------------------------------------ setup
    def _inequalities(self, mu, f_min):
        """Constant constraints (they don't depend on the state): build once."""
        r_ = self.robot
        C = np.zeros((self.n_in, self.n_var))
        l = np.zeros(self.n_in)
        u = np.zeros(self.n_in)
        f0 = self.nv + self.nu
        r = 0
        for k in range(self.nc):
            fx, fy, fz = f0 + 3 * k, f0 + 3 * k + 1, f0 + 3 * k + 2
            # 3) f_z >= f_min
            C[r, fz] = 1;                   l[r], u[r] = f_min, INF; r += 1
            # 4) f_x - mu f_z <= 0  and  f_x + mu f_z >= 0   (same for y)
            C[r, fx] = 1; C[r, fz] = -mu;   l[r], u[r] = -INF, 0;    r += 1
            C[r, fx] = 1; C[r, fz] = mu;    l[r], u[r] = 0, INF;     r += 1
            C[r, fy] = 1; C[r, fz] = -mu;   l[r], u[r] = -INF, 0;    r += 1
            C[r, fy] = 1; C[r, fz] = mu;    l[r], u[r] = 0, INF;     r += 1
        # 5) tau_min <= tau <= tau_max
        C[r:r + self.nu, self.IT] = np.eye(self.nu)
        l[r:r + self.nu], u[r:r + self.nu] = r_.ctrl_lo, r_.ctrl_hi
        
        r += self.nu
        assert r == self.n_in
        return C, l, u

    @staticmethod
    def _add_task(H, g, A_t, b_t, w):
        """Add w * ||A_t x - b_t||^2 to the cost."""
        H += 2 * w * A_t.T @ A_t
        g -= 2 * w * A_t.T @ b_t

    # ---------------------------------------------------------------- compute
    def compute(self, t):
        r, m, d = self.robot, self.robot.m, self.robot.d
        nv, IQ, IT, IF = self.nv, self.IQ, self.IT, self.IF
        qd = d.qvel
        dt = m.opt.timestep

        # --- equality constraints  A x = b ---
        M = r.mass_matrix()
        h = d.qfrc_bias.copy()
        J_c = r.contact_jacobian()
        J_feet, Jdot_qd_feet = r.feet_jacobians()

        A = np.zeros((self.n_eq, self.n_var))
        b = np.zeros(self.n_eq)
        A[:nv, IQ] = M                     # 1) [M  -S^T  -J_c^T] x = -h
        A[:nv, IT] = -self.S.T
        A[:nv, IF] = -J_c.T
        b[:nv] = -h
        A[nv:, IQ] = J_feet                # 2) J_feet q_dd = -Jdot q_d - kd v_feet
        b[nv:] = -Jdot_qd_feet - self.kd_foot * (J_feet @ qd)

        # --- cost ---
        H = np.zeros((self.n_var, self.n_var))
        g = np.zeros(self.n_var)

        # 6) COM position (Jdot q_d by finite difference of J_com)
        J_com = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, d, J_com, r.pelvis)
        Jdot_qd_com = np.zeros(3) if self._J_com_prev is None else (J_com - self._J_com_prev) / dt @ qd
        self._J_com_prev = J_com.copy()
        com_ref = self.com0 + self.com_offset(t)
        a_com = self.kp_com * (com_ref - d.subtree_com[r.pelvis]) - self.kd_com * (J_com @ qd)
        A_t = np.zeros((3, self.n_var)); A_t[:, IQ] = J_com
        self._add_task(H, g, A_t, a_com - Jdot_qd_com, self.w_com)

        # 7) pelvis orientation, world frame: e_R = rotation vector of q_ref * q^-1
        J_ori = np.zeros((3, nv)); Jdot_ori = np.zeros((3, nv))
        mujoco.mj_jacBody(m, d, None, J_ori, r.pelvis)
        mujoco.mj_jacDot(m, d, None, Jdot_ori, d.xpos[r.pelvis], r.pelvis)
        quat_conj = np.zeros(4); quat_err = np.zeros(4); e_R = np.zeros(3)
        mujoco.mju_negQuat(quat_conj, d.xquat[r.pelvis])
        mujoco.mju_mulQuat(quat_err, self.quat_ref, quat_conj)
        mujoco.mju_quat2Vel(e_R, quat_err, 1.0)
        a_ori = self.kp_ori * e_R - self.kd_ori * (J_ori @ qd)
        A_t = np.zeros((3, self.n_var)); A_t[:, IQ] = J_ori
        self._add_task(H, g, A_t, a_ori - Jdot_ori @ qd, self.w_ori)

        # 8) posture: S selects q_dd[6:] (the joints), Jdot = 0
        a_post = self.kp_post * (self.q_ref - d.qpos[7:]) - self.kd_post * qd[6:]
        A_t = np.zeros((self.nu, self.n_var)); A_t[:, IQ] = self.S
        self._add_task(H, g, A_t, a_post, self.w_post)

        # 9) force and 10) torque regularization
        H[IF, IF] += 2 * self.w_f * np.eye(self.nf)
        H[IT, IT] += 2 * self.w_tau * np.eye(self.nu)

        # --- solve ---
        if not self._initialized:
            self.qp.init(H, g, A, b, self.C, self.l, self.u)
            self._initialized = True
        else:
            self.qp.update(H=H, g=g, A=A, b=b)        # C, l, u are constant
        self.qp.solve()

        if self.qp.results.info.status != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
            print(f"t={d.time:.3f}  QP status: {self.qp.results.info.status}")

        x = self.qp.results.x
        return ControlOutput(tau=x[IT].copy(), f=x[IF].copy(), q_ref=self.q_ref)
