"""Unitree G1 model wrapper: loading, initial keyframe, feet contact geometry and Jacobians.

Everything that depends on *this* robot model (body / site / geom names, foot spheres)
lives here, so the controllers only deal with matrices.
"""
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = PROJECT_ROOT / 'models' / 'unitree_g1' / 'scene.xml'


class Robot:
    def __init__(self, xml_path=SCENE_XML, keyframe='stand_bent'):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.key_id = self.m.key(keyframe).id

        m = self.m
        self.pelvis = m.body('pelvis').id
        self.feet_sites = [m.site('left_foot').id, m.site('right_foot').id]

        # contact spheres under each foot (4 per foot); this order defines the
        # order of the 24-row contact Jacobian and of the 24 contact-force variables
        self.left_spheres = self._foot_geoms('left_ankle_roll_link')
        self.right_spheres = self._foot_geoms('right_ankle_roll_link')
        self.spheres = self.left_spheres + self.right_spheres
        self.n_left = len(self.left_spheres)
        self._sphere_body = m.geom_bodyid[self.spheres]
        self._sphere_rad = m.geom_size[self.spheres, 0]

        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange.T.copy()
        self.mass = m.body_subtreemass[self.pelvis]
        self.mg = self.mass * np.linalg.norm(m.opt.gravity)       # robot weight [N]

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Initial condition from the <keyframe> in the XML."""
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key_id)
        mujoco.mj_forward(self.m, self.d)

    @property
    def q_key(self):
        """Joint angles of the initial keyframe (29), i.e. qpos[7:]."""
        return self.m.key_qpos[self.key_id, 7:].copy()

    def mass_matrix(self):
        M = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, self.d, M)
        return M

    # -------------------------------------------------------------- Jacobians
    def _foot_geoms(self, body):
        m = self.m
        bid = m.body(body).id
        return [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def contact_jacobian(self):
        """Point Jacobians at the bottom of every foot sphere: (3 * n_spheres) x nv, world frame."""
        m, d = self.m, self.d
        J = np.zeros((3 * len(self.spheres), m.nv))
        for k, g in enumerate(self.spheres):
            p = d.geom_xpos[g] - np.array([0.0, 0.0, self._sphere_rad[k]])
            mujoco.mj_jac(m, d, J[3*k:3*k+3], None, p, self._sphere_body[k])
        return J

    def feet_jacobians(self):
        """6D Jacobians of both foot sites [lin; ang] (12 x nv) and Jdot @ qvel (12)."""
        m, d = self.m, self.d
        J = np.zeros((12, m.nv))
        Jdot = np.zeros((12, m.nv))
        for i, s in enumerate(self.feet_sites):
            mujoco.mj_jacSite(m, d, J[6*i:6*i+3], J[6*i+3:6*i+6], s)
            mujoco.mj_jacDot(m, d, Jdot[6*i:6*i+3], Jdot[6*i+3:6*i+6],
                             d.site_xpos[s], m.site_bodyid[s])
        return J, Jdot @ d.qvel

    # --------------------------------------------------------- contact forces
    def split_fz(self, f):
        """Vertical force per foot from a stacked contact-force vector (3 per sphere)."""
        nl = 3 * self.n_left
        return f[2:nl:3].sum(), f[nl + 2::3].sum()

    def simulated_foot_fz(self):
        """Vertical contact force on each foot as simulated by MuJoCo, world frame [N].
        Valid after mj_step / mj_step2 (needs this step's constraint forces)."""
        m, d = self.m, self.d
        fz_left, fz_right = 0.0, 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom
            if g1 not in self.spheres and g2 not in self.spheres:
                continue
            mujoco.mj_contactForce(m, d, i, f6)               # contact frame, force on geom2
            fz = (c.frame.reshape(3, 3).T @ f6[:3])[2]        # -> world frame, z component
            foot = g2 if g2 in self.spheres else g1
            if foot == g1:
                fz = -fz                                      # foot is geom1 -> flip sign
            if foot in self.left_spheres:
                fz_left += fz
            else:
                fz_right += fz
        return fz_left, fz_right
