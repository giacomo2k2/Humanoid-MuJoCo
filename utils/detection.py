"""Failure detectors for push experiments.

d0 is a snapshot (copy) of MjData at the initial state; detectors compare against it.
Call them after a step (positions / contacts of the current step are then available).
"""
import numpy as np

FALL_DROP = 0.15                 # [m]   pelvis height drop
TILT_MAX = np.radians(30)        # [rad] pelvis tilt
SLIP_XY = 0.02                   # [m]   foot horizontal displacement
SLIP_YAW = np.radians(5)         # [rad] foot yaw change
TIP_MAX = np.radians(5)          # [rad] foot roll / pitch (sole no longer flat)


def _z_axis(xmat):
    return xmat.reshape(3, 3)[:, 2]


def _angle(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def _yaw(xmat):
    R = xmat.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def detect_failures(d, robot, d0):
    """Name of the first failure detected, or None if the robot is fine."""
    if detect_fallen(d, robot, d0):
        return 'fallen'
    if detect_tilt(d, robot, d0):
        return 'tilt'
    if body_contact(d, robot):
        return 'body_contact'
    if slip(d, robot, d0):
        return 'slip'
    for s in robot.feet_sites:
        if foot_tipping(d, robot, robot.m.site_bodyid[s]):
            return 'tipping'
    return None


def detect_fallen(d, robot, d0):
    """Return True if the pelvis has dropped more than 0.15 m from its initial height."""
    return d0.xpos[robot.pelvis, 2] - d.xpos[robot.pelvis, 2] > FALL_DROP


def detect_tilt(d, robot, d0):
    """Return True if the pelvis has tilted more than 30° from its initial orientation."""
    return _angle(_z_axis(d.xmat[robot.pelvis]), _z_axis(d0.xmat[robot.pelvis])) > TILT_MAX


def body_contact(d, robot):
    """Return True if any robot geom other than the foot spheres touches the ground."""
    bodyid = robot.m.geom_bodyid
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom
        for ground, other in ((g1, g2), (g2, g1)):
            if bodyid[ground] == 0 and bodyid[other] != 0 and other not in robot.spheres:
                return True
    return False


def slip(d, robot, d0):
    """Return True if a foot has slipped more than 2 cm in xy or 5° in yaw."""
    for s in robot.feet_sites:
        if np.linalg.norm(d.site_xpos[s, :2] - d0.site_xpos[s, :2]) > SLIP_XY:
            return True
        dyaw = _yaw(d.site_xmat[s]) - _yaw(d0.site_xmat[s])
        if abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))) > SLIP_YAW:
            return True
    return False


def foot_tipping(d, robot, body_id):
    """Return True if the foot is tipping more than 5° of roll or pitch (sole tilted from flat)."""
    return _angle(_z_axis(d.xmat[body_id]), np.array([0.0, 0.0, 1.0])) > TIP_MAX
