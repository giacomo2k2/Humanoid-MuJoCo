"""External push on the pelvis: a rectangular force pulse in the horizontal plane."""
import numpy as np

PUSH_START = 0.5      # [s] wait for the robot to settle first
PUSH_DURATION = 0.1   # [s]


def push_force(t, impulse, theta, start=PUSH_START, duration=PUSH_DURATION):
    """Force [N] at time t for a push of `impulse` [N s] in direction `theta`
    (rad, world frame, 0 = +x = forward, pi/2 = +y = left). Zero outside the pulse."""
    if start <= t < start + duration:
        return impulse / duration * np.array([np.cos(theta), np.sin(theta), 0.0])
    return np.zeros(3)


def apply_disturbance(d, robot, impulse, theta, start=PUSH_START, duration=PUSH_DURATION):
    """Write the push into xfrc_applied (world frame, acts at the pelvis COM). Call before mj_step2.

    xfrc_applied persists between steps, so this must run every step to also switch the force off.
    """
    d.xfrc_applied[robot.pelvis, 0:3] = push_force(d.time, impulse, theta, start, duration)
