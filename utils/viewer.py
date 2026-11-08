"""Simulation loop: interactive MuJoCo viewer or headless, with optional video recording."""
import time

import mujoco
import mujoco.viewer
import numpy as np

from utils.disturbance import apply_disturbance

SPACE = 32   # GLFW keycode


class VideoRecorder:
    def __init__(self, m, fps=30, width=1280, height=720):
        self.renderer = mujoco.Renderer(m, height=height, width=width)
        self.interval = 1.0 / fps
        self.fps = fps
        self.frames = []
        self._last = -np.inf

    def capture(self, d, camera):
        if d.time - self._last >= self.interval:
            self.renderer.update_scene(d, camera=camera)
            self.frames.append(self.renderer.render())
            self._last = d.time

    def close(self):
        self.renderer.close()


def _tracking_camera(m, body):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body
    cam.distance, cam.azimuth, cam.elevation = 2.5, 135.0, -15.0
    return cam


def run(robot, controller, logger=None, recorder=None,
        headless=False, duration=None, start_paused=True,
        disturbance_magnitude=0.0, disturbance_theta=0.0, stop=None,
        overlay=None, speed=1.0):
    """Simulate `controller` on `robot`.

    Interactive: Space starts / pauses; closes when the window is closed or `duration` is reached.
    Headless: runs `duration` seconds as fast as possible (no window).

    disturbance_magnitude: impulse [N s] of a push on the pelvis (see utils/disturbance.py),
                           in direction disturbance_theta [rad]. 0 = no push.
    stop: optional callable checked after every headless step; returning True ends the run early.
    overlay: optional callable overlay(scn, d), called every viewer frame (inside the lock)
             to draw extra geoms into scn = viewer.user_scn.
    speed: playback speed in the viewer (0.25 = 4x slow motion). Upper bound only: the loop
           can't run faster than the controller computes.
    """
    m, d = robot.m, robot.d

    def step():
        mujoco.mj_step1(m, d)
        out = controller.compute(d.time)
        d.ctrl[:] = np.clip(out.tau, robot.ctrl_lo, robot.ctrl_hi)
        if disturbance_magnitude:   # otherwise leave xfrc_applied alone (viewer Ctrl+drag pushes)
            apply_disturbance(d, robot, disturbance_magnitude, disturbance_theta)
        mujoco.mj_step2(m, d)
        if logger is not None:
            logger.log(out)

    if headless:
        if duration is None:
            raise ValueError("headless mode needs a duration")
        cam = _tracking_camera(m, robot.pelvis)
        while d.time < duration:
            step()
            if recorder is not None:
                recorder.capture(d, cam)
            if stop is not None and stop():
                break
        return

    state = {'running': not start_paused}

    def key_callback(keycode):
        if keycode == SPACE:
            state['running'] = not state['running']

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running() and (duration is None or d.time < duration):
            step_start = time.time()
            with viewer.lock():
                if state['running']:
                    step()
                    if recorder is not None:
                        recorder.capture(d, viewer.cam)
                else:
                    mujoco.mj_forward(m, d)       # keep the view in sync with manual qpos edits
                if overlay is not None:
                    overlay(viewer.user_scn, d)
            viewer.sync()

            remaining = m.opt.timestep / speed - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
