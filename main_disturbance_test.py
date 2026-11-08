"""Visual test of a single push on the pelvis, in the MuJoCo viewer.

The push (utils/disturbance.py) starts at PUSH_START s and lasts PUSH_DURATION s. While it is
active a red arrow shows the force on the pelvis (length proportional to the force). The terminal
prints when the push starts / ends and the first time each failure detector triggers
(utils/detection.py); the simulation keeps running so you can watch what happens next.

Examples (from the project root):
    python main_disturbance_test.py                                  # QP, 10 N s forward
    python main_disturbance_test.py --impulse 14 --theta 90          # push to the left
    python main_disturbance_test.py --controller PD --impulse 12 --theta 180 --speed 0.25

theta: push direction in degrees, world frame (0 = +x = forward, 90 = +y = left).
Viewer: Space starts / pauses. --video saves what the viewer camera sees to results/<name>/video.mp4.
"""
import argparse
import copy

import mediapy as media
import mujoco
import numpy as np

from controllers.PD import PDController
from controllers.QP import QPController
from utils.detection import (body_contact, detect_fallen, detect_tilt, foot_tipping, slip)
from utils.disturbance import PUSH_DURATION, PUSH_START, push_force
from utils.robot import PROJECT_ROOT, Robot
from utils.viewer import VideoRecorder, run

ARROW_M_PER_N = 1 / 400          # arrow length scale: 400 N -> 1 m
ARROW_RGBA = np.array([0.89, 0.29, 0.28, 1.0], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--controller', choices=['QP', 'PD'], default='QP')
    p.add_argument('--impulse', type=float, default=10.0, help='push impulse [N s]')
    p.add_argument('--theta', type=float, default=0.0, help='push direction [deg], 0 = forward, 90 = left')
    p.add_argument('--speed', type=float, default=1.0, help='playback speed (0.25 = 4x slow motion)')
    p.add_argument('--duration', type=float, help='close the viewer after this many simulated seconds')
    p.add_argument('--keyframe', default='stand_bent')
    p.add_argument('--video', action='store_true', help='save the viewer camera to results/<name>/video.mp4')
    p.add_argument('--name', help='folder name for --video (default: derived from the options)')
    p.add_argument('--headless', action='store_true', help='no viewer window (needs --duration)')
    return p.parse_args()


class PushOverlay:
    """Draws the push arrow and reports push events / failures once each."""

    def __init__(self, robot, impulse, theta):
        self.robot = robot
        self.impulse, self.theta = impulse, theta
        self.d0 = copy.copy(robot.d)            # initial state, reference for the detectors
        self.reported = set()

    def _report_once(self, key, t, text):
        if key not in self.reported:
            self.reported.add(key)
            print(f't = {t:6.3f} s   {text}')

    def _failures(self, d):
        r, d0 = self.robot, self.d0
        checks = {'fallen (pelvis dropped > 15 cm)': detect_fallen(d, r, d0),
                  'tilt (pelvis > 30 deg)': detect_tilt(d, r, d0),
                  'body contact with the ground': body_contact(d, r),
                  'slip (foot > 2 cm or 5 deg yaw)': slip(d, r, d0)}
        for side, s in zip(('left', 'right'), r.feet_sites):
            checks[f'tipping, {side} foot (> 5 deg)'] = foot_tipping(d, r, r.m.site_bodyid[s])
        return [name for name, hit in checks.items() if hit]

    def __call__(self, scn, d):
        force = push_force(d.time, self.impulse, self.theta)
        scn.ngeom = 0
        if np.any(force):
            self._report_once('start', d.time, f'push start: {np.linalg.norm(force):.0f} N for {PUSH_DURATION} s'
                                               f' ({self.impulse:g} N s)')
            tip = d.xipos[self.robot.pelvis]                    # force acts at the pelvis COM
            tail = tip - force * ARROW_M_PER_N                  # arrow points into the body
            g = scn.geoms[0]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), ARROW_RGBA)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, tail, tip)
            scn.ngeom = 1
        elif d.time >= PUSH_START + PUSH_DURATION:
            self._report_once('end', d.time, 'push end')

        for name in self._failures(d):
            self._report_once(name, d.time, f'FAILURE: {name}')


def main():
    args = parse_args()
    theta = np.radians(args.theta)

    robot = Robot(keyframe=args.keyframe)
    if args.controller == 'QP':
        controller = QPController(robot)
    else:
        controller = PDController(robot, feedforward='contact')

    print(f'{args.controller}: push of {args.impulse:g} N s at {args.theta:g} deg, '
          f't = {PUSH_START}-{PUSH_START + PUSH_DURATION} s.  Space = start / pause.')
    recorder = VideoRecorder(robot.m) if args.video else None
    try:
        run(robot, controller, recorder=recorder, duration=args.duration,
            disturbance_magnitude=args.impulse, disturbance_theta=theta,
            overlay=PushOverlay(robot, args.impulse, theta), speed=args.speed, headless=args.headless)
    finally:
        if recorder is not None:
            recorder.close()

    if recorder is not None and recorder.frames:
        name = args.name or f'disturbance_test_{args.controller}_{args.impulse:g}Ns_{args.theta:g}deg'
        out_dir = PROJECT_ROOT / 'results' / name
        out_dir.mkdir(parents=True, exist_ok=True)
        media.write_video(out_dir / 'video.mp4', recorder.frames, fps=recorder.fps)
        print('saved', out_dir / 'video.mp4')


if __name__ == '__main__':
    main()
