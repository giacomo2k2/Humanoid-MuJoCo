"""Regenerate the README media in docs/media/ (headless, no window).

    MUJOCO_GL=egl python docs/make_media.py

Figures are built from results/ (run the balancing, single-push and the three sweep configurations
first, see the README). Each GIF clip is simulated with the same loop as the main scripts (utils.viewer.run), recorded with a
tracking camera; pushes are drawn as a red arrow on the pelvis while they are active.
"""
import contextlib
import io
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco                                   # noqa: E402
import numpy as np                              # noqa: E402
from PIL import Image, ImageDraw, ImageFont     # noqa: E402

from controllers.base import ControlOutput      # noqa: E402
from controllers.PD import PDController         # noqa: E402
from controllers.QP import QPController         # noqa: E402
from utils.disturbance import push_force        # noqa: E402
from utils.plotter import save_polar_plot       # noqa: E402
from utils.robot import Robot                   # noqa: E402
from utils.viewer import VideoRecorder, run     # noqa: E402

OUT = ROOT / 'docs' / 'media'
FPS = 15
WIDTH, HEIGHT = 480, 360
ARROW_M_PER_N = 1 / 400
ARROW_RGBA = np.array([0.89, 0.29, 0.28, 1.0], dtype=np.float32)


class ZeroTorque:
    """No control at all: the robot just falls."""
    def __init__(self, robot):
        self.robot = robot

    def compute(self, t):
        r = self.robot
        return ControlOutput(tau=np.zeros(r.m.nu), f=np.zeros(3 * len(r.spheres)), q_ref=r.q_key)


class Recorder(VideoRecorder):
    """VideoRecorder with a fixed tracking camera and the push arrow drawn into each frame."""

    def __init__(self, robot, azimuth, impulse=0.0, theta=0.0, distance=2.6):
        super().__init__(robot.m, fps=FPS, width=WIDTH, height=HEIGHT)
        self.robot, self.impulse, self.theta = robot, impulse, theta
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(robot.m, self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = robot.pelvis
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, -12.0

    def capture(self, d, camera=None):          # ignore run()'s camera, use ours
        if d.time - self._last < self.interval:
            return
        self.renderer.update_scene(d, camera=self.cam)
        force = push_force(d.time, self.impulse, self.theta)
        scn = self.renderer.scene
        if np.any(force) and scn.ngeom < scn.maxgeom:
            tip = d.xipos[self.robot.pelvis]
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), ARROW_RGBA)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, tip - force * ARROW_M_PER_N, tip)
            scn.ngeom += 1
        self.frames.append(self.renderer.render())
        self._last = d.time


def simulate(make_controller, duration, azimuth, impulse=0.0, theta_deg=0.0, com_offset=None, distance=2.6):
    robot = Robot()
    controller = make_controller(robot) if com_offset is None else make_controller(robot, com_offset)
    rec = Recorder(robot, azimuth, impulse, np.radians(theta_deg), distance)
    with contextlib.redirect_stdout(io.StringIO()):          # QP warnings while falling
        run(robot, controller, recorder=rec, headless=True, duration=duration,
            disturbance_magnitude=impulse, disturbance_theta=np.radians(theta_deg))
    rec.close()
    return rec.frames


def label(frames, text):
    font = ImageFont.load_default(size=22)
    out = []
    for f in frames:
        im = Image.fromarray(f)
        ImageDraw.Draw(im).text((14, 10), text, fill=(255, 255, 255), font=font,
                                stroke_width=2, stroke_fill=(20, 30, 50))
        out.append(np.asarray(im))
    return out


def side_by_side(a, b):
    n = min(len(a), len(b))
    return [np.concatenate([a[i], b[i]], axis=1) for i in range(n)]


def save_gif(frames, name):
    images = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.FASTOCTREE) for f in frames]  # keeps small red arrow
    path = OUT / name
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / FPS), loop=0,
                   optimize=True)
    print(f'saved {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)')


def make_figures():
    """README figures from the experiment outputs in results/ (run the main scripts first)."""
    res = ROOT / 'results'
    sweep = res / 'sweep_elliptic_fmin5'
    save_polar_plot({'QP': sweep / 'sweep_QP.csv', 'PD': sweep / 'sweep_PD.csv'}, OUT / 'push_polar.png',
                    bound=sweep / 'bound.csv')
    configs = {'pyramidal cone, f_min = 0': ('sweep_pyramidal_fmin0', '#8a8985'),
               'elliptic cone, f_min = 0': ('sweep_elliptic_fmin0', '#eb6834'),
               'elliptic cone, f_min = 5 N': ('sweep_elliptic_fmin5', '#2a78d6')}
    save_polar_plot({k: res / f / 'sweep_QP.csv' for k, (f, _) in configs.items()}, OUT / 'findings_qp_fmin.png',
                    title='QP: friction cone vs f_min', colors={k: c for k, (_, c) in configs.items()})
    for src, dst in [('test_disturbance_forward_push/angular_momentum.png', 'angmom_forward_qp.png'),
                     ('test_disturbance_lateral_push/angular_momentum.png', 'angmom_lateral_qp.png'),
                     ('QP_lateral/foot_vertical_force.png', 'qp_lateral_foot_force.png')]:
        shutil.copy(res / src, OUT / dst)
    print(f'saved figures in {OUT}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_figures()

    save_gif(label(simulate(ZeroTorque, 2.0, azimuth=150), 'no control'), 'fall_no_control.gif')

    lateral = lambda t: np.array([0.0, 0.05 * np.sin(2 * np.pi * t), 0.0])
    save_gif(label(simulate(QPController, 3.0, azimuth=180, com_offset=lateral), 'QP: COM sway, 5 cm @ 1 Hz'),
             'qp_com_lateral.gif')

    pd = lambda r: PDController(r, feedforward='contact')
    qp_push = label(simulate(QPController, 3.0, azimuth=90, impulse=9.5, theta_deg=0, distance=2.0), 'QP')
    pd_push = label(simulate(pd, 3.0, azimuth=90, impulse=9.5, theta_deg=0, distance=2.0), 'PD')
    save_gif(side_by_side(qp_push, pd_push), 'push_forward_qp_vs_pd.gif')

    save_gif(label(simulate(QPController, 3.0, azimuth=180, impulse=15.0, theta_deg=90, distance=2.2),
                   "QP: 15 N·s push toward the robot's left"), 'push_lateral_qp.gif')


if __name__ == '__main__':
    main()
