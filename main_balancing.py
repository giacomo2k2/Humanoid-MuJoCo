"""Balance test: run a controller on the Unitree G1 in MuJoCo and save logs / plots / video.

Examples (from the project root):
    python main_balancing.py                                   # QP, static reference, viewer (Space = start/pause)
    python main_balancing.py --controller QP --ref forward
    python main_balancing.py --controller PD --feedforward gravity
    MUJOCO_GL=egl python main_balancing.py --headless --duration 5      # no window

Results go to results/<name>/ : logs.csv, plots (PNG), video.mp4
"""
import argparse

import mediapy as media
import numpy as np

from controllers.PD import PDController
from controllers.QP import QPController
from utils.logger import Logger
from utils.plotter import save_plots
from utils.robot import PROJECT_ROOT, Robot
from utils.viewer import VideoRecorder, run

# COM reference for the QP: offset [m] added to the initial COM, as a function of time
COM_REFERENCES = {
    'static':  lambda t: np.zeros(3),
    'forward': lambda t: np.array([0.03 * np.sin(np.pi * t), 0.0, 0.0]),
    'lateral': lambda t: np.array([0.0, 0.05 * np.sin(2 * np.pi * t), 0.0]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--controller', choices=['QP', 'PD'], default='QP')
    p.add_argument('--ref', choices=list(COM_REFERENCES), default='static', help='QP: COM reference')
    p.add_argument('--feedforward', choices=PDController.FEEDFORWARD, default='contact', help='PD: feedforward')
    p.add_argument('--keyframe', default='stand_bent')
    p.add_argument('--name', help='experiment name (default: derived from the options)')
    p.add_argument('--headless', action='store_true', help='no viewer window (needs --duration)')
    p.add_argument('--duration', type=float, help='stop after this many simulated seconds')
    p.add_argument('--no-save', action='store_true', help="don't write logs / plots / video")
    p.add_argument('--no-video', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    robot = Robot(keyframe=args.keyframe)
    if args.controller == 'QP':
        controller = QPController(robot, com_offset=COM_REFERENCES[args.ref])
        name = args.name or f'QP_{args.ref}'
    else:
        controller = PDController(robot, feedforward=args.feedforward)
        name = args.name or f'PD_{args.feedforward}'

    save = not args.no_save
    logger = Logger(robot)
    recorder = VideoRecorder(robot.m) if save and not args.no_video else None

    try:
        run(robot, controller, logger=logger, recorder=recorder,
            headless=args.headless, duration=args.duration)
    finally:
        if recorder is not None:
            recorder.close()

    if not save or len(logger) == 0:
        return

    out_dir = PROJECT_ROOT / 'results' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.save(out_dir / 'logs.csv')
    print('saved', out_dir / 'logs.csv')
    for path in save_plots(out_dir / 'logs.csv'):
        print('saved', path)
    if recorder is not None and recorder.frames:
        media.write_video(out_dir / 'video.mp4', recorder.frames, fps=recorder.fps)
        print('saved', out_dir / 'video.mp4')


if __name__ == '__main__':
    main()
