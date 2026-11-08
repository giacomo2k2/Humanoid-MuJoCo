"""Push sweep: maximum impulse on the pelvis that the controller survives, per direction -> polar plot.

For each push direction theta (horizontal, world frame; 0 deg = +x = forward, 90 deg = +y = left)
a rectangular push of PUSH_DURATION s hits the pelvis at PUSH_START s. A trial fails as soon as any
detector in utils/detection.py triggers (fallen, tilt, body contact, foot slip, foot tipping).
The max impulse is found by doubling the impulse until the first failure, then bisection
(this assumes "fails" is monotonic in the impulse).

QP steps that are not solved are counted per trial (the controller then reuses its previous
command) and written to the CSV, so solver failures can't silently cause or hide a failure.

Examples (from the project root):
    python main_disturbance_sweep.py                                  # elliptic cone, f_min = 5 N
    python main_disturbance_sweep.py --cone pyramidal --f-min 0       # the original configuration
    python main_disturbance_sweep.py --controllers QP --directions 16 --tol 0.25

Results go to results/<name>/ (default name: sweep_<cone>_fmin<f_min>):
sweep_<controller>.csv, bound.csv, push_polar.png
"""
import argparse
import contextlib
import copy
import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from controllers.PD import PDController
from controllers.QP import QPController
from utils.compute_base_support import compute_base_support
from utils.detection import detect_failures
from utils.disturbance import PUSH_DURATION, PUSH_START
from utils.plotter import save_polar_plot
from utils.robot import CONES, PROJECT_ROOT, Robot
from utils.viewer import run

T_END = PUSH_START + 2.5          # observe 2.5 s after the push starts


def make_controller(name, robot, f_min):
    if name == 'QP':
        return QPController(robot, f_min=f_min)
    return PDController(robot, feedforward='contact')


def trial(controller_name, impulse, theta, cfg):
    """One fresh simulation from the keyframe.
    Returns (failure mode or None if it survived, number of QP steps not solved)."""
    robot = Robot(keyframe=cfg['keyframe'], cone=cfg['cone'])
    controller = make_controller(controller_name, robot, cfg['f_min'])
    d0 = copy.copy(robot.d)                       # initial state, reference for the detectors
    result = {'failure': None}

    def stop():
        result['failure'] = detect_failures(robot.d, robot, d0)
        return result['failure'] is not None

    with contextlib.redirect_stdout(io.StringIO()):   # QP warnings are counted below instead
        run(robot, controller, headless=True, duration=T_END,
            disturbance_magnitude=impulse, disturbance_theta=theta, stop=stop)
    return result['failure'], getattr(controller, 'n_solver_failures', 0)


def max_impulse(job):
    """Largest impulse survived in one direction: double until failure, then bisect.
    Also returns the unsolved-QP-step counts of the survived trial and of the failing one."""
    controller_name, theta, cfg = job
    safe, fail, mode = 0.0, None, None
    n_safe, n_fail = 0, 0

    impulse = cfg['start']
    while impulse <= cfg['cap']:
        failure, n = trial(controller_name, impulse, theta, cfg)
        if failure is None:
            safe, n_safe, impulse = impulse, n, 2 * impulse
        else:
            fail, mode, n_fail = impulse, failure, n
            break
    if fail is None:
        return theta, safe, 'none (cap reached)', n_safe, 0

    while fail - safe > cfg['tol']:
        mid = 0.5 * (safe + fail)
        failure, n = trial(controller_name, mid, theta, cfg)
        if failure is None:
            safe, n_safe = mid, n
        else:
            fail, mode, n_fail = mid, failure, n
    return theta, safe, mode, n_safe, n_fail


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--controllers', nargs='+', choices=['QP', 'PD'], default=['QP', 'PD'])
    p.add_argument('--cone', choices=list(CONES), help='friction cone of the simulation (default: scene.xml)')
    p.add_argument('--f-min', type=float, default=5.0, help='QP: minimum normal force per foot sphere [N]')
    p.add_argument('--directions', type=int, default=32, help='number of push directions over 360 deg')
    p.add_argument('--start', type=float, default=2.0, help='first impulse tried [N s]')
    p.add_argument('--tol', type=float, default=0.1, help='bisection tolerance [N s]')
    p.add_argument('--cap', type=float, default=200.0, help='stop doubling above this impulse [N s]')
    p.add_argument('--keyframe', default='stand_bent')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--name', help='results folder (default: sweep_<cone>_fmin<f_min>)')
    return p.parse_args()


def main():
    args = parse_args()
    cone = args.cone or Robot(keyframe=args.keyframe).cone
    cfg = dict(keyframe=args.keyframe, cone=cone, f_min=args.f_min,
               start=args.start, tol=args.tol, cap=args.cap)
    out_dir = PROJECT_ROOT / 'results' / (args.name or f'sweep_{cone}_fmin{args.f_min:g}')
    out_dir.mkdir(parents=True, exist_ok=True)
    thetas = np.linspace(0, 2 * np.pi, args.directions, endpoint=False)
    print(f'{cone} cone, QP f_min = {args.f_min:g} N; push: {PUSH_DURATION} s at t={PUSH_START} s, '
          f'observed until t={T_END} s')

    series = {}
    for name in args.controllers:
        baseline, _ = trial(name, 0.0, 0.0, cfg)
        if baseline is not None:
            print(f'{name}: fails without any push ({baseline}), skipping')
            continue

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(max_impulse, [(name, th, cfg) for th in thetas]))

        csv_path = out_dir / f'sweep_{name}.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['theta_deg', 'max_impulse_Ns', 'max_force_N', 'failure_mode',
                        'qp_unsolved_steps_survived', 'qp_unsolved_steps_failed'])
            for th, imp, mode, n_safe, n_fail in results:
                w.writerow([f'{np.degrees(th):.1f}', f'{imp:.3f}', f'{imp / PUSH_DURATION:.1f}', mode,
                            n_safe, n_fail])
        series[name] = csv_path

        print(f'\n{name}: max impulse per direction (failure mode just above it, unsolved QP steps)')
        for th, imp, mode, n_safe, n_fail in results:
            print(f'  {np.degrees(th):6.1f} deg  {imp:7.2f} N s  ({imp / PUSH_DURATION:7.1f} N)  {mode:12s}'
                  f'  unsolved: {n_safe} survived / {n_fail} failed')
        print('saved', csv_path)

    if series:
        bound_path = save_bound(args.keyframe, out_dir / 'bound.csv')
        path = save_polar_plot(series, out_dir / 'push_polar.png', bound=bound_path,
                               title=f'Max push impulse on the pelvis ({cone} cone, f_min = {args.f_min:g} N)')
        print('saved', path)


def save_bound(keyframe, csv_path, n=360):
    """Capture-point bound from the initial state, on a fine angle grid (smooth curve)."""
    b = compute_base_support(Robot(keyframe=keyframe), np.linspace(0, 2 * np.pi, n, endpoint=False))
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['theta_deg', 'd_m', 'v_max_mps', 'J_bound_Ns'])
        for th, d, v, J in zip(b.thetas, b.d, b.v_max, b.J):
            w.writerow([f'{np.degrees(th):.1f}', f'{d:.4f}', f'{v:.4f}', f'{J:.3f}'])
    print(f'\ncapture-point bound: z_c = {b.z_c:.3f} m, omega_0 = {b.omega_0:.3f} 1/s, '
          f'J = {b.J.min():.1f} - {b.J.max():.1f} N s')
    print('saved', csv_path)
    return csv_path


if __name__ == '__main__':
    main()
