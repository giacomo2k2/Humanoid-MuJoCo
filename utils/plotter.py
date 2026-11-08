"""Plot a controller log (logs.csv written by main.py) and save PNGs next to it.

Usage:
    python -m utils.plotter results/<EXPERIMENT_NAME>/logs.csv
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')  # file output only; never opens a window (no clash with the MuJoCo viewer)
import matplotlib.pyplot as plt
import numpy as np

# Colors: categorical slots follow the entity (left foot is always blue, right always orange)
LEFT = '#2a78d6'
RIGHT = '#eb6834'
SERIES = '#2a78d6'
REFERENCE = '#8a8985'      # neutral gray for reference lines (m*g)
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=TEXT, fontsize=12, fontweight='bold')
    ax.set_xlabel('time [s]', color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def _label_end(ax, t, y, text):
    """Direct label at the end of a line, in text ink (not the series color)."""
    ax.annotate(text, (t[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=9, color=TEXT_2, annotation_clip=False)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return path


def load_log(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    return np.atleast_1d(data)


def save_plots(csv_path, out_dir=None):
    """Read logs.csv and write one PNG per metric into out_dir (default: the CSV's folder)."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)
    L = load_log(csv_path)
    t = L['time']
    saved = []

    # 1. RMS leg joint error
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, np.degrees(L['rms_leg_joint_err_rad']), color=SERIES, linewidth=2)
    _style(ax, 'RMS joint error, legs (12 joints)', 'RMS error [deg]')
    ax.set_ylim(bottom=0)
    saved.append(_save(fig, out_dir, 'rms_leg_joint_error.png'))

    # 2. Vertical force per foot: simulated (solid) vs commanded by pinv (dashed)
    #    one panel per foot (left and right are nearly equal and would hide each other);
    #    shared y so the two feet are directly comparable
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
    for ax, side, color in ((axes[0], 'left', LEFT), (axes[1], 'right', RIGHT)):
        ax.plot(t, L[f'fz_{side}_N'], color=color, linewidth=2, label='simulated')
        ax.plot(t, L[f'fz_{side}_cmd_N'], color=REFERENCE, linewidth=1.5, linestyle='--',
                label='commanded (pinv)')
        _style(ax, f'Vertical contact force, {side} foot', 'Fz [N]')
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc='upper right')
    axes[0].set_xlabel('')
    fig.tight_layout()
    saved.append(_save(fig, out_dir, 'foot_vertical_force.png'))

    # 3. Total vertical force vs robot weight
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, L['fz_total_N'], color=SERIES, linewidth=2, label='total Fz (simulated)')
    ax.plot(t, L['mg_N'], color=REFERENCE, linewidth=1.5, linestyle='--', label='m·g')
    _label_end(ax, t, L['mg_N'], f"m·g = {L['mg_N'][-1]:.1f} N")
    _style(ax, 'Total vertical force vs robot weight', 'force [N]')
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc='upper right')
    saved.append(_save(fig, out_dir, 'total_force_vs_mg.png'))

    # 4. Actuator saturation count
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(t, L['saturation_count'], where='post', color=SERIES, linewidth=2,
            zorder=3, clip_on=False)  # stays visible on top of the x axis when the count is 0
    _style(ax, 'Saturated actuators per step', 'count (of 29)')
    ax.set_ylim(bottom=0, top=max(1, L['saturation_count'].max()) * 1.15)
    ax.yaxis.get_major_locator().set_params(integer=True)
    saved.append(_save(fig, out_dir, 'saturation_count.png'))

    return saved


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for p in save_plots(sys.argv[1]):
        print('saved', p)
