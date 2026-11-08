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
                label='commanded (controller)')
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


# fixed categorical order: the color follows the controller, never its position in the call
SERIES_COLORS = {'QP': '#2a78d6', 'PD': '#eb6834'}
FALLBACK_COLORS = ['#1baf7a', '#6250d6', '#e87ba4', '#008300']

# marker shape = failure mode just above the max impulse (color stays the controller)
FAILURE_MARKERS = {
    'tipping': '^',
    'slip': 's',
    'tilt': 'D',
    'fallen': 'v',
    'body_contact': 'X',
    'none (cap reached)': 'o',
}


def save_polar_plot(series, out_path, title='Max push impulse on the pelvis', bound=None):
    """Polar plot of max impulse vs push direction, top view with the robot facing up.

    series: {label: path to a sweep CSV (theta_deg, max_impulse_Ns, ...)}
    bound:  optional path to a bound CSV (theta_deg, J_bound_Ns), drawn as a dashed reference
    """
    import csv

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_theta_zero_location('N')      # 0 deg (+x, forward) at the top
    ax.set_theta_direction(1)            # counter-clockwise: +y (left) on the left

    from matplotlib.lines import Line2D

    extra = iter(FALLBACK_COLORS)
    controller_handles, modes_seen = [], []

    if bound is not None:                # reference first, so the measured data sits on top
        with open(bound) as f:
            rows = list(csv.DictReader(f))
        th = np.radians([float(r['theta_deg']) for r in rows])
        jb = np.array([float(r['J_bound_Ns']) for r in rows])
        ax.plot(np.append(th, th[0]), np.append(jb, jb[0]), color=REFERENCE, linewidth=1.5,
                linestyle='--', zorder=1)
        controller_handles.append(Line2D([], [], color=REFERENCE, linewidth=1.5, linestyle='--',
                                         label='capture-point bound (LIPM)'))

    for label, path in series.items():
        with open(path) as f:
            rows = list(csv.DictReader(f))
        th = np.radians([float(r['theta_deg']) for r in rows])
        imp = np.array([float(r['max_impulse_Ns']) for r in rows])
        modes = [r['failure_mode'] for r in rows]
        color = SERIES_COLORS.get(label) or next(extra)

        th_c, imp_c = np.append(th, th[0]), np.append(imp, imp[0])    # close the curve
        ax.plot(th_c, imp_c, color=color, linewidth=2, zorder=2)
        ax.fill(th_c, imp_c, color=color, alpha=0.08, linewidth=0)

        for mode in dict.fromkeys(modes):                              # unique, in order
            sel = np.array([m == mode for m in modes])
            ax.scatter(th[sel], imp[sel], marker=FAILURE_MARKERS.get(mode, 'P'), s=55,
                       color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)   # surface ring
            if mode not in modes_seen:
                modes_seen.append(mode)
        controller_handles.insert(len(controller_handles) - (bound is not None),
                                  Line2D([], [], color=color, linewidth=2, label=label))

    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ['0°\nforward', '45°', 'left 90°', '135°', '180°\nback', '225°', '270° right', '315°'],
                      color=TEXT_2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))   # few radial labels, less clutter
    ax.tick_params(axis='y', colors=TEXT_2, labelsize=8)
    ax.set_rlabel_position(157.5)        # radial labels between the left and back axes
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines['polar'].set_color(GRID)
    ax.set_title(f'{title}\n', loc='left', color=TEXT, fontsize=12, fontweight='bold')
    ax.text(0.5, -0.1, 'radius: max impulse [N·s]  ·  angle: push direction (world frame)',
            transform=ax.transAxes, ha='center', color=TEXT_2, fontsize=9)

    # legend 1: controllers (color); legend 2: failure mode just above the limit (shape, neutral ink)
    if len(series) > 1 or bound is not None:
        fig.legend(handles=controller_handles, frameon=False, fontsize=9, labelcolor=TEXT_2,
                   loc='upper right', bbox_to_anchor=(1.0, 1.0))
    mode_handles = [Line2D([], [], linestyle='none', marker=FAILURE_MARKERS.get(m, 'P'), markersize=7,
                           color=TEXT_2, label=m.replace('_', ' ')) for m in modes_seen]
    ax.legend(handles=mode_handles, title='failure mode above the limit', title_fontsize=9,
              frameon=False, fontsize=9, labelcolor=TEXT_2, loc='upper center',
              bbox_to_anchor=(0.5, -0.13), ncol=len(mode_handles))

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return out_path


AXIS_COLORS = {'Lx': '#2a78d6', 'Ly': '#eb6834', 'Lz': '#1baf7a'}


def save_angular_momentum_plot(csv_path, out_path, push_window=None, title=None):
    """Whole-body angular momentum about the COM (world frame) vs time.

    csv_path: CSV with time, Lx, Ly, Lz (AngularMomentumLogger)
    push_window: optional (t_start, t_end) shaded as the push
    """
    L = load_log(csv_path)
    t = L['time']
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if push_window is not None:
        ax.axvspan(*push_window, color=GRID, alpha=0.8, linewidth=0, zorder=0)
        ax.annotate('push', ((push_window[0] + push_window[1]) / 2, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -2), textcoords='offset points', ha='center', va='top',
                    fontsize=9, color=TEXT_2)
    ax.axhline(0, color=REFERENCE, linewidth=0.8, zorder=1)
    names = {'Lx': 'Lx (about x, forward)', 'Ly': 'Ly (about y, left)', 'Lz': 'Lz (about z, vertical)'}
    for key, label in names.items():
        ax.plot(t, L[key], color=AXIS_COLORS[key], linewidth=2, label=label, zorder=2)
    _style(ax, title or 'Angular momentum about the COM (world frame)', 'L [kg·m²/s]')
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc='upper right')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for p in save_plots(sys.argv[1]):
        print('saved', p)
