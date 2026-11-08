"""Base of support and capture-point push bound (linear inverted pendulum model).

After an impulse J on the robot, the COM velocity jumps by J / M (M = whole-robot mass).
Starting at rest, the capture point  xi = c + c_dot / omega_0  then moves by J / (M omega_0)
in the push direction. Recovering without stepping requires xi to stay inside the support
polygon, i.e.

    J / (M omega_0) <= d(theta)   ->   v_max = omega_0 d(theta),   J_bound = M omega_0 d(theta)

with omega_0 = sqrt(g / z_c) and d(theta) the horizontal distance from the COM to the edge of
the support polygon along the push direction (ray cast). Instantaneous-impulse, point-mass,
ankle-strategy bound: upper-body motion can beat it, slip / foot tipping can fail before it.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


@dataclass
class SupportBound:
    thetas: np.ndarray      # push directions [rad], world frame (0 = +x forward, pi/2 = +y left)
    d: np.ndarray           # distance COM -> support polygon edge along each direction [m]
    v_max: np.ndarray       # omega_0 * d [m/s]
    J: np.ndarray           # M * v_max [N s]
    com: np.ndarray         # horizontal COM [m]
    polygon: np.ndarray     # support polygon vertices, counter-clockwise (k x 2) [m]
    omega_0: float          # [1/s]
    z_c: float              # COM height [m]


def support_polygon(robot):
    """Convex hull of the foot contact spheres projected on the floor (current state)."""
    points = robot.d.geom_xpos[robot.spheres, :2]     # sphere centres = contact points in xy
    return ConvexHull(points)


def distance_to_edge(hull, com, theta):
    """Distance from `com` (inside the hull) to the hull boundary along direction theta.

    hull.equations rows [n_x, n_y, off]: outward unit normal n, inside <=> n.x + off < 0.
    Ray com + t u hits edge k at t_k = -(n_k.com + off_k) / (n_k.u); only edges the ray moves
    toward (n_k.u > 0) count, and the first one hit (smallest t_k) is the boundary.
    """
    u = np.array([np.cos(theta), np.sin(theta)])
    n, off = hull.equations[:, :2], hull.equations[:, 2]
    n_u = n @ u
    ahead = n_u > 1e-12
    return np.min(-(n[ahead] @ com + off[ahead]) / n_u[ahead])


def compute_base_support(robot, thetas):
    """Capture-point bound for each push direction in `thetas`, from the robot's current state
    (call it on the initial state, e.g. right after Robot())."""
    hull = support_polygon(robot)
    com = robot.d.subtree_com[robot.pelvis, :2].copy()
    z_c = robot.d.subtree_com[robot.pelvis, 2]          # floor at z = 0
    n, off = hull.equations[:, :2], hull.equations[:, 2]
    if not np.all(n @ com + off < 0):
        raise ValueError('COM is outside the support polygon: no static balance to start from')

    g = -robot.m.opt.gravity[2]
    omega_0 = np.sqrt(g / z_c)
    thetas = np.asarray(thetas, dtype=float)
    d = np.array([distance_to_edge(hull, com, th) for th in thetas])
    v_max = omega_0 * d
    return SupportBound(thetas=thetas, d=d, v_max=v_max, J=robot.mass * v_max, com=com,
                        polygon=hull.points[hull.vertices].copy(), omega_0=omega_0, z_c=z_c)
