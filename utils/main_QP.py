import os
import time
import mediapy as media
import mujoco
import mujoco.viewer

import numpy as np

import proxsuite

from utils.plot_logs import save_plots  # utils/plot_logs.py (same folder as this script)

SAVE = True
EXPERIMENT_NAME = "QP_control_forward_ref"


m = mujoco.MjModel.from_xml_path('./models/unitree_g1/scene.xml')
d = mujoco.MjData(m)

# Initial condition from the <keyframe> in the XML
mujoco.mj_resetDataKeyframe(m, d, m.key('stand_bent').id)
mujoco.mj_forward(m, d)  # compute body/geom poses from qpos so the viewer shows them

fps = 30
width, height = 1280, 720
renderer = mujoco.Renderer(m, height=height, width=width)

frames = []
sim_dt = m.opt.timestep
render_interval = 1 / fps
last_render_time = 0.0


logs = []

# State flag: toggle running when Space is pressed
is_running_sim = False


def foot_geoms(m, body):
    bid = m.body(body).id
    return [g for g in range(m.ngeom)
            if m.geom_bodyid[g] == bid and (m.geom_contype[g] or m.geom_conaffinity[g])]

LEFT_SPHERES = foot_geoms(m, 'left_ankle_roll_link')
RIGHT_SPHERES = foot_geoms(m, 'right_ankle_roll_link')
SPHERES = LEFT_SPHERES + RIGHT_SPHERES
BODY = m.geom_bodyid[SPHERES]      # body of each sphere
RAD  = m.geom_size[SPHERES, 0]     # sphere radii (0.005)
NL = len(LEFT_SPHERES)             # rows 0:3*NL of J_c / contact_forces are the left foot

MG = m.body_subtreemass[m.body('pelvis').id] * np.linalg.norm(m.opt.gravity)  # robot weight [N]


def foot_vertical_forces(m, d):
    """Vertical contact force on each foot as simulated by MuJoCo, world frame [N]."""
    fz_left, fz_right = 0.0, 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom
        if g1 not in SPHERES and g2 not in SPHERES:
            continue
        mujoco.mj_contactForce(m, d, i, f6)               # contact frame, force on geom2
        fz = (c.frame.reshape(3, 3).T @ f6[:3])[2]        # -> world frame, z component
        foot = g2 if g2 in SPHERES else g1
        if foot == g1:
            fz = -fz                                      # foot is geom1 -> flip sign
        if foot in LEFT_SPHERES:
            fz_left += fz
        else:
            fz_right += fz
    return fz_left, fz_right



# --- every step, after mj_forward / mj_step ---
def contact_jacobian(m, d):
    J = np.zeros((3 * len(SPHERES), m.nv))
    for k, g in enumerate(SPHERES):
        p = d.geom_xpos[g] - np.array([0, 0, RAD[k]])   # bottom of the sphere
        mujoco.mj_jac(m, d, J[3*k:3*k+3], None, p, BODY[k])
    return J

def key_callback(keycode):
  global is_running_sim
  # 32 is the GLFW keycode for Spacebar
  if keycode == 32:
    is_running_sim = not is_running_sim


KEY = m.key('stand_bent').id

# ---------------------------------------------------------------------------
# QP dimensions
#   decision variables x = [q_dd (35) | tau (29) | f (24)]  -> 88
# ---------------------------------------------------------------------------
NV = m.nv                       # 35
NU = m.nu                       # 29
NC = len(SPHERES)               # 8 contact points
NF = 3 * NC                     # 24
N_VAR = NV + NU + NF            # 88
N_EQ = NV + 12                  # 35 dynamics + 2 feet x 6D contact = 47
N_IN = NC + 4 * NC + NU         # 8 (f_z >= 0) + 32 (friction pyramid) + 29 (torque limits) = 69

IQ = slice(0, NV)               # q_dd
IT = slice(NV, NV + NU)         # tau
IF = slice(NV + NU, N_VAR)      # f

MU = 0.42
INF = 1e20
S = np.hstack([np.zeros((NU, 6)), np.eye(NU)])   # selection matrix: tau acts on joint rows only
lo, hi = m.actuator_ctrlrange.T

# --- task gains (accelerations, not torques) and weights ---
KP_COM, KD_COM, W_COM = 100.0, 20.0, 100.0
KP_ORI, KD_ORI, W_ORI = 100.0, 20.0, 50.0
KP_POST, KD_POST, W_POST = 100.0, 20.0, 1.0
W_F = 1e-4
W_TAU = 1e-5
KD_FOOT = 10.0                  # foot contact: J q_dd + Jdot q_d = -KD_FOOT * v_foot

# --- references (taken from the initial keyframe pose) ---
PELVIS = m.body('pelvis').id
FEET_SITES = [m.site('left_foot').id, m.site('right_foot').id]



# --- inequality constraints: constant, so build them once ---
C = np.zeros((N_IN, N_VAR))
l = np.zeros(N_IN)
u = np.zeros(N_IN)
r = 0
F_MIN = np.full(NC, .0)
for k in range(NC):
    fx, fy, fz = NV + NU + 3 * k, NV + NU + 3 * k + 1, NV + NU + 3 * k + 2
    # 3) f_z >= f_min
    C[r, fz] = 1;                      l[r], u[r] = F_MIN[k], INF;  r += 1
    # 4) |f_x| <= mu f_z  ->  f_x - mu f_z <= 0  and  f_x + mu f_z >= 0   (same for y)
    C[r, fx] = 1;  C[r, fz] = -MU;     l[r], u[r] = -INF, 0; r += 1
    C[r, fx] = 1;  C[r, fz] = MU;      l[r], u[r] = 0, INF;  r += 1
    C[r, fy] = 1;  C[r, fz] = -MU;     l[r], u[r] = -INF, 0; r += 1
    C[r, fy] = 1;  C[r, fz] = MU;      l[r], u[r] = 0, INF;  r += 1
# 5) tau_min <= tau <= tau_max
C[r:r + NU, IT] = np.eye(NU)
l[r:r + NU], u[r:r + NU] = lo, hi
r += NU
assert r == N_IN


def feet_jacobians(m, d):
    """6D Jacobians of both foot sites (12 x nv) and Jdot*q_d (12)."""
    J = np.zeros((12, m.nv))
    Jdot = np.zeros((12, m.nv))
    for i, s in enumerate(FEET_SITES):
        b, p = m.site_bodyid[s], d.site_xpos[s]
        mujoco.mj_jacSite(m, d, J[6*i:6*i+3], J[6*i+3:6*i+6], s)
        mujoco.mj_jacDot(m, d, Jdot[6*i:6*i+3], Jdot[6*i+3:6*i+6], p, b)
    return J, Jdot @ d.qvel


def add_task(H, g, A_t, b_t, w):
    """Add w * ||A_t x - b_t||^2 to the cost  (proxsuite form: 1/2 x^T H x + g^T x)."""
    H += 2 * w * A_t.T @ A_t
    g -= 2 * w * A_t.T @ b_t


# ---------------------------------------------------------------------------
# proxsuite solver (dense: the problem is small and all matrices are dense)
# ---------------------------------------------------------------------------
qp = proxsuite.proxqp.dense.QP(N_VAR, N_EQ, N_IN)
qp.settings.eps_abs = 1e-5
qp.settings.eps_rel = 0.0
qp.settings.max_iter = 1000
qp.settings.verbose = False
qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
qp_initialized = False

J_com_prev = None

with mujoco.viewer.launch_passive(
    m, d, key_callback=key_callback
) as viewer:
  while viewer.is_running():
    step_start_time = time.time()

    if is_running_sim:
      mujoco.mj_step1(m, d)
      
      
      # decision variables: q_dot_dot(35), tau (29), f (24) = 88
      
      # equality constraints: 
      # 1) M q_dot_dot + h = S^T tau + J_c^T f # base+joints dynamics
      # 2) J_foot q_dot_dot + Jdot_foot q_dot = -K_d v_foot # foot contact constraints
      
      # inequality constraints:
      # 3) f_z >= 0
      # 4) |f_x| <= mu f_z, |f_y| <= mu f_z with mu = 0.6. (linear approximation of friction cone)
      # 5) tau_min <= tau <= tau_max
      
      # cost
      # each part of the cost is weigthed and used norm2. it has the following form:
      
      # || J_t q_dot_dot + J_tdot q_dot - a_t||^2
      
      # where a_t is the desired acceleration 
      
      # Task, Jacobian, a_t, weight
      # 6) COM position, mj_jacSubtreeCom, Kp (c_ref - c) - Kd v_c, 100
      # 7) Pelvis orientation, angular part of mj_jacBody, Kp e_R - Kd w, 50, 
      # 8) Posture(joints), 0, I, Kp (q_ref - q) - Kd v_q,
      # 9) force regularization, 0,||f||^2 , 1e-4
      # 10) torque regularization, 0, ||tau||^2, 1e-5
      
      
      # references
      q_ref = m.key_qpos[KEY, 7:].copy()
      com_ref = d.subtree_com[PELVIS].copy()
      quat_ref = d.xquat[PELVIS].copy()
      
      com_ref[0] = 0.03 * np.sin(2 * np.pi * d.time/2)  # small oscillation in x for testing
      #com_ref[1] = 0.05 * np.sin(2 * np.pi * d.time)  # small oscillation in x for testing
      #quat_ref[3] = 0.2 * np.sin(2 * np.pi * d.time )  # small oscillation in x for testing
      
      #q_ref[18] = 0.8 + 1 * np.sin(2 * np.pi * d.time*2)  # small oscillation in left knee for testing
      #q_ref[23] = -0.8 - 1 * np.sin(2 * np.pi * d.time*2)  # small oscillation in right knee for testing
      
      
      qd = d.qvel
      h = d.qfrc_bias.copy()                 # Coriolis + gravity
      M = np.zeros((NV, NV))
      mujoco.mj_fullM(m, d, M)
      J_c = contact_jacobian(m, d)           # 24 x 35, one 3-row block per sphere
      J_feet, Jdot_qd_feet = feet_jacobians(m, d)

      # --- equality constraints  A x = b ---
      A = np.zeros((N_EQ, N_VAR))
      b = np.zeros(N_EQ)
      # 1) M q_dd + h = S^T tau + J_c^T f   ->   [M  -S^T  -J_c^T] x = -h
      A[:NV, IQ] = M
      A[:NV, IT] = -S.T
      A[:NV, IF] = -J_c.T
      b[:NV] = -h
      # 2) J_feet q_dd + Jdot_feet q_d = -KD_FOOT v_feet
      A[NV:, IQ] = J_feet
      b[NV:] = -Jdot_qd_feet - KD_FOOT * (J_feet @ qd)

      # --- cost  sum_t w_t ||A_t x - b_t||^2 ---
      H = np.zeros((N_VAR, N_VAR))
      g = np.zeros(N_VAR)

      # 6) COM position
      J_com = np.zeros((3, NV))
      mujoco.mj_jacSubtreeCom(m, d, J_com, PELVIS)
      Jdot_qd_com = np.zeros(3) if J_com_prev is None else (J_com - J_com_prev) / sim_dt @ qd
      J_com_prev = J_com.copy()
      com = d.subtree_com[PELVIS]
      a_com = KP_COM * (com_ref - com) - KD_COM * (J_com @ qd)
      A_t = np.zeros((3, N_VAR)); A_t[:, IQ] = J_com
      add_task(H, g, A_t, a_com - Jdot_qd_com, W_COM)

      # 7) pelvis orientation (world frame) (instead of using quaternion, use rotation vector q_err = q_ref x q_current^-1)
      # where q_current^-1 is q_conj, meaning the inverse of the current orientation quaternion,  obtained with negQuat
      J_ori = np.zeros((3, NV)); Jdot_ori = np.zeros((3, NV))
      mujoco.mj_jacBody(m, d, None, J_ori, PELVIS) # Jacobian of the pelvis orientation 
      mujoco.mj_jacDot(m, d, None, Jdot_ori, d.xpos[PELVIS], PELVIS)
      q_conj = np.zeros(4); q_err = np.zeros(4); e_R = np.zeros(3)
      mujoco.mju_negQuat(q_conj, d.xquat[PELVIS])
      mujoco.mju_mulQuat(q_err, quat_ref, q_conj)      # rotation current -> ref, world frame
      mujoco.mju_quat2Vel(e_R, q_err, 1.0)             # as rotation vector (axis * angle) 
      a_ori = KP_ORI * e_R - KD_ORI * (J_ori @ qd)
      A_t = np.zeros((3, N_VAR)); A_t[:, IQ] = J_ori
      add_task(H, g, A_t, a_ori - Jdot_ori @ qd, W_ORI)

      # 8) posture (joints): J = [0 I], Jdot = 0
      q_err = q_ref - d.qpos[7:]
      a_post = KP_POST * q_err - KD_POST * qd[6:]
      A_t = np.zeros((NU, N_VAR)); A_t[:, IQ] = S      # S selects q_dd[6:] (the joints)
      add_task(H, g, A_t, a_post, W_POST)

      # 9) force regularization, 10) torque regularization
      H[IF, IF] += 2 * W_F * np.eye(NF)
      H[IT, IT] += 2 * W_TAU * np.eye(NU)

      # --- solve ---
      if not qp_initialized:
        qp.init(H, g, A, b, C, l, u)
        qp_initialized = True
      else:
        qp.update(H=H, g=g, A=A, b=b)                  # C, l, u are constant
      qp.solve()

      if qp.results.info.status != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
        print(f"t={d.time:.3f}  QP status: {qp.results.info.status}")

      x = qp.results.x
      tau = x[IT]
      f = x[IF]

      n_clips = int(np.sum((tau <= lo + 1e-6) | (tau >= hi - 1e-6)))   # torques at their limit

      #write ctrl
      d.ctrl[:] = np.clip(tau, lo, hi)

      # QP vertical force per foot: z components are every 3rd entry
      fz_left_cmd = f[2:3*NL:3].sum()
      fz_right_cmd = f[3*NL+2::3].sum()
      rms_leg_err = np.sqrt(np.mean(q_err[:12] ** 2))   # legs = first 12 joints = qpos[7:19]

      mujoco.mj_step2(m, d)

      # actual vertical force per foot from the simulated contacts (valid after mj_step2)
      fz_left, fz_right = foot_vertical_forces(m, d)
      logs.append((d.time, rms_leg_err,
                   fz_left_cmd, fz_right_cmd,
                   fz_left, fz_right, fz_left + fz_right, MG,
                   n_clips))
      
      if d.time - last_render_time >= render_interval:
          with viewer.lock():
            renderer.update_scene(d, camera=viewer.cam)
          pixels = renderer.render()
          frames.append(pixels)
          last_render_time = d.time

      
    
    with viewer.lock():
      mujoco.mj_forward(m, d)        # recompute xpos/xquat/geom poses from qpos

    viewer.sync()
    
    time_until_next_step = sim_dt - (time.time() - step_start_time)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

  renderer.close()
      
      
if SAVE:
  # create directory in results
  if not os.path.exists(f'./results/{EXPERIMENT_NAME}'):
    os.makedirs(f'./results/{EXPERIMENT_NAME}')    
  
  
  if frames:
    media.write_video(f'./results/{EXPERIMENT_NAME}/video.mp4', frames, fps=fps)
  
  if logs:
    # saved as a CSV file
    np.savetxt(f'./results/{EXPERIMENT_NAME}/logs.csv', np.array(logs), delimiter=',', fmt='%.6g', header='time,rms_leg_joint_err_rad,fz_left_cmd_N,fz_right_cmd_N,'
                                   'fz_left_N,fz_right_N,fz_total_N,mg_N,saturation_count',
               comments='')

    # plots (PNG) saved in the same folder as the CSV
    for path in save_plots(f'./results/{EXPERIMENT_NAME}/logs.csv'):
      print('saved', path)