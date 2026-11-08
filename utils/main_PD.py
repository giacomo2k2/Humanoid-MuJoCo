import os
import time
import mediapy as media
import mujoco
import mujoco.viewer

import numpy as np

from utils.plot_logs import save_plots  # utils/plot_logs.py (same folder as this script)

SAVE = True
EXPERIMENT_NAME = "PD_control_gravity_naive"


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


def sole_height(m, d, body_name):
    bid = m.body(body_name).id
    geoms = [g for g in range(m.ngeom)
             if m.geom_bodyid[g] == bid and m.geom_contype[g] | m.geom_conaffinity[g]]
    return min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in geoms)

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



KP = np.diag([200, 200, 200,
      200, 200, 200,
      200, 200, 200,
      200, 200, 200,
      80, 80, 80,
      30, 30, 30,
      30, 30, 30,
      30,
      30, 30, 30,
      30, 30, 30,
      30])


M = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, d, M)

KD = np.zeros(29)
for i in range(29):
  kd = 2 * np.sqrt(KP[i, i] * M[6 + i, 6 + i])
  KD[i] = kd

KEY = m.key('stand_bent').id

with mujoco.viewer.launch_passive(
    m, d, key_callback=key_callback
) as viewer:
  while viewer.is_running():
    step_start_time = time.time()

    if is_running_sim:
      mujoco.mj_step1(m, d)
      
      # compute tau
      h = d.qfrc_bias  #with velocity almost zero, this is gravity
      M = np.zeros((m.nv, m.nv))
      mujoco.mj_fullM(m, d, M)
      J_c = contact_jacobian(m, d)
     
      contact_forces = np.linalg.pinv(J_c[:, :6].T) @ h[0:6]  # only consider the first 6 DOFs (base)
      
            
      joint_torque = h[6:] #- J_c[:, 6:].T @ contact_forces  # compute joint torques 
        
      q_des = m.key_qpos[KEY, 7:]
      q_err = q_des - d.qpos[7:]
      ctrl_torque =  joint_torque + KP @ q_err - KD * d.qvel[6:]  # PD control for joints (KD is a vector)


      n_clips = 0
      for i in range(len(ctrl_torque)):
        if abs(ctrl_torque[i]) > m.actuator_ctrlrange[i][1]:
          ctrl_torque[i] = np.sign(ctrl_torque[i]) * m.actuator_ctrlrange[i][1]
          n_clips += 1

      #write ctrl
      d.ctrl[:] = ctrl_torque

      # planned (pinv) vertical force per foot: z components are every 3rd entry
      fz_left_cmd = contact_forces[2:3*NL:3].sum()
      fz_right_cmd = contact_forces[3*NL+2::3].sum()
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