import time

import mujoco
import mujoco.viewer
import mediapy as media

m = mujoco.MjModel.from_xml_path('./models/unitree_g1/scene.xml')
d = mujoco.MjData(m)

fps = 30
width, height = 1280, 720
renderer = mujoco.Renderer(m, height=height, width=width)

frames = []
sim_dt = m.opt.timestep
render_interval = 1 / fps
last_render_time = 0.0

with mujoco.viewer.launch_passive(m, d) as viewer:
  while viewer.is_running():
    step_start_time = time.time()

    mujoco.mj_step(m, d)
    
    if d.time - last_render_time >= render_interval:
      with viewer.lock():
        renderer.update_scene(d, camera=viewer.cam)
        
      pixels = renderer.render()
      frames.append(pixels)
      last_render_time = d.time

    viewer.sync()

    time_until_next_step = sim_dt - (time.time() - step_start_time)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
      
  renderer.close()
      
      
if frames:
  media.write_video('./results/media/robot_no_control.mp4', frames, fps=fps)
  print(f"Video saved ({len(frames)} frames).")
