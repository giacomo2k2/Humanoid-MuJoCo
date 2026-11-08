# mujoco_humanoid

Standing / balance controllers for the Unitree G1 in MuJoCo.

## Layout

```
main.py                 entry point
controllers/base.py     ControlOutput: what every controller returns (tau, planned forces, q_ref)
controllers/PD.py       joint PD + feedforward (none | gravity | contact)
controllers/QP.py       whole-body QP (proxsuite): dynamics, feet contact, friction, torque limits
utils/robot.py          G1 model: keyframe, foot spheres, Jacobians, simulated foot forces
utils/viewer.py         simulation loop (viewer or headless) + video recorder
utils/logger.py         per-step metrics -> logs.csv
utils/plotter.py        logs.csv -> PNG plots
models/unitree_g1/      MJCF model (scene.xml includes g1_v2.xml)
```

## Run (from the project root)

```bash
python main.py                                   # QP, static COM reference, viewer
python main.py --controller QP --ref forward     # COM oscillates forward/back
python main.py --controller QP --ref lateral     # COM oscillates left/right
python main.py --controller PD --feedforward gravity
MUJOCO_GL=egl python main.py --headless --duration 5   # no window
python -m utils.plotter results/QP_static/logs.csv     # re-plot an existing run
```

In the viewer, **Space** starts / pauses the simulation. Results are written to
`results/<name>/` (`logs.csv`, plots, `video.mp4`); `--name` overrides the folder,
`--no-save` / `--no-video` disable saving.

## Adding a controller

Implement a class with `compute(self, t) -> ControlOutput` (it can read `robot.m` / `robot.d`;
the loop calls it between `mj_step1` and `mj_step2` and clips `tau` to the actuator limits),
then add it to `main.py`.
