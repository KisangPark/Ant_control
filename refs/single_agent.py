#for control test & single agent

"""
structure for the code
(0. read xml for ant)
1.make xml file for ant & interactive obj (reference!!!! mujoco tutorial)
2. define single RL agent (reference: f1_tenth, gym ant code)
3. Try multi-agent swarm optimization & RL (homogeneous agents, identical policies)
4. 
"""


import time
import random

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('ant.xml')
d = mujoco.MjData(m)
i=0

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    
    d.ctrl[1]=random.uniform(-1,1)
    d.ctrl[3]=random.uniform(-1,1)
    d.ctrl[5]= random.uniform(-1,1)
    d.ctrl[7] = random.uniform(-1,1)
    i +=1
    #d.ctrl[:]=random.uniform(-1,1)
    """
d.ctrl[7] = random.uniform(-1, 1)
    """
    #print("xpos::",d.xpos[0])
    #print("qpos, angle", d.qpos[0:2])
    ID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    print("qpos:",d.qpos[2], i)

    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)