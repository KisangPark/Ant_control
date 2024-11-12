import mujoco
import time
import itertools
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import cv2

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from IPython.display import clear_output
clear_output()



# Load the MuJoCo model
model_path = "/home/kisang-park/mujoco/ant.xml"  # Adjust this to your Ant model path
model = mujoco.MjModel.from_xml_path(model_path)
model2 = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
data2 = mujoco.MjData(model2)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

#rendering
mujoco.mj_resetData(model, data)
mujoco.mj_resetData(model2, data2)
while 1:
    with mujoco.Renderer(model) as renderer:
        mujoco.mj_step(model, data)
        renderer.update_scene(data, scene_option=scene_option)
        pixels = renderer.render()
        image = np.array(pixels, dtype = np.uint8)
        #image = np.swapaxes(image, 0, 2)
        print (image.shape)
        #plt.imshow(image)
        #plt.show()
        cv2.imshow('image',image)
        cv2.waitKey(1)
        #time.sleep(0.5)

cv2.destroyAllwindows()

