#mujoco contact environment

"""
final data

env returns list of states, action, next_states, reward, done_mask, success
states return list of qpos(15) - qvel(14) - qacc(14) - contact_force(3) - box_pos(3)
    -> use divide_state for MACC, use np.concatenate for single_agent
        ex) self.state = np.concatenate((np.ndarray.flatten(qpos_equalized), np.ndarray.flatten(qvel_equalized)))

picture save lesser... not every success

box xml
body name = "box" pos = "define"
<inertial pos = "0 0 0" mass = "1.0(define)" diaginertia = "0.1 0.1 0.1"/>
<geom type = "box" size = "1 1 1" rgba="1 0 0 1"/>
<joint type="free"/>
<body/>

"""

import numpy as np
import os
import time
import mujoco

#hyper parameters
max_action_num = 200000
minimum_dist = 1
target_position = [0, 8]



def calc_distance(a,b):
    dist = np.sqrt(np.sum(np.square(a-b)))
    return dist


"""
Contact Environment

box & ant defined in xml -> get position & velocity value from xml data
method of get_position: break down qpos, qvel, contact_force (MACC -> divide state method included...)
# final: send states as 1d array

*****
qpos, qvel, qacc, contact force (if available -> How to implement?)
Theory: QACC can replace
average contact force for global reward.......

check_done, step, reset, is_healthy, is_moving

1) check_done
    done if Ant unhealthy, ant out of bound (distance)
    done if box out of bound

2) step - reward
    Ant: moving condition (global moving, joint moving), healthy reward (if unhealthy minus reward)
    Box: moving reward (oriented, big reward), distance reward
    additional: box-ant distance, stability term (for ant & box)

3) is_healthy
    modify to return values
    smaller pitch & roll for bigger reward
    
4) is_moving (for ant): nothing to be modified
5) box_is_moving (for box): for terminalization

how to get position of box
    mj_name2id -> muOBJ_BODY for ID
    body ID: box
    data.xpos ID -> xpos returns the position of each -> pass 3*body_id for xpos
    get data and check!


try test code of returning qpos, qvel, qacc in box_ant
mean contact force (global)

"""


class CONTACT_ENV():

    def __init__(self, **kwargs):

        try:
            self.xml_path = kwargs['xml_path']
        except:
            self.xml_path = "C:/Users/gpu/kisang/Ant_control/rl_env/ant_box.xml"

        #initialize model, starting condition
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.action_num = 0
        
        #Ant
        self.qpos = []
        self.qvel = []
        self.qacc = []
        self.contact_force = []

        #Box
        self.box_pos = []

        #total state
        self.state = [np.array([-1.28745179e-18 , 4.00000000e+00 , 6.37708673e-01 , 1.00000000e+00,
  2.34819987e-18,  0.00000000e+00,  0.00000000e+00 , 2.14378271e-20,
 -4.28563624e-20 , 7.59658514e-01 , 1.00000000e+00,  1.04681436e-18,
  0.00000000e+00,  0.00000000e+00,  4.19109008e-21,  1.38842213e-01,
  2.76536143e-22, -1.38842213e-01 ,-2.82606611e-21, -1.38842213e-01,
 -1.26737467e-21 , 1.38842213e-01]), np.array([-1.16641870e-16 ,-1.99841913e-16 , 9.46836559e+00  ,2.81053677e-16,
 -1.57771747e-16,  3.32167444e-19,  1.39433741e-18, -3.00047259e-18,
  6.05208933e-01, -8.20163556e-17 ,-3.69768675e-17,  2.29349934e-19,
  5.13551618e-19,  9.63761834e+00, -1.61808905e-20, -9.63761834e+00,
 -3.89748384e-19 ,-9.63761834e+00, -9.09256629e-20,  9.63761834e+00]) ,np.array([ 3.80944285e-18  ,4.14547122e-18, -9.81459055e+00, -8.74321637e-18,
  7.53717729e-18 ,-1.24139885e-20,  1.84038748e-18,  1.53754252e-18,
 -1.17887078e+01, -3.37525547e-18, -1.93947247e-18, -4.43743143e-23,
  5.47754564e-17, -1.01089942e+01, -5.50734590e-17,  1.01089942e+01,
 -5.47972522e-17 , 1.01089942e+01,  5.50955464e-17, -1.01089942e+01]), np.array([0, 0, 0])]

        self.dist = 8 #initial distance
        self.inter_dist = 4

        # IDs, fixed
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")



    def __del__(self):
        pass


    """
    1) check_done
    done if Ant unhealthy, ant out of bound (distance)
    done if box out of bound
    """

    def check_done(self):
        
        done_mask = 0
        success = 0
        dist = 0

        #return x, y position value for ant and box
        torso_pos = self.data.xpos[self.torso_id][0:2]
        box_pos = self.data.xpos[self.box_id][0:2]

        #calculate box distance
        box_dist = calc_distance(box_pos, target_position)
        ant_dist = calc_distance(torso_pos, target_position)

        #done check
        #1. ant healthy
        if self.is_healthy() == 0: #????????????????????????????????????????????????????????????????????
            done_mask = 1
            success = 0

        #2. ant out of bounds
        elif ant_dist > 10:
            done_mask = 1
            success = 0

        #3. box out of bounds
        elif box_dist > 10:
            done_mask = 1
            success = 0

        #4. success = box in region
        elif box_dist < minimum_dist:
            done_mask = 1
            success = 1

        #5. max action num
        elif self.action_num > max_action_num:
            done_mask = 1
            success = 0
        
        else:
            done_mask = 0
            success = 0
        

        return done_mask, success


    """
    2) step - reward
    Ant: moving condition (global moving, joint moving), healthy reward (if unhealthy minus reward)
    Box: moving reward (oriented, big reward), distance reward
    additional: box-ant distance, stability term (for ant & box)

    get control array -> put and mj step -> get qpos, qvel, qacc, box_pos, contact force

    form of state: qpos + qvel + qacc + contact_force + box_pos
    total length: 15+14+14+3+3 = 49

    not like this... just return the tuple (iterable) of them
    5 element tuple return
    or list..... state append

    in MACC divide_state, get each & index
    """

    def calc_reward(self, done_mask, success):
        """
        reward function

        2) step - reward
        Box: moving reward (oriented, big reward), distance reward
        Ant: moving condition (global moving, joint moving), healthy reward (if unhealthy minus reward)

        additional: box-ant distance, stability term (for ant & box), contact advantage
        """
        #box distance, box velocity, inter-distance
        old_dist = self.dist
        box_pos = self.data.xpos[self.box_id][0:2]
        ant_pos = self.data.xpos[self.torso_id][0:2]
        self.dist = calc_distance(box_pos, target_position)

        box_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data, mujoco.mjtObj.mjOBJ_BODY, self.box_id, box_velocity, 0)

        self.inter_dist = calc_distance(box_pos, ant_pos)

        
        #1. box reward (default values... velocity and distance)
        if old_dist > self.dist: 
            reward = (9 - self.dist)*3
            #reward = np.exp((10 - dist)/2) # 15
        else:
            reward = 0

        #2. Ant moving condition (global)
        if self.is_moving():
            reward += 2
        else:
            reward -= 2 #10 or 16
        
        #3. Ant healthy
        if self.is_healthy() != 0:
            reward += self.is_healthy()

        #4. inter-distance
        reward += np.min([2-self.inter_dist, 1])*2

        #5. contact advantage -> nope... bottom contact number always exists
        #if self.data.ncon !=0:
        #    reward += 1    

        #5. qvel
        if np.any(self.data.qvel == 0):
            reward -= 5
        else:
            reward += 1

        #6. done case
        if done_mask and not success:
            reward = 0

        return reward



    def step(self, ctrl_array):
        #state: list of qpos, qvel, qacc, contact force, box_pos

        #initialize forcetorque
        forcetorque = np.zeros(6)
        force = np.zeros(3)

        #mujoco step
        current_state = self.state
        action = ctrl_array

        for i in range(8):
            if ctrl_array[i] != ctrl_array[i]: #preventing nan input
                self.data.ctrl[i] = 0
                print("nan case occurred")
            else:
                self.data.ctrl[i] = ctrl_array[i]

        mujoco.mj_step(self.model, self.data)
        self.action_num += 1

        #print("qacc:", len(self.data.qacc))
        #print("qpos & qvel:", len(self.data.qpos)+len(self.data.qvel)) # -> box pos & velocity returned together!!!!!!! wow...
        #print("qpos:", self.data.qpos, self.data.qvel, self.data.qacc)
        #print("torso pos:", self.data.xpos[self.torso_id][0:3])
        #print("box pos:", self.data.xpos[self.box_id][0:3])
        #front 7 qpos & front 6 qvel & front 6 qacc is for box!!
        #print("contact:", self.data.contact)

        #get contact force
        if self.data.ncon == 0:
            pass
        else:
            for j, c in enumerate(self.data.contact):
                mujoco.mj_contactForce(self.model, self.data, j, forcetorque)
                force += forcetorque[0:3] #now, force is sum
                #print("force:", force)

        #equalize values
        qvel_equalized = self.data.qvel * 10
        qpos_equalized = self.data.qpos *10
        qacc_equalized = self.data.qacc *10
        #box_pos_eqalized = self.data.xpos[self.box_id][0:2] *10


        #next_state get
        self.state =[qpos_equalized, qvel_equalized, qacc_equalized, force] #box position always x & y
        done_mask, success = self.check_done()

        #reward calculation
        reward = self.calc_reward(done_mask, success)

        return current_state, action, self.state, reward, done_mask, success



    def reset(self):

        mujoco.mj_resetData(self.model, self.data)

        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)

        self.action_num = 0
        #self.state = np.zeros(29)

        self.dist = 8
        self.inter_dist = 4

        #check for other values



    def is_healthy(self):
        #qpos z value check
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        z_pos = self.data.xpos[torso_id][2]
        #print(self.data.qpos[0:2], z_pos)

        #get quaternion, change into euler angle
        w, x, y, z = self.data.qpos[3:7]
        pitch = np.arcsin(2.0*(w*y - z*x))
        roll = np.arctan2(2.0*(w*x+y*z), 1.0-2.0*(x*x + y*y))
        yaw = np.arctan2(2.0*(w*z+y*x), 1.0-2.0*(y*y + z*z))
        #print("pitch:", pitch)
        #print("roll:", roll)

        max_angle = 1 #np.pi/2
        min_angle = -1

        #reward: if pitch & roll near 0, plus reward / maximum max or min angle
        pitch_reward = np.min([abs(max_angle - pitch),abs(min_angle - pitch)])
        roll_reward = np.min([abs(max_angle - roll), abs(min_angle - roll)])
        reward = (pitch_reward+roll_reward)/2

        if pitch>max_angle or pitch<min_angle:
            return 0
        if roll>max_angle or roll<min_angle:
            return 0
        if z_pos < 0.4 or z_pos>1:
            return 0
        
        return reward

        """
        if z_pos < 0.45 and z_pos>1:
            return 0
        elif pitch>max_angle or pitch<min_angle:
            return 0
        elif roll>max_angle or roll<min_angle:
            return 0
        else:
            return 1
        #torso id position
        """


    def is_moving(self): #global velocity
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso_id, velocity, 0)
        xyz_velocity = velocity[:3]
        absolute_velocity = calc_distance(np.zeros(3), xyz_velocity)

        if absolute_velocity <0.1:
            return 0
        else:
            return 1


    def return_self_action_num(self):
        return self.action_num

    def return_dist(self):
        return self.dist

