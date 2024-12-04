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
minimum_dist = 1 #curriculum method... goal distance increase & closer target position
target_position = [0, 5]



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
            self.xml_path = "C:/kisang/Ant_control/rl_env/ant_box.xml"
		#C:/kisang/Ant_control/result_macc
		#C:/Users/gpu/kisang/Ant_control/rl_env/ant_box.xml

        #initialize model, starting condition
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        #parameter need to be saved -> action number, initial distance, ID and Global position
        self.action_num = 0
        self.box_dist = 3.7
        self.ant_dist = 5
        self.inter_dist = 1.3

        # IDs, fixed
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")


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
 -3.89748384e-19 ,-9.63761834e+00, -9.09256629e-20,  9.63761834e+00]), np.array([0, 0]), np.array([3.7, 5, 1.3])]





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
        elif ant_dist > 7:
            done_mask = 1
            success = 0

        #3. box out of bounds
        elif box_dist > 5:
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

        1) procedure
        old distance save -> new distance (three distances) calcuate
        -> distance comparison

        2) step - reward
        Box: moving reward (oriented, big reward), distance reward
        Ant: moving condition (global moving, joint moving), healthy reward (if unhealthy minus reward)

        additional: box-ant distance, stability term (for ant & box), contact advantage
        """
        #box distance, box velocity, inter-distance
        old_box_dist = self.box_dist
        old_ant_dist = self.ant_dist
        old_inter_dist = self.inter_dist

        box_pos = self.data.xpos[self.box_id][0:2]
        ant_pos = self.data.xpos[self.torso_id][0:2]
        
        self.box_dist = calc_distance(box_pos, target_position)
        self.ant_dist = calc_distance(ant_pos, target_position)
        self.inter_dist = calc_distance(box_pos, ant_pos)

        box_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data, mujoco.mjtObj.mjOBJ_BODY, self.box_id, box_velocity, 0)

        
        #1. box reward
        if old_box_dist > self.box_dist: 
            reward = (4.7 - self.box_dist)*4
        else:
            reward = 0

        #2. inter distance & ant distance reward
        if old_ant_dist > self.ant_dist:
            temp = 2
        else:
            temp = 0
        if old_inter_dist > self.inter_dist:
            temp += 2
        reward += temp

        #3. Ant healthy
        if self.is_healthy() != 0:
            reward += 0.5 * self.is_healthy()

        #4. Ant moving condition (global)
        if self.is_moving():
            reward += 1
        else:
            reward -= 1

        #5. qvel
        if np.any(abs(self.data.qvel[7:]) < 0.01): #problem... box included
            reward -= 1

        if np.all(abs(self.data.qvel[7:]) < 0.01):
            reward -= 3

        #6. done case
        if done_mask and not success:
            reward = 0

        return reward



    def step(self, ctrl_array):
        #state: list of qpos, qvel, qacc, contact force, box_pos

        #initialize forcetorque
        forcetorque = np.zeros(6)
        force = np.zeros(2)

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
                force += forcetorque[0:2] #now, force is sum
                #print("force:", force)

        #equalize values
        qvel_equalized = self.data.qvel * 10
        qpos_equalized = self.data.qpos *10

        #reward, donemask, success calculated -> self distances calculated
        done_mask, success = self.check_done()
        reward = self.calc_reward(done_mask, success)
        distance_set = [self.box_dist, self.ant_dist, self.inter_dist]


        #next_state get
        self.state =[qpos_equalized, qvel_equalized, force, distance_set] #box position always x & y



        return current_state, action, self.state, reward, done_mask, success



    def reset(self):

        mujoco.mj_resetData(self.model, self.data)

        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)

        self.action_num = 0
        #self.state = np.zeros(29)

        self.box_dist = 3.7
        self.ant_dist = 5
        self.inter_dist = 1.3

        #check for other values



    def is_healthy(self):
        #qpos z value check
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        z_pos = self.data.xpos[torso_id][2]
        #print(self.data.qpos[0:2], z_pos)

        #get quaternion, change into euler angle
        w, x, y, z = self.data.qpos[10:14] #wrong...
        pitch = np.arcsin(2.0*(w*y - z*x))
        roll = np.arctan2(2.0*(w*x+y*z), 1.0-2.0*(x*x + y*y))
        yaw = np.arctan2(2.0*(w*z+y*x), 1.0-2.0*(y*y + z*z))
        #print("pitch:", pitch)
        #print("roll:", roll)

        max_angle = 2 #np.pi/2
        min_angle = -2

        #reward: if pitch & roll near 0, plus reward / maximum max or min angle
        pitch_reward = np.min([abs(max_angle - pitch),abs(min_angle - pitch)])
        roll_reward = np.min([abs(max_angle - roll), abs(min_angle - roll)])
        reward = (pitch_reward+roll_reward)/4

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
        return self.box_dist


