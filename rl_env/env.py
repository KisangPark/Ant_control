#mujoco ant environment

#library imports

import numpy as np
import os
import time
import mujoco

#hyper parameters
max_action_num = 10000
minimum_dist = 1
target_position = [5, 0]



def calc_distance(a,b):
    dist = np.sqrt(np.sum(np.square(a-b)))
    return dist



class ANTENV():

    def __init__(self, **kwargs): #multiple variables input, with dictionary
    #initializing variables? -> xml file path, parameters(if needed), number of agents, timestep
    #additional initialize -> self_pose variable, collision, 

        try:
            self.xml_path = kwargs['xml_path']
        except:
            self.xml_path = "ant.xml"

        #initialize model, starting condition
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # environment should be defined only once at the main code...

        self.action_num = 0
        
        self.qpos = [] #needed?
        self.xpos = []

        self.state = []
        self.dist = 5



    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass



    def check_done(self):
        
        done_mask = 0
        success = 0
        dist = 0

        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_pos = self.data.xpos[torso_id][0:2]
        #global_position = self.data.xpos[torso_id][0:1]

        #global_position = self.data.xpos[torso_id][0:1]
        #0,1,2 index for global x,y,z position

        dist = calc_distance(torso_pos, target_position) #self.data.qpos[0:2]

        if self.is_healthy() == 0:
            done_mask = 1
            success = 0

        #elif self.action_num > max_action_num:
        #    done_mask = 1
        #    success = 0

        elif dist < minimum_dist: #success case
            done_mask = 1
            success = 1

        elif dist > 10:
            done_mask = 1
            sucecss = 0

        else:
            done_mask = 0
            success = 0
        

        return done_mask, success




    def step(self, ctrl_array):

        current_state = self.state
        #self.sard.append(current_state)

        action = ctrl_array
        #self.sard.append(action)
        #print("action:", action)

        # control signal to data control
        for i in range(8):

            if ctrl_array[i] != ctrl_array[i]:
                #print("nan input..")
                self.data.ctrl[i] = 1
                print("ctrl signal changed to 0")
            else:
                self.data.ctrl[i] = ctrl_array[i]

        #print(self.data.ctrl)
        
        mujoco.mj_step(self.model, self.data)
        self.action_num += 1

        #print ("position:", self.data.qpos)
        #print ("velocity:", self.data.qvel)
        qvel_equalized = self.data.qvel * 10
        qpos_equalized = self.data.qpos *10

        #print("position values:", qpos_equalized)
        #print("velocity values:", qvel_equalized)

        self.state = np.concatenate((np.ndarray.flatten(qpos_equalized), np.ndarray.flatten(qvel_equalized)))# 29 number array
        #making all state variables to np array
        #self.sard.append(self.next_state)
        
        #done mask
        done_mask, success = self.check_done()

        #reward calculation
        """
        reward function

        1. if not success -> reward = 0
        2. reward by distance -> the nearer the robot is to the goal
        3. healthy reward (also added to terminalization -> check_done) -> with is_healthy method, check done & reward
        + calc distance method??? -> self variable
        """
        # reward return

        old_dist = self.dist
        self.dist = calc_distance(self.data.qpos[0:2], target_position)

        if old_dist > self.dist:
            reward = (6 - self.dist)*2 #for gradient, better learning, added gradient
            #reward = np.exp((10 - dist)/2) # 15
            #starting from 0.9, end almost at 13~14
        else:
            reward = -2
       

        if self.is_moving():
            reward += 1 #2 in goal 10
        else:
            reward -= 10

        #if self.is_healthy():
        #    reward += 1

        if done_mask and not success:
            reward = 0

        return current_state, action, self.state, reward, done_mask, success
        #five returns -> current state, action, next state, reward, done_mask



    def reset(self):

        mujoco.mj_resetData(self.model, self.data)

        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)

        self.action_num = 0
        self.state = np.zeros(29)

        self.dist = 5



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

        if z_pos < 0.45 and z_pos>1:
            return 0
        elif pitch>max_angle:
            return 0
        elif roll>max_angle:
            return 0
        else:
            return 1
        #torso id position


    def is_moving(self):
        #velocity check: non moving minus reward
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data, mujoco.mjtObj.mjOBJ_BODY, torso_id, velocity, 0)
        xyz_velocity = velocity[:3]
        absolute_velocity = calc_distance(np.zeros(3), xyz_velocity)

        #print(absolute_velocity)

        if absolute_velocity <0.1:
            return 0
        else:
            return 1



    def return_self_action_num(self):
        return self.action_num

    def return_dist(self):
        return self.dist


"""
output form

self.sard
current state / action / next state / reward / done_mask -> append
1-dimensional array returned

f1 env: observation, reward, done returned (each local variables)
self variable delete!!

direction:no self.sard array, return each variables!! -> 

variables that should left in self: action_num (number of step executed), collision (=done_mask, if different method should be added), current state
variables that can be used inside: next_state, reward (should be calculated), done_mask (should use check_done)
(action no need to return cause main code)

!! -> state update is executed in Qnet code, so if want to return state, there should be state self-variable
"""

"""
output form

all to initial conditions
zero value action
return observation, reward, done
"""

