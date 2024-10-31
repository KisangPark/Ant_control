#mujoco ant environment

#library imports

import numpy as np
import os
import time
import mujoco

#hyper parameters
max_action_num = 1000
minimum_dist = 1
target_position = [10, 0]



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

        #global_position = self.data.xpos[torso_id][0:1]
        #0,1,2 index for global x,y,z position

        dist = calc_distance(self.data.qpos[0:2], target_position) #global_position

        if self.is_healthy() == 0:
            done_mask = 1
            success = 0

        elif self.action_num > max_action_num:
            done_mask = 1
            success = 0

        elif dist < minimum_dist: #success case
            done_mask = 1
            success = 1
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
        self.state = np.concatenate((np.ndarray.flatten(self.data.qpos), np.ndarray.flatten(qvel_equalized)))# 29 number array
        self.state = self.state
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

        dist = calc_distance(self.data.qpos[0:2], target_position)
        
        reward = np.exp((10 - dist)/2) # 15
        #starting from 0.9, end almost at 13~14

        if self.is_healthy():
            reward += 1

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



    def is_healthy(self):
        #qpos z value check
        z_pos = self.data.qpos[2]
        if z_pos < 0.45 and z_pos > 0.8: #unhealthy case
            return 0
        else:
            return 1


    def return_self_action_num(self):
        return self.action_num


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

