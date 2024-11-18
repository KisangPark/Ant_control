"""

MACC
Multi-agent DDPG - using Centralized Critic

same sequence: getting result from envrionment
difference
    1) four actors & two buffers in MACC agent ->sort states in 4 actors, append in action buffer
    2) train function: direct learning in critic, 4 times iteration in actor

*** Consider CTDE case... Too many networks




**for divide state
all dividing state executed inside agent... thus no dividing in main code
"""

import os
import random
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import Normal

import mujoco
import mujoco.viewer

from IPython import display
plt.ion()
os.environ["PYOPENGL_PLATFORM"] = 'egl'

import rl_env.contact_env
from rl_env.OUNoise import OUNoise


# Hyperparameters
num_episodes = 1000
learning_rate = 0.0001
gamma = 0.99
TAU = 0.005
batch_size = 32 #64
buffer_size = 30000
num_epochs = 2

state_dim = 49 # plus goal, contact force
actor_state_dim = 25
action_dim = 8

model = mujoco.MjModel.from_xml_path('ant_box.xml') #xml file changed
data = mujoco.MjData(model)

highest_speed = 5000 # maximum steps

work_dir = "/home/kisang-park/Ant_control/result_macc"



def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s


def plot(reward, dist, episode, flag):
    if episode%5 == 0:

        plt.figure(2)
        plt.cla() #delete all
        #durations_t = torch.tensor(laptimes, dtype=torch.float) #torch float tensor for duration
        plt.title('Result plot')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.plot(reward) # torch tensor to numpy
        plt.plot(dist)
        plt.pause(0.01)

    if flag==1:
        plt.savefig('result_plot.png')


def divide_state(state):
    """
    receiving state: qpos, qvel, qacc, force, box_pos
    qpos: 7 for torso, 8 for each joints
    qvel & qacc: 6 for torso, 8 for each joints
    force: common
    box_pos: common
    """
    state_list = []
    for qpos, qvel, qacc, force, box_pos in state:
        common_state = np.concatenate([qpos[0:7], qvel[0:6], qacc[0:6], force, box_pos])
    
    start = 6
    finish = 8
    for i in range(4):
        temp_state = np.concatenate([common_state, qpos[start+1:finish+1], qvel[start:finish], qacc[start:finish]])
        state_list.append(temp_state)
        start += 2
        finish += 2

    return state_list
    #list of four lists



"""pre code"""


"""
Network, Buffer def

Actor: forward (state_dim, action_dim)
Critic: forward (state_dim, action_dim)
buffer: add, sample, size
"""

class Actor(nn.Module):
    def __init__(self, actor_state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(actor_state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x)) #relu
        x = F.tanh(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        #print(state.size(), action.size())
        x = torch.cat([state, action], dim=1) #1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) 
        states, actions, next_states, rewards, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(next_states),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(dones).unsqueeze(1)#,
            )#returns 

    def size(self):
        return len(self.buffer)



"""MACC agent definition"""

"""
1. initialize
    four actors (+ four targets) & one critic (+ one target) & two buffers initialize
    actor & actor target network init (+ optimizer)
    *****critic network input: state + action*****

2. action
    put state to four actor networks, numpy, flatten

3. train
    if buffer too small, return
    sample from replay buffer critic
    - actor learning
    in dpg, parameter to Q gradient = parameter to action X action to Q
    action to Q parameter already calculated & updated in critic update
    therefore, using Q mean of actor network output, can update the parameter to action that is policy update

4. main
environment & dimension & agent declare
in episodes, if not done env step & buffer push (two sequence)

"""

class MACCagent:
    def __init__ (self, state_dim, actor_state_dim, action_dim): #actor state dimension

        #critic part
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        #actor part - 4 actors
        self.actor1 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor1_target = Actor(actor_dim, action_dim)#.to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())
        self.actor1_optimizer = optim.Adam(self.actor1.parameters(), lr=learning_rate)

        self.actor2 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor2_target = Actor(actor_dim, action_dim)#.to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())
        self.actor2_optimizer = optim.Adam(self.actor2.parameters(), lr=learning_rate)

        self.actor3 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor3_target = Actor(actor_dim, action_dim)#.to(device)
        self.actor3_target.load_state_dict(self.actor3.state_dict())
        self.actor3_optimizer = optim.Adam(self.actor3.parameters(), lr=learning_rate)

        self.actor4 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor4_target = Actor(actor_dim, action_dim)#.to(device)
        self.actor4_target.load_state_dict(self.actor4.state_dict())
        self.actor4_optimizer = optim.Adam(self.actor4.parameters(), lr=learning_rate)

        #buffer
        self.Qbuffer = ReplayBuffer(buffer_size)


    def action (self, in_state): #multiple actors!!

        output = []

        arr_state = divide_state(in_state)
        i=0

        for actor in (self.actor1, self.actor2, self.actor3, self.actor4):
            state = torch.FloatTensor(arr_state[i])
            out = actor(state).detach().numpy()
            output.append(out)
            i+=1
        return output.flatten()
        


    def train (self):
        if self.Qbuffer.size() < batch_size:
            print("short")
            return

        #critic training
        states, actions, next_states, rewards, dones = self.Qbuffer.sample(batch_size) #buffer have 1d state
        arr_states = divide_state(states)
        arr_next_states = divide_state(next_states)

        with torch.no_grad():
            #make iterable..?
            i=0
            next_actions = []

            for target_network in (self.actor1_target, self.actor2_target, self.actor3_target, self.actor4_target):
                next_action = target_network(arr_next_states[i]) #next_action = size 2
                next_actions.append(next_action)
                i+=1

            next_actions = torch.cat(next_actions, dim=1) #size 8            
            target_q = rewards + (1 - dones) * gamma * self.critic_target(next_states, next_actions) #next_states = length 49
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #actor trainig
        actions = []
        j=0
        for actor in (self.actor1_target, self.actor2_target, self.actor3_target, self.actor4_target):
            action = actor(arr_states[j])
            actions.append(action)
            j+=1

        actor_actions = torch.cat(actions, dim=1)
        actor_loss = -self.critic(states, actor_actions).mean()

        for optimizer in (self.actor1_optimizer, self.actor2_optimizer, self.actor3_optimizer, self.cator4_optimizer):
            optimizer.zero_Grad()
            actor_loss.backward()
            optimizer.step()

        #target update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor3.parameters(), self.actor3_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor4.parameters(), self.actor4_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def return_net(self, num): #returning network parameters
        
        today = get_today()

        i = 1
        for actor in (self.actor1, self.actor2, self.actor3, self.actor4):
            torch.save(actor.state_dict(), work_dir + "/actor" + i + "_" + str(num)+str(today)+".pt")
            i+=1
        highest_speed = num

        print("success case returned, highest_speed:", highest_speed)




"""main"""

def main():
    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load environment
    env = rl_env.contact_env.CONTACT_ENV()

    #define PPO agent
    agent = MACCagent(state_dim, actor_state_dim, action_dim)

    #for plot
    rewards_forplot = []
    dist_forplot = []


    #episode loop
    for episode in range(num_episodes):

        #pre-execution
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0
        success = 0
        timestep =0

        #Noise distrbution, preventing deadlock
        #distribution = Normal(0, std_dev)
        
        #initialize environment
        env.reset()
        state, action, next_state, reward, done_mask, success = env.step(np.zeros(action_dim))
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)



        """execute one episode"""
        while done_mask == 0:
            
            action = agent.action(state)
            noise = OUNoise(action_dim).noise()
            action += noise #noisy action returned
            
            state, action, next_state, reward, done_mask, success = env.step(action) #env returns: np ndarray
            state = np.array(state)
            action = np.array(action)
            next_state = np.array(next_state)
            agent.Qbuffer.add(state, action, next_state,reward, done_mask)
            #noticed! -> can access to self variables using dot methods
            
            state = next_state
            total_reward += reward
            timestep+=1

            if timestep%100 == 0:
                plot_reward = total_reward/timestep
                print(timestep, "steped, total reward:", plot_reward)
                rewards_forplot.append(plot_reward)
                final_dist = env.return_dist()
                dist_forplot.append(final_dist)
                plot(rewards_forplot, dist_forplot, timestep, 0)

                #for num in range(num_epochs):
                agent.train()
        
        plot(rewards_forplot,dist_forplot, 1, 1)

        #if success
        if success == 1:
            num = env.return_self_action_num()
            agent.return_net(num)
            plot(rewards_forplot,dist_forplot, 1, 1)

    print ("all episodes executed")
    plot(rewards_forplot,dist_forplot, 1, 1)




"""evaluation net"""

class eval_net():
    def __init__():
        super().__init__()

    def forward(self, state):
        


def eval():

    actor_path = os.path.join(work_dir, "actor_1001_2024-11-09_12-48-17.pt")
    #dev_path = os.path.join(work_dir, "dev_368_2024-10-31_15-48-11.pt")

    i=0

    agent = eval_net(state_dim, action_dim, actor_path)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        while viewer.is_running():
            qvel_equalized = data.qvel * 10
            qpos_eqalized = data.qpos *10
            state = np.concatenate((np.ndarray.flatten(qpos_eqalized), np.ndarray.flatten(qvel_equalized)))
            action = agent(state)
            data.ctrl = action
            mujoco.mj_step(model,data)

            #print("dist:", np.sqrt(np.sum(np.square(data.qpos[0:2] - [15, 0]))))
            #dist correctly calculated

            i+=1
            if (i%100 == 0):
                print("100 steps", data.qpos[2])
            #print("step")
            viewer.sync()

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 1




if __name__ == '__main__':
    main()

    #eval()
