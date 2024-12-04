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
num_episodes = 10000
learning_rate = 0.0001 #0.0001
gamma = 0.99
TAU = 0.1
batch_size = 32 #64
buffer_size = 200000
num_epochs = 2

state_dim = 47 # plus goal, contact force
actor_state_dim = 35
action_dim = 8 #each agent

#model = mujoco.MjModel.from_xml_path('C:/kisang/Ant_control/rl_env/box_walk.xml') #curriculum method! easier problem
model = mujoco.MjModel.from_xml_path('C:/kisang/Ant_control/rl_env/ant_box.xml') #xml file changed
data = mujoco.MjData(model)

highest_speed = 5000 # maximum steps

work_dir = "C:/kisang/Ant_control/result_macc"
#C:/kisang/Ant_control/result_macc
#/home/kisang-park/Ant_control/result_macc
#C:/Users/gpu/kisang/Ant_control/rl_env



def get_today():
    now = time.localtime()
    s = "%02d_%02d-%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    return s


def plot(reward, dist, timestep, flag):
    if timestep%10 == 0:

        plt.figure(2)
        plt.cla() #delete all
        #durations_t = torch.tensor(laptimes, dtype=torch.float) #torch float tensor for duration
        plt.title('Result plot')
        plt.xlabel('timestep / 10')
        plt.ylabel('Total Reward')
        plt.plot(reward) # torch tensor to numpy
        plt.plot(dist)
        plt.pause(0.01)

    if flag==1:
        save_path = os.path.join(work_dir + "/result_plot_" + str(timestep) + ".png")
        plt.savefig(save_path)


def flat_vectorize(state, batch_size):
    flattened = []

    if batch_size == 1: #for actions.... forwarding
        temp = []
        for q in state:
            for item in q:
                temp.append(item)
        return torch.FloatTensor(temp)

    for i in range(batch_size):
        temp = []
        #qpos, qvel, force, distance = state[i]
        
        for q in state[i]:
            for item in q:
                temp.append(item)

        flattened.append(temp)
        #print("flattened, temp:", len(flattened), len(temp))

    flattened = torch.FloatTensor(flattened)
    return flattened


def divide_state(state):
    """
    receiving state: qpos, qvel, force, distances
    qpos: 7 + 7 + 8 = 22 (7 for box, 7 for torso, 8 for joints)
    qvel: 6 + 6 + 8 = 20 (6 for box, 6 for torso, 8 for joints)
    force: common, 2
    distances: common, 3

    thus common state 31, private 4
    total of 47, individual of 35
    """
    state_list = []
    qpos, qvel, force, distance_list = state
    common_state = np.concatenate([qpos[0:14], qvel[0:12], force, distance_list]) #common knowledge, length 41
    
    start = 12
    finish = 14
    for i in range(4):
        temp_state = np.concatenate([common_state, qpos[start+2:finish+2], qvel[start:finish], qacc[start:finish]])
        state_list.append(temp_state)
        start += 2
        finish += 2

    return state_list
    #list of four lists


def get_cube(states, batch_size):
    cube, state1, state2, state3, state4 = [],[],[],[], []

    for j in range(batch_size):
        four_state = divide_state(states[j])

        i = 0
        for state_num in (state1, state2, state3, state4):
            state_num.append(four_state[i])
            i += 1
    
    for state_num in (state1, state2, state3, state4):
        cube.append(state_num)
    
    return cube


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
        self.fc3 = nn.Linear(256, 2) #not able to use action dim

    def forward(self, x):
        x = F.tanh(self.fc1(x)) #relu
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x/2


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
            states,
            torch.FloatTensor(actions),
            next_states,
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(dones).unsqueeze(1)#,
            ) #states not tensor because array

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
        self.actor1_target = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())
        self.actor1_optimizer = optim.Adam(self.actor1.parameters(), lr=learning_rate)

        self.actor2 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor2_target = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())
        self.actor2_optimizer = optim.Adam(self.actor2.parameters(), lr=learning_rate)

        self.actor3 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor3_target = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor3_target.load_state_dict(self.actor3.state_dict())
        self.actor3_optimizer = optim.Adam(self.actor3.parameters(), lr=learning_rate)

        self.actor4 = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor4_target = Actor(actor_state_dim, action_dim)#.to(device)
        self.actor4_target.load_state_dict(self.actor4.state_dict())
        self.actor4_optimizer = optim.Adam(self.actor4.parameters(), lr=learning_rate)

        #buffer
        self.Qbuffer = ReplayBuffer(buffer_size)

        self.highest_speed = 5000


    def action (self, in_state): #multiple actors!!

        output = []

        arr_state = divide_state(in_state)

        for i, actor in enumerate([self.actor1, self.actor2, self.actor3, self.actor4]):
            state = torch.FloatTensor(arr_state[i])
            out = actor(state).detach().numpy()
            output.append(out)
        return np.concatenate(output)#numpy output
        

    """
    problem in train
    -> short not visible, to big to get
    -> deque problem
    """
    def train (self):
        if self.Qbuffer.size() < batch_size:
            print("short")
            return

        #critic training
        states, actions, next_states, rewards, dones = self.Qbuffer.sample(batch_size) #buffer have 1d state

        #make states to batch cube states
        cube = get_cube(states, batch_size)
        next_cube = get_cube(next_states, batch_size)

        #now, states is cubic -> forwarding for each agent with no gradient
        #next_cube = array, thus first index represents depthwise

        with torch.no_grad():

            for i, target_network in enumerate([self.actor1_target, self.actor2_target, self.actor3_target, self.actor4_target]):
                #torch tensor right here
                next_action = target_network(torch.FloatTensor(next_cube[i])) #cutting state row-wise
                #print("next_action:", next_action)
                if i == 0:
                    next_actions = next_action
                else:
                    next_actions = torch.cat([next_actions, next_action], dim=1)

            #print("next:", next_actions.size())
            
            #now, make total state for critic 
            next_stata = flat_vectorize(next_states, batch_size) 
            target_q = rewards + (1 - dones) * gamma * self.critic_target(next_stata, next_actions) #next_states = length 49
            #print("target Q:", target_q)
        #print("this:", states.size(), actions.size())
        stata = flat_vectorize(states, batch_size)
        actions = torch.FloatTensor(actions)
        current_q = self.critic(stata, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #actor trainig
        actions = []
        for j, actor in enumerate ([self.actor1, self.actor2, self.actor3, self.actor4]):
            action = actor(torch.FloatTensor(cube[j])) #arr_state
            
            actions.append(action)

        actor_actions = torch.cat(actions, dim=1)
        actor_loss = -self.critic(stata, actor_actions).mean()
        #print("actor loss:", actor_loss.item())


        for optimizer in (self.actor1_optimizer, self.actor2_optimizer, self.actor3_optimizer, self.actor4_optimizer):
            optimizer.zero_grad()
        actor_loss.backward()
        for optimizer in (self.actor1_optimizer, self.actor2_optimizer, self.actor3_optimizer, self.actor4_optimizer):
            try:
                optimizer.step()
            except:
                pass

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
            torch.save(actor.state_dict(), work_dir + "/actor" + str(i) + "_" + str(num)+str(today)+".pt")
            i+=1
        highest_speed = num

        print("success case returned, highest_speed:", highest_speed)




"""main"""

def main():
    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load environment
    #env = rl_env.contact_env.CONTACT_ENV()
    env = rl_env.contact_env.WALK_ENV()

    #define MACC agent
    agent = MACCagent(state_dim, actor_state_dim, action_dim)

    #for plot
    rewards_forplot = []
    dist_forplot = []
    

    #episode loop
    for episode in range(num_episodes):

        #pre-execution
        states, actions, rewards, next_states, dones = [], [], [], [], []
        old_reward = 0
        total_reward = 0
        success = 0
        timestep =0
        
        #initialize environment
        env.reset()
        state, action, next_state, reward, done_mask, success = env.step(np.zeros(action_dim))
        action = np.array(action)
        #state
        #next_state = np.array(next_state) #need to use tuple? state shape inhomogeneous



        """execute one episode"""
        while done_mask == 0:
            
            action = agent.action(state) #here, state = 0x1 problem occurred
            noise = OUNoise(action_dim).noise()
            #print("action & noise:", len(action), len(noise))
            action += noise*0.5 #noisy action returned
            
            state, action, next_state, reward, done_mask, success = env.step(action) #env returns: np ndarray
            #state = np.array(state)
            action = np.array(action)
            #next_state = np.array(next_state)
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
            if timestep%10 == 0:
                agent.train()
                #agent.train()
        
        #plot(rewards_forplot,dist_forplot, 1, 1)

        old_reward = total_reward

        #if success
        if success == 1:
            num = env.return_self_action_num()
            if total_reward>old_reward:
                agent.return_net(num)
                plot(rewards_forplot,dist_forplot, 1, 1)
            rewards_forplot, dist_forplot = [], []
            
        old_reward= 0
        rewards_forplot, dist_forplot = [], []
        print("episode end:", episode)

    print ("all episodes executed")
    plot(rewards_forplot,dist_forplot, 1, 1)




"""evaluation net"""

"""class eval_net():
    def __init__():
        super().__init__()

    def forward(self, state):
        
"""

class eval_net(nn.Module):
    def __init__(self, state_dim, action_dim, actor_path):
        super().__init__()
        self.actor1 = Actor(state_dim, action_dim)
        self.actor1.load_state_dict(torch.load(actor_path+"/actor1_832372024-11-30_10-50-56.pt", weights_only = True))
        self.actor2 = Actor(state_dim, action_dim)
        self.actor2.load_state_dict(torch.load(actor_path+"/actor2_832372024-11-30_10-50-56.pt", weights_only = True))
        self.actor3 = Actor(state_dim, action_dim)
        self.actor3.load_state_dict(torch.load(actor_path+"/actor3_832372024-11-30_10-50-56.pt", weights_only = True))
        self.actor4 = Actor(state_dim, action_dim)
        self.actor4.load_state_dict(torch.load(actor_path+"/actor4_832372024-11-30_10-50-56.pt", weights_only = True))

    def forward(self, state):
        #divide state with cube, forward and return action array
        action_list = []
        state_list = divide_state(state)

        for i, actor in enumerate((self.actor1, self.actor2, self.actor3, self.actor4)):
            action_temp = actor(torch.FloatTensor(state_list[i])).detach().numpy()
            action_list.append(action_temp)
        return np.concatenate(action_list)


def eval():

    actor_path = "C:/kisang/Ant_control/result_macc"
    #dev_path = os.path.join(work_dir, "dev_368_2024-10-31_15-48-11.pt")

    i=0
    agent = eval_net(actor_state_dim, action_dim, actor_path)

    #agent = eval_net(state_dim, action_dim, actor_path)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        while viewer.is_running():

            time.sleep(0.001)# for stable rendering
            forcetorque = np.zeros(6)
            force = np.zeros(3)
            if data.ncon==0:
                pass
            else:
                for j, c in enumerate(data.contact):
                    mujoco.mj_contactForce(model, data, j, forcetorque)
                    force += forcetorque[0:3]

            observation = [data.qpos, data.qvel, data.qacc, force]
            action = agent(observation)
            print(action)
            data.ctrl = action
            mujoco.mj_step(model, data)


            i+=1
            if (i%100 == 0):
                print(i, "steps")
                #print("pitch:", pitch)
                #print ("roll:", roll)
            #print("step")
            viewer.sync()

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 1



if __name__ == '__main__':
    main()

    #eval()
