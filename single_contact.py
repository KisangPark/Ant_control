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
learning_rate = 0.001 #0.0001
gamma = 0.99
TAU = 0.1
batch_size = 64 #64
buffer_size = 500000
num_epochs = 2

state_dim = 65 # plus goal, contact force
action_dim = 8 #each agent

model = mujoco.MjModel.from_xml_path('C:/kisang/Ant_control/rl_env/ant_box.xml') #xml file changed
data = mujoco.MjData(model)

highest_speed = 5000 # maximum steps

work_dir = "C:/kisang/Ant_control/result_single_contact"
#C:/kisang/Ant_control/result_macc
#/home/kisang-park/Ant_control/result_macc
#C:/Users/gpu/kisang/Ant_control/rl_env



def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
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

    if batch_size == 1:
        temp = []
        qpos, qvel, qacc, force = state
        for q in (qpos, qvel, qacc, force):
            for item in q:
                temp.append(item)
        return torch.FloatTensor(temp)

    for i in range(batch_size):
        temp = []
        qpos, qvel, qacc, force = state[i]
        
        for q in (qpos, qvel, qacc, force):
            for item in q:
                temp.append(item)

        flattened.append(temp)
        #print("flattened, temp:", len(flattened), len(temp))

    flattened = torch.FloatTensor(flattened)
    return flattened



"""pre code"""

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim) #not able to use action dim

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
            states,#torch float tensor?
            torch.FloatTensor(actions),
            next_states,
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(dones).unsqueeze(1)#,
            ) #states not tensor because array

    def size(self):
        return len(self.buffer)


class SINGLECONTACTagent:
    def __init__ (self, state_dim, action_dim): #actor state dimension

        #critic part
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        #actor part - 4 actors
        self.actor = Actor(state_dim, action_dim)#.to(device)
        self.actor_target = Actor(state_dim, action_dim)#.to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        #buffer
        self.Qbuffer = ReplayBuffer(buffer_size)

        self.highest_speed = 5000


    def action (self, in_state): #multiple actors!!

        output = []

        state = flat_vectorize(in_state, 1)

        #state = torch.FloatTensor(state) #-> already tensor
        out = self.actor(state).detach().numpy()
        
        return out
        

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

        #states vectorized -> torch tensor
        states = flat_vectorize(states, batch_size)
        next_states = flat_vectorize(next_states, batch_size)

        with torch.no_grad():

            next_actions = self.actor_target(next_states) #cutting state row-wise

            #now, make total state for critic 
            target_q = rewards + (1 - dones) * gamma * self.critic_target(next_states, next_actions)
        #print("this:", states.size(), actions.size())
        actions = torch.FloatTensor(actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #actor training
        current_action = self.actor(states)
        actor_loss = -self.critic(states, current_action).mean()
        #print("actor loss:", actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        #target update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


    def return_net(self, num): #returning network parameters
        
        today = get_today()

        torch.save(self.actor.state_dict(), work_dir + "/actor" + "_" + str(num) + "_" +str(today)+".pt")
        highest_speed = num

        print("success case returned, highest_speed:", highest_speed)




"""main"""

def main():
    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load environment
    #env = rl_env.contact_env.CONTACT_ENV()
    env = rl_env.contact_env.WALK_ENV()

    #define SINGLECONTACTagent agent
    agent = SINGLECONTACTagent(state_dim, action_dim)

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
            action += noise #noisy action returned
            
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

        #if success
        if success == 1:
            num = env.return_self_action_num()
            agent.return_net(num)
            plot(rewards_forplot,dist_forplot, 1, 1)
            rewards_forplot, dist_forplot = [], []

        rewards_forplot, dist_forplot = [], []
        print("episode end:", episode)

    print ("all episodes executed")
    plot(rewards_forplot,dist_forplot, 1, 1)




class eval_net(nn.Module):
    def __init__(self, state_dim, action_dim, actor_path):

        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.actor.load_state_dict(torch.load(actor_path+"/actor_32693_2024-11-30_15-03-11.pt", weights_only = True))
       
    def forward(self, state):
        #divide state with cube, forward and return action array
        state_list = flat_vectorize(state, 1)

        action = self.actor(state_list).detach().numpy()
        return action


def eval():

    actor_path = "C:/kisang/Ant_control/result_single_contact"
    #dev_path = os.path.join(work_dir, "dev_368_2024-10-31_15-48-11.pt")

    i=0
    agent = eval_net(state_dim, action_dim, actor_path)

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
            #print(action)
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
