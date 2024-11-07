#DDPG algorithm application
"""
DDPG

1) Off-policy algorithm
2) High sample efficiency
3) MA environment

"""

import os
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import MultivariateNormal

import mujoco
import mujoco.viewer

from IPython import display
plt.ion()
os.environ["PYOPENGL_PLATFORM"] = 'egl'

import rl_env.env


# Hyperparameters
num_episodes = 1000
learning_rate = 0.0005
gamma = 0.99
tau = 0.005
batch_size = 64
buffer_size = 100000
num_epochs = 10

state_dim = 29
action_dim = 8

model = mujoco.MjModel.from_xml_path('ant_with_goal.xml') #xml file changed
data = mujoco.MjData(model)
state_dim = len(data.qpos) + len(data.qvel)

highest_speed = 1000 # maximum steps

work_dir = "/home/kisang-park/Ant_control/result_files"



def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s


def plot(reward, episode, flag):
    if episode%10 == 0:

        plt.figure(2)
        plt.cla() #delete all
        #durations_t = torch.tensor(laptimes, dtype=torch.float) #torch float tensor for duration
        plt.title('Result plot')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.plot(reward) # torch tensor to numpy
        plt.pause(0.01)

    if flag==1:
        plt.savefig('result_plot.png')

"""pre code"""


"""
Network, Buffer def

Actor: forward (state_dim, action_dim)
Critic: forward (state_dim, action_dim)
buffer: add, sample, size
"""

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)



"""DDPG agent definition"""

"""
1. initialize

actor & actor target network init (+ optimizer)
*****critic network input: state + action*****
load actor network state dictionary to target network
[critic same]
replay buffer define, max action (그냥 전역변수 쓰면 안되나?)

2. select_action
put state to actor network, take it to cpu, numpy, flatten

3. train
if buffer too small, return
- sample from replay buffer (returning state, action, reward, next_state, done)
- critic learning: target_net forwarding & calculate target Q value
                    compare with critic forwarding // update critic with backward
- actor learning
    in dpg, parameter to Q gradient = parameter to action X action to Q
    action to Q parameter already calculated & updated in critic update
    therefore, using Q mean of actor network output, can update the parameter to action that is policy update

4. main
environment & dimension & agent declare
in episodes, if not done env step

change in main -> episode execution & buffer add
                    after one episode, epoch training
"""

class DDPGAgent:
    def __init__(self, state_dim, action_dim): #max_action
        self.actor = Actor(state_dim, action_dim, max_action)#.to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action)#.to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim)#.to(device)
        self.critic_target = Critic(state_dim, action_dim)#.to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        #self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)#.to(device)
        return self.actor(state).numpy().flatten()
        #return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * GAMMA * self.critic_target(next_states, next_actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


    def return_net(self, num): #returning network parameters
        
        today = get_today()

        torch.save(self.actor.state_dict(), work_dir + "/actor" + "_" + str(num) + "_" + str(today) + ".pt")
        torch.save(self.critic.state_dict(), work_dir + "/critic" + "_" + str(num) + "_" + str(today) + ".pt")
        
        highest_speed = num

        print("success case returned, highest_speed:", highest_speed)


def main():
    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load environment
    env = rl_env.env.ANTENV()

    #define PPO agent
    agent = DDPGAgent(state_dim + action_dim)

    #for plot
    rewards_forplot = []

    #episode loop
    for episode in range(num_episodes):
        
        #pre-execution
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0
        success = 0
        
        #initialize environment
        env.reset()
        state, action, next_state, reward, done_mask, success = env.step(np.zeros(8))


        """execute one episode"""
        while done_mask == 0:
            
            action = agent.select_action(state)
            
            state, action, next_state, reward, done_mask, success = env.step(action)
            agent.replay_buffer.add(state, action, next_state,reward, done_mask)
            #noticed! -> can access to self variables using dot methods
            
            state = next_state
            total_reward += reward
        
        total_reward = total_reward*0.001
        print(episode, "Episode executed, total reward:", total_reward)

        rewards_forplot.append(total_reward)
        plot(rewards_forplot, episode, 0)


        #if success
        if success == 1:
            num = env.return_self_action_num()
            agent.return_net(num)
            plot(rewards_forplot, 1, 1)


        #train loop for epochs
        for i in range(num_epochs):
            agent.train()
            #print("train", i)
            #print(i, "epochs trained")
            #one episode trained
        #epoch number trained
        #print("episode trained")

    print ("all episodes executed")
    plot(rewards_forplot, 1, 1)
