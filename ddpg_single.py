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

Actor: state_dim, action_dim
Critic: state_dim, action_dim
buffer: max_size, learning params, batch_size
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
