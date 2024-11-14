#DDPG algorithm application
"""
DDPG

1) Off-policy algorithm
2) High sample efficiency
3) MA environment

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

import rl_env.env
from rl_env.OUNoise import OUNoise


# Hyperparameters
num_episodes = 1000
learning_rate = 0.0001
gamma = 0.99
TAU = 0.005
batch_size = 64 #64
buffer_size = 50000
num_epochs = 2

state_dim = 29
action_dim = 8

model = mujoco.MjModel.from_xml_path('ant_with_goal.xml') #xml file changed
data = mujoco.MjData(model)
#state_dim = len(data.qpos) + len(data.qvel)

highest_speed = 5000 # maximum steps

work_dir = "/home/kisang-park/Ant_control/result_files" 
#/home/kisang-park/Ant_control/result_files
#C:/Users/gpu/kisang/Ant_control/result_files


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
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

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
        #print ("thisthils", len(states), len(next_states))
        #print(states)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(next_states),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(dones).unsqueeze(1)#,
            )#returns 

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

2. action
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
        self.actor = Actor(state_dim, action_dim)#.to(device)
        self.actor_target = Actor(state_dim, action_dim)#.to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim)#.to(device)
        self.critic_target = Critic(state_dim, action_dim)#.to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.highest_speed = 5000
        #self.max_action = max_action

    def action(self, state):
        state = torch.FloatTensor(state) #.unsqueeze(0)#.to(device)
        return self.actor(state).detach().numpy()
            #return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if self.replay_buffer.size() < batch_size:
            print("short")
            return

        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size) #order...
        #print("sampled")
        #return 1x64 array, each element is np ndarray

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states) #torch tensor of list of np nd array return
            #print("sizes:", states.size(), next_states.size(), next_actions.size())
            target_q = rewards + (1 - dones) * gamma * self.critic_target(next_states, next_actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        #print("critic:", critic_loss)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #print("backprop")

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        #print ("actor:", actor_loss)

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #print(self.replay_buffer.size())

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


    def return_net(self, num): #returning network parameters

        if num > self.highest_speed:
            return
        
        today = get_today()

        torch.save(self.actor.state_dict(), work_dir + "/actor" + "_" + str(num) + "_" + str(today) + ".pt")
        torch.save(self.critic.state_dict(), work_dir + "/critic" + "_" + str(num) + "_" + str(today) + ".pt")
        
        self.highest_speed = num

        print("********success case returned, highest_speed:", self.highest_speed, "********")

    def load_parameters(self):

        #need to return critic also
        self.actor.load_state_dict(torch.load(work_dir + "/success!!/" + "actor_587_2024-11-13_19-43-18.pt", weights_only = True))
        self.actor_target.load_state_dict(torch.load(work_dir + "/success!!/" + "actor_587_2024-11-13_19-43-18.pt", weights_only = True))

        self.critic.load_state_dict(torch.load(work_dir + "/success!!/" + "critic_587_2024-11-13_19-43-18.pt", weights_only = True))
        self.critic_target.load_state_dict(torch.load(work_dir + "/success!!/" + "critic_587_2024-11-13_19-43-18.pt", weights_only = True))


def main():
    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load environment
    env = rl_env.env.ANTENV()

    #define PPO agent
    agent = DDPGAgent(state_dim, action_dim)
    agent.load_parameters()

    #standard deviation
    std_dev = 0.1

    #episode loop
    for episode in range(num_episodes):

        #for plot
        rewards_forplot = []
        dist_forplot = []
        
        #pre-execution
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0
        success = 0
        timestep =0

        #Noise distrbution, preventing deadlock
        #distribution = Normal(0, std_dev)
        
        #initialize environment
        env.reset()
        state, action, next_state, reward, done_mask, success = env.step(np.zeros(8))
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        #reward = np.array(reward)
        #done_mask = np.array(done_mask)
        #print("env0:", state.shape, next_state.shape, reward.shape)


        """execute one episode"""
        while done_mask == 0:
            
            action = agent.action(state)
            #print("before:", action)
            noise = OUNoise(action_dim).noise()
            #noise = []
            #for i in range(8):
            #    noise.append(distribution.sample()) 
            action += noise
            #print("after:", action)
            
            state, action, next_state, reward, done_mask, success = env.step(action) #env returns: np ndarray
            state = np.array(state)
            action = np.array(action)
            next_state = np.array(next_state)
            #reward = np.array(reward)
            #done_mask = np.array(done_mask)
            #print("env1:", state.shape, next_state.shape)
            agent.replay_buffer.add(state, action, next_state,reward, done_mask)
            #noticed! -> can access to self variables using dot methods
            
            state = next_state
            total_reward += reward
            timestep+=1

            """training per timestep"""
            plot_reward = total_reward/timestep
            if timestep%100 == 0:
                print(timestep, "steped, total reward:", plot_reward)
            rewards_forplot.append(plot_reward)
            final_dist = env.return_dist()
            dist_forplot.append(final_dist)
            plot(rewards_forplot, dist_forplot, timestep, 0)

                #for num in range(num_epochs):
            #if timestep%10 == 0:
            agent.train()
        
        plot(rewards_forplot,dist_forplot, 1, 1)
        #total_reward = total_reward/timestep *0.001
        #print(episode, "Episode executed, total reward:", total_reward)

        #rewards_forplot.append(total_reward)
        #final_dist = env.return_dist()
        #dist_forplot.append(final_dist)
        #plot(rewards_forplot, dist_forplot, episode, 0)


        #test
        #if(episode == 100):
        #    num = env.return_self_action_num()
        #    agent.return_net(num)
        #    print("saved episode", episode)

        print("episode terminalized, timestep:", timestep)
        #if success
        if success == 1:
            num = env.return_self_action_num()
            agent.return_net(num)
            plot(rewards_forplot,dist_forplot, 1, 1)


        #train loop for epochs
        #for i in range(num_epochs):
        #    agent.train()
            #print("train", i)
            #print(i, "epochs trained")
            #one episode trained
        #epoch number trained
        #print("episode trained")

        #if (episode%100 == 0):
        std_dev = std_dev*0.9

    print ("all episodes executed")
    plot(rewards_forplot,dist_forplot, 1, 1)




"""evaluation net"""

class eval_net(nn.Module):
    def __init__(self, state_dim, action_dim, actor_path):
        super().__init__()

        self.actor_net = Actor(state_dim, action_dim)
        #self.critic_net = Critic(state_dim, action_dim)

        self.actor_net.load_state_dict(torch.load(actor_path, weights_only = True))
        #self.critic_net.load_state_dict(torch.load(critic_path, weights_only = True))

    def forward(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)

            action = self.actor_net(state_tensor)

            return action.numpy()


def eval():

    actor_path = os.path.join(work_dir, "actor_940_2024-11-14_12-28-36.pt")
    #dev_path = os.path.join(work_dir, "dev_368_2024-10-31_15-48-11.pt")

    i=0

    agent = eval_net(state_dim, action_dim, actor_path)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        while viewer.is_running():
            time.sleep(0.001)# for stable rendering
            qvel_equalized = data.qvel * 10
            qpos_eqalized = data.qpos *10
            state = np.concatenate((np.ndarray.flatten(qpos_eqalized), np.ndarray.flatten(qvel_equalized)))
            action = agent(state)
            data.ctrl = action
            mujoco.mj_step(model,data)

            #angle test
            w, x, y, z = data.qpos[3:7]
            pitch = np.arcsin(2.0*(w*y - z*x))
            roll = np.arctan2(2.0*(w*x+y*z), 1.0-2.0*(x*x + y*y))
            yaw = np.arctan2(2.0*(w*z+y*x), 1.0-2.0*(y*y + z*z))

            #print("dist:", np.sqrt(np.sum(np.square(data.qpos[0:2] - [15, 0]))))
            #dist correctly calculated

            i+=1
            if (i%100 == 0):
                print(i, "steps", data.qpos[2])
                print("pitch:", pitch)
                print ("roll:", roll)
            #print("step")
            viewer.sync()

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 1




if __name__ == '__main__':
    main()

    #eval()