"""Pre-code"""

from multiprocessing import Process, Pipe

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

import mujoco
import mujoco.viewer

os.environ["PYOPENGL_PLATFORM"] = 'egl'

import rl_env.env


# Hyperparameters
num_episodes = 30000
learning_rate = 0.0005
gamma = 0.99
epsilon = 0.2
batch_size = 64
num_epochs = 10
min_covariance = 0.7

model = mujoco.MjModel.from_xml_path('ant_with_goal.xml') #xml file changed
data = mujoco.MjData(model)

state_dim = len(data.qpos) + len(data.qvel)

highest_speed = 3000 # maximum steps

work_dir = "/home/kisang-park/Ant_control/result_files"

#timestep is checked in the environment...
#PPO : on-policy algorithm


def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

"""pre-code"""


"""Network Definition"""

class PolicyActNetwork(nn.Module):

    def __init__(self, state_dim):

        super(PolicyActNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) #from -1 to 1
        return x

class PolicyDevNetwork(nn.Module):

    def __init__(self, state_dim):

        super(PolicyDevNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) #for standard deviation
        return x

class QNetwork(nn.Module):

    def __init__(self, state_dim):

        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # for Q value
        return x


class value_network(nn.Module):

    def __init__(self, state_dim):

        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

"""Network Definition"""



"""PPO agent, for act and train"""

class PPOagent():

    def __init__(self, state_dim):

        self.act_net = PolicyActNetwork(state_dim)
        self.dev_net = PolicyDevNetwork(state_dim)
        self.value_net = QNetwork(state_dim)

        self.act_optimizer = optim.Adam(self.act_net.parameters(), lr=learning_rate)
        self.dev_optimizer = optim.Adam(self.dev_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

    def action (self, state): #state is np.ndarray
        #for action case, no need to multivariate normal... just normal with std_dev
        with torch.no_grad():

            state_tensor = torch.FloatTensor(state)

            mean = self.act_net(state_tensor)#.unsqueeze(dim=0).transpose(0,1)
            #positive definite
            deviation = self.dev_net(state_tensor)
            cov_matrix = torch.diag(deviation)
            cov_matrix= torch.mm(cov_matrix, cov_matrix.t()) + torch.eye(len(deviation))*min_covariance

        #calculate distribution
            #print(deviation.size(), mean.size())
            distribution = MultivariateNormal(mean, cov_matrix)
            #print(distribution)

            #action = distribution.sample()

            return distribution #, q_value
    
    #-> Pipe no need? parameters keep updated -> no... network separation

    def train(self, states, actions, rewards, next_states, dones, old_probs):

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        """calculate returns & advantages"""
        for i in reversed(range(len(rewards))):

            next_value = 0 if dones[i] else self.value_net(torch.from_numpy(next_states[i]).type(torch.FloatTensor)).item()
            delta = rewards[i] + gamma*next_value - self.value_net(torch.from_numpy(states[i]).type(torch.FloatTensor)).item()
            
            advantages[i] = delta
            returns[i] = rewards[i] + gamma*(returns[i+1] if i+1<len(rewards) else 0)
        

        """make batch"""
        batch_indices = np.random.permutation(batch_size)

        #for start in range(0, len(states), batch_size):

            #end = start + batch_size
            #batch_indices = indices[start:end]
        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]
        batch_old_probs = old_probs[batch_indices]
        batch_advantages = advantages[batch_indices]
        batch_returns = returns[batch_indices]

        #indent
        """Critic, Qnet backward"""
        #critic_val = self.value_net(torch.FloatTensor(batch_states)).detach().squeeze().numpy()
        #print(critic_val, batch_returns)
        criterion = nn.MSELoss()
        critic_loss = criterion(self.value_net(torch.FloatTensor(batch_states)), torch.FloatTensor(batch_returns).unsqueeze(1))
        
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()


        """GAE & surrogate loss"""
        losses = []
        batch_mean = self.act_net(torch.FloatTensor(batch_states))
        batch_dev = self.dev_net(torch.FloatTensor(batch_states))

        for i in range(len(batch_states)):
            cov_matrix = torch.diag(batch_dev[i])
            cov_matrix= torch.mm(cov_matrix, cov_matrix.t()) + torch.eye(len(batch_dev[i]))*min_covariance
            batch_dist = MultivariateNormal(batch_mean[i], cov_matrix)
            new_prob = batch_dist.log_prob(torch.FloatTensor(batch_actions[i]))
            ratio = torch.exp(new_prob - batch_old_probs[i])

            sur_loss = ratio * batch_advantages[i]
            clipped_loss = torch.clamp(ratio, 1-epsilon, 1+epsilon) * batch_advantages[i]
            #print(sur_loss, clipped_loss)
            loss = -torch.min(sur_loss, clipped_loss)
            #print(act_loss)
            losses.append(loss)
            #print(new_prob) # good values...

        act_loss = torch.mean(torch.stack(losses))
        #print(act_loss)
        """act_loss = -torch.mean(losses)
        print(act_loss)

        surrogate_loss = ratios * batch_advantages#.detach()
        clipped_loss = torch.clamp(rations, 1-epsilon, 1+epsilon) * batch_advantages#.detach
        act_loss = torch.mean(torch.min(surrogate_loss, clipped_loss))"""


        """act network backward"""
        self.act_optimizer.zero_grad()
        #losses.backward()
        act_loss.backward()
        self.act_optimizer.step()
            

    """PPO agent, for act and train"""

    def return_state_dict(self):
        return self.act_net.state_dict(), self.dev_net.state_dict() 


    def return_net(self, num):
        
        today = get_today()

            #if num < highest_speed:

        torch.save(self.act_net.state_dict(), work_dir + "/action" + "_" + str(num) + "_" + str(today) + ".pt")
        torch.save(self.dev_net.state_dict(), work_dir + "/dev" + "_" + str(num) + "_" + str(today) + ".pt")

        highest_speed = num

        print("success case returned, highest_speed:", highest_speed)




"""evaluation network, receive params & act"""

class eval_net(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.act_net = PolicyActNetwork(state_dim)
        self.dev_net = PolicyDevNetwork(state_dim)

    def forward(self, state):

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)

            mean = self.act_net(state_tensor)
            deviation = self.dev_net(state_tensor)
            cov_matrix = torch.diag(deviation)
            cov_matrix= torch.mm(cov_matrix, cov_matrix.t()) + torch.eye(len(deviation))*min_covariance

            distribution = MultivariateNormal(mean, cov_matrix)
            action = distribution.sample()

            return action.numpy()


    def get_param(self, a,b): # a, b is parameter variable

        self.act_net.load_state_dict(a)
        self.dev_net.load_state_dict(b) #torch.load(dev_path, weights_only = True)

    def return_net(self, num):
        
        today = get_today()

            #if num < highest_speed:

        torch.save(self.act_net.state_dict(), work_dir + "/action" + "_" + str(num) + "_" + str(today) + ".pt")
        torch.save(self.dev_net.state_dict(), work_dir + "/dev" + "_" + str(num) + "_" + str(today) + ".pt")

        highest_speed = num

        print("success case returned, highest_speed:", highest_speed)

    

"""main"""

def mujoco_stepping(conn, env):

    agent = eval_net(state_dim)

    print("agent:" ,agent) # OK

    #parameter setting
    param = conn.recv()
    #print("parameter received", param[0], param[1]) # OK, well received
    agent.get_param(param[0],param[1]) #impossible... no memory share


    #one episode, make appended lists 
    states, actions, rewards, next_states, dones, old_probs = [], [], [], [], [], []
    total_reward = 0
    env.reset()
    print("environment reset") #OK
    state, action, next_state, reward, done_mask, success = env.step(np.zeros(8))
    print(state) #29 states, OK
    state = np.zeros((29)) + 0.01

    """execute one episode"""
    while done_mask == 0:
        print("while sentence") #just once printed -> ?

        #problem in the code down -> because of the parameter sharing..
        policy_distribution = agent(state) #return action & qvalue
        print("forwarded")
        policy_action = policy_distribution.sample().numpy()
        state, action, next_state, reward, done_mask, success = env.step(policy_action) #mujoco step

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done_mask)

        old_probs.append(policy_distribution.log_prob(torch.FloatTensor(action)).detach())

        total_reward += reward
        print("one step") # NO... not printed
        
    total_reward = total_reward*0.0001

    print(episode, "Episode executed, total reward:", total_reward)

    conn.send([states, actions, rewards, next_states, dones, old_probs]) #pipe


    #if success
    if success == 1 or total_reward>3:

        num = env.return_self_action_num()

        agent.return_net(num)

    return



def main():

    """basic declaration: environment, ppo agent"""

    #load environment
    env = rl_env.env.ANTENV()

    #define PPO agent & action agent

    train_agent = PPOagent(state_dim)

    #act_agent = eval_net(state_dim)


    """Multi processing start"""

    #pipeline
    conn1, conn2 = Pipe(duplex = True) #conn1 for input, conn2 for output

    #create processes
    mujoco_step = Process(target = mujoco_stepping, args = (conn1, env))

    #initialize values

    act_path = os.path.join(work_dir, "action_3001_2024-10-17_13-30-20.pt")
    dev_path = os.path.join(work_dir, "dev_3001_2024-10-17_13-30-20.pt")
    p1 = torch.load(act_path, weights_only = True)
    p2 = torch.load(dev_path, weights_only = True)

    conn2.send([p1, p2])

    mujoco_step.start() # nothing to receive... stall
    mujoco_step.join() #no return -> 

    print("joint complete")
    
    sets = conn2.recv() # stall?
    print("received")
    states = sets[0]
    actions = sets[1]
    rewards= sets[2]
    next_states= sets[3]
    dones= sets[4]
    old_probs= sets[5]

    #loop
    for episode in range(num_episodes):
        """process start
        np array wise -> epoch train
        join, receive data
        conn2 send (parameters)
        """

        mujoco_step.start() # multi process

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones)
        old_probs = np.asarray(old_probs)

        for i in range(num_epochs):
            train_agent.train(states, actions, rewards, next_states, dones, old_probs)

        print("one episode trained")

        mujoco_step.join()

        sets = conn2.recv()
        states = sets[0]
        actions = sets[1]
        rewards= sets[2]
        next_states= sets[3]
        dones= sets[4]
        old_probs= sets[5]

        conn2.send(parameters)

        param1, param2 = train_agent.return_net

        conn2.send([param1, param2])

        


if __name__ == '__main__':
    main()

