"""Pre-code"""

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

import rl_env.env


# Hyperparameters
num_episodes = 50
learning_rate = 0.001
gamma = 0.99
epsilon = 0.2
batch_size = 64
num_epochs = 10
min_covariance = 0.0001

model = mujoco.MjModel.from_xml_path('ant.xml')
data = mujoco.MjData(model)

state_dim = len(data.qpos) + len(data.qvel)

highest_speed = 1000 # maximum steps

work_dir = "~/Ant_control/result_files"

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
        indices = np.random.permutation(len(states))

        for start in range(0, len(states), batch_size):

            end = start + batch_size
            batch_indices = indices[start:end]
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



"""main"""


def main():

    #load environment
    env = rl_env.env.ANTENV()

    #define PPO agent
    agent = PPOagent(state_dim)

    #episode loop
    for episode in range(num_episodes):
        
        states, actions, rewards, next_states, dones, old_probs = [], [], [], [], [], []
        total_reward = 0
        env.reset()

        state, action, next_state, reward, done_mask, success = env.step(np.zeros(8))
        #print(state)

        """execute one episode"""
        while done_mask == 0:
            policy_distribution = agent.action(state) #return action & qvalue
            policy_action = policy_distribution.sample().numpy()
            state, action, next_state, reward, done_mask, success = env.step(policy_action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done_mask)

            old_probs.append(policy_distribution.log_prob(torch.FloatTensor(action)).detach())

            total_reward += reward

        print("Episode, total reward:", total_reward)


        #if success
        if success == 1:
            
            today = get_today()
            num = env.return_self_action_num()

            if num < highest_speed:

                torch.save(policy_net, work_dir + "/policy" + "_" + num + "_" + today + ".pt")
                torch.save(value_net, work_dir + "/value" + "_" + num + "_" + today + ".pt")

                highest_speed = num



        #train loop for epochs

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones)
        old_probs = np.asarray(old_probs)

        for i in range(num_epochs):
            agent.train(states, actions, rewards, next_states, dones, old_probs)
            print(i, "epochs trained")
            #one episode trained
        #epoch number trained
        print("episode trained")

    print ("all episodes executed")



"""
1. meaning of advantage in PPO?
need to calculate loss of policy
policy loss should be weighted by the difference of q value -> advantage!

2. meaning of variables
next_value: next_state input, Qt+1 from value_network
delta: difference of Q value calculated & calculated from value network (correct answer - incorrect answer..)

then!!
-> advantages should be multiplied to the backprop of policy network
-> advantages and returns should be used to backprop the value network

"""


def eval():
    return 0
    #test the torch model
    #evaluation


if __name__ == '__main__':
    main()
    #main_render()
    #eval()





"""

def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 10개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
"""