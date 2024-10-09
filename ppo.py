import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mujoco
import mujoco.viewer

import rl_env.env


# Hyperparameters
num_episodes = 50
learning_rate = 0.001
gamma = 0.99
epsilon = 0.
batch_size = 64
num_epochs = 10

model = mujoco.MjModel.from_xml_path('ant.xml')
data = mujoco.MjData(model)
#model info needed for functions
state_dim = len(data.qpos) + len(data.qvel)

#for torch.save
highest_speed = 1000

work_dir = "home/mujoco/result_files"


"""
timestep is checked in the environment...

PPO : on-policy algorithm -> no replay buffer & target network needed..
"""

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s


class policy_network(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
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

"""
Policy network: qpos & qvel input using state_dim, 8 control signal output
tanh activation function to get value -1~1

Value network: qpos input, 1 Q value output
"""

def train(state, next_state, reward, done_mask,
returns, advantages, policy_net, value_net, policy_optimizer, value_optimizer):
    #codes for training
    #state(np array), action (array), next state, reward(single value), done mask list return
    indices = np.random.permutation(len(state_list_tensor))

    #make batches
    for start in range(0, len(states_tensor), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        #all torch tensors
        batch_states = state[batch_indices]
        #batch_actions = action[batch_indices]
        batch_advantages = advantages[batch_indices]
        batch_returns = returns[batch_indices]

        #compute ratio, surrogate loss, clipped surrogate loss
        # 8 action space -> 8 times repeat, column-wise

        old_log_signal = policy_net(batch_states)
        new_log_signal = policy_net(batch_states)

        for column in range(8):
            
            index = np.zeros(8, len(batch_indices))
            index[:][column] = 1
            #make index matrix

            old_log_probs = old_log_signal.gather(1, index).log()
            new_log_probs = new_log_signal.gather(1, index).log()
        
            ratios = torch.exp(new_log_probs - old_log_probs)

            surrogate_loss = ratios * batch_advantages
            clipped_surrogate_loss = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * batch_advantages
            policy_loss = -torch.mean(torch.min(surrogate_loss, clipped_surrogate_loss))

            # Update policy network
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Update value network
            value_loss = F.mse_loss(value_net(batch_states), batch_returns)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()




def main():
    #just training, no rendering

    #load environment
    env = rl_env.env.ANTENV()
    
    #define all the networks & optimizers needed
    policy_net = policy_network(state_dim)
    value_net = value_network(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)


    #episode loop

    for episode in range(num_episodes): 
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
        total_reward = 0

        env.reset()
        state, action, next_state, reward, done_mask = env.step(np.zeros(8))
        # append values during timesteps -> no from envs... multi agents no!!

        """execute one episode"""
        while done_mask == 0:

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done_mask) # needed?? 

            state_tensor = torch.FloatTensor(next_state)

            with torch.no_grad():
                policy_action = policy_net(state_tensor).numpy() # torch tensor to np

            state, action, next_state, reward, done_mask, success = env.step(policy_action)

            total_reward += reward

        print("Episode, total reward:", total_reward)

        #now one episode finished, list returned
        #returns, GAE (advantages), convert to tensors -> and then train (for epoch, batch)

        """check success"""

        if success == 1:
            
            today = get_today()
            num = env.return_self_action_num()

            if num < highest_speed:

                torch.save(policy_net, work_dir + "/policy" + "_" + num + "_" + today + ".pt")
                torch.save(value_net, work_dir + "/value" + "_" + num + "_" + today + ".pt")

                highest_speed = num

        # initialize returns & rewards
        reward_list = np.array(reward_list)
        returns = np.zeros_like(reward_list)
        advantages = np.zeros_like(reward_list)

        #Generalized Advantage Estimation (GAE)
        #calculate advantages and returns
        for i in reversed(range(len(rewards))):

            next_value = 0 if done_list[i] else value_net(torch.FloatTensor(next_state_list[i])).item() # why item()?
            delta = reward_list[i] + gamma*next_value - value_net(torch.FloatTensor(state_list[i])).item()
            #delta: Q value difference from value net

            advantages[i] = delta
            returns[i] = reward_list[i] + gamma*(returns[i+1] if t+1 < len(rewards) else 0)

        state_list_tensor = torch.FloatTensor(state_list)
        action_list_tensor = torch.FloatTensor(action_list)
        next_state_tensor = torch.FloatTensor(next_state_list)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)

        #now train
        for _ in range(num_epochs):

            train(state_list_tensor, next_state_tensor, reward_list, done_list,
            returns_tensor, advantages_tensor, policy_net, value_net, policy_optimizer, value_optimizer)
            
            #one episode trained
        #epoch number trained

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


def main_render():
    return 0
    #training with rendering, for check
    #advantages_tensor

def eval():
    return 0
    #test the torch model
    #evaluation


if __name__ == '__main__':
    main()
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