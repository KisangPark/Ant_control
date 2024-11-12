import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the neural network for the policy

class PolicyNetwork(nn.Module):
def **init**(self, state_dim, action_dim):
super(PolicyNetwork, self).**init**()
self.fc1 = nn.Linear(state_dim, 128)
self.fc2 = nn.Linear(128, 128)
self.fc3 = nn.Linear(128, action_dim)

```
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

```

# Define the neural network for the value function

class ValueNetwork(nn.Module):
def **init**(self, state_dim):
super(ValueNetwork, self).**init**()
self.fc1 = nn.Linear(state_dim, 128)
self.fc2 = nn.Linear(128, 128)
self.fc3 = nn.Linear(128, 1)

```
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

```

# Hyperparameters

num_episodes = 1000
timesteps_per_episode = 1000
learning_rate = 0.001
gamma = 0.99
epsilon = 0.2
batch_size = 64

# Load MuJoCo model

model = mujoco.MjModel.from_xml_path('path_to_your_ant_model.xml')
data = mujoco.MjData(model)

# Initialize the policy and value networks and their optimizers

state_dim = model.nq + model.nv  # Number of state variables (qpos + qvel)
action_dim = [model.nu](http://model.nu/)  # Number of control signals (joint controls)
policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

def calculate_reward(data):
# Implement your reward function here
return data.reward  # Example placeholder

# Training loop

for episode in range(num_episodes):
mujoco.mj_resetData(model, data)
states, actions, rewards, next_states, dones = [], [], [], [], []

```
total_reward = 0

for t in range(timesteps_per_episode):
    # Get the current state
    state = np.concatenate([data.qpos, data.qvel])
    state_tensor = torch.FloatTensor(state)

    # Get control signals (actions) from the policy
    with torch.no_grad():
        control_signals = policy_net(state_tensor).numpy()

    # Apply control signals
    data.ctrl[:] = control_signals

    # Step the simulation
    mujoco.mj_step(model, data)

    # Store experience
    next_state = np.concatenate([data.qpos, data.qvel])
    reward = calculate_reward(data)
    done = t == timesteps_per_episode - 1

    states.append(state)
    actions.append(control_signals)
    rewards.append(reward)
    next_states.append(next_state)
    dones.append(done)

    total_reward += reward

# Compute advantages and returns
rewards = np.array(rewards)
returns = np.zeros_like(rewards)
advantages = np.zeros_like(rewards)

# Generalized Advantage Estimation (GAE)
for t in reversed(range(len(rewards))):
    next_value = 0 if dones[t] else value_net(torch.FloatTensor(next_states[t])).item()
    delta = rewards[t] + gamma * next_value - value_net(torch.FloatTensor(states[t])).item()
    ## delta: Q value difference from value_net
    advantages[t] = delta
    returns[t] = rewards[t] + gamma * (returns[t + 1] if t + 1 < len(rewards) else 0)

# Convert to tensors
states_tensor = torch.FloatTensor(states)
actions_tensor = torch.FloatTensor(actions)
returns_tensor = torch.FloatTensor(returns)
advantages_tensor = torch.FloatTensor(advantages)

# Update policy and value networks
for _ in range(10):  # Update for several epochs
    indices = np.random.permutation(len(states_tensor))
    for start in range(0, len(states_tensor), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        batch_states = states_tensor[batch_indices]
        batch_actions = actions_tensor[batch_indices]
        batch_advantages = advantages_tensor[batch_indices]
        batch_returns = returns_tensor[batch_indices]

        # Compute the ratio
        old_log_probs = policy_net(batch_states).gather(1, batch_actions.unsqueeze(-1)).log()
        new_log_probs = policy_net(batch_states).gather(1, batch_actions.unsqueeze(-1)).log()
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Compute the surrogate loss
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

print(f'Episode {episode + 1}: Total Reward = {total_reward}')

```

# Save the trained policy

torch.save(policy_net.state_dict(), 'ppo_policy.pth')
torch.save(value_net.state_dict(), 'ppo_value.pth')










"""
Continuous action space
"""

import torch
import torch.nn.functional as F

class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc_mu = torch.nn.Linear(128, action_dim)  # Mean of the Gaussian
        self.fc_sigma = torch.nn.Linear(128, action_dim)  # Standard deviation of the Gaussian

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))  # Ensure std dev is positive
        return mu, sigma

def calculate_ppo_ratios(states, actions, advantages, policy_net_old, policy_net_new):
    # Get old mean and std dev
    old_mu, old_sigma = policy_net_old(states)
    old_probs = torch.distributions.Normal(old_mu, old_sigma)
    old_log_probs = old_probs.log_prob(actions).sum(dim=-1)  # Log probability for continuous actions

    # Get new mean and std dev
    new_mu, new_sigma = policy_net_new(states)
    new_probs = torch.distributions.Normal(new_mu, new_sigma)
    new_log_probs = new_probs.log_prob(actions).sum(dim=-1)  # Log probability for continuous actions

    # Calculate ratios
    ratios = torch.exp(new_log_probs - old_log_probs)

    return ratios, old_log_probs, new_log_probs

# Example usage
if __name__ == "__main__":
    # Initialize policy networks
    input_dim = 4  # Example state dimension
    action_dim = 2  # Example continuous action dimension

    policy_net_old = PolicyNetwork(input_dim, action_dim)
    policy_net_new = PolicyNetwork(input_dim, action_dim)

    # Example data
    states = torch.rand((10, input_dim))  # 10 samples of states
    actions = torch.rand((10, action_dim))  # Random continuous actions taken
    advantages = torch.rand((10,))  # Random advantages

    # Calculate PPO ratios
    ratios, old_log_probs, new_log_probs = calculate_ppo_ratios(states, actions, advantages, policy_net_old, policy_net_new)

    print("Ratios:", ratios)
    print("Old Log Probabilities:", old_log_probs)
    print("New Log Probabilities:", new_log_probs)
