import torch
from torch.distributions import MultivariateNormal
import numpy as np

action_size = 8
batch_size = 64

actions = np.random.rand(batch_size,action_size)
actions =  torch.FloatTensor(actions)
#print(actions)

#std_dev = torch.full((1,action_size), 0.5)
cov_mat = torch.diag(torch.full((action_size,), 0.5))
dist = MultivariateNormal(actions, cov_mat)

#print (cov_matrix)
#print(dist) # distribution! when sample the value is made.

samples = dist.sample()
print(samples)

