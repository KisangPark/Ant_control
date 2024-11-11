import torch
from torch.distributions import MultivariateNormal, Normal
import numpy as np

action_size = 8
batch_size = 64

actions = np.random.rand(1,action_size) #batch_size,
actions =  torch.FloatTensor(actions)
#print(actions)

std_dev = torch.full((1,action_size), 0.5)
print(std_dev, std_dev.size())
cov_mat = torch.diag(torch.full((action_size,), 0.5))
#print(torch.full((action_size,),0.5), cov_mat)
#print(cov_mat.size())
print(actions.size())
#dist = MultivariateNormal(actions, cov_mat)
dist = Normal(actions, std_dev)

#print (cov_matrix)
#print(dist) # distribution! when sample the value is made.

samples = dist.sample()
print(samples)

