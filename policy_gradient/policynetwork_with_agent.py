import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available())
        
    def forward(self, obs):
        state = torch.Tensor(obs)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Agent(object):
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4, 
            l1_size=256, l2_size=256):
        self.gamma = gamma
        self.rewards_history = list()
        self.actions_history = list()
        self.policy = PolicyNetwork(lr, input_dims, l1_size, l2_size, n_actions)
        
    def choose_action(self, obs):
        probs = F.softmax(self.policy.forward(obs))
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample() # tensor
        log_probs = action_probs.log_prob(action)
        self.actions_history.append(log_probs)
        return action.item() # integer
    
    def store_reward(self, reward):
        self.rewards_history.append(reward)
        
    def learn(self):
        self.policy.optimizer.zero_grad()
        G = torch.zeros_like(torch.Tensor(self.rewards_history), dtype=torch.float)
        for t in range(len(self.rewards_history)): # t timesteps
            G_sum = 0
            discount = 1
            for k in range(t, len(self.rewards_history)):
                G_sum += self.rewards_history[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = (G - torch.mean(G)) / torch.std(G)
#         G = torch. (G)
        
        loss = torch.zeros(1, requires_grad=True)
        for g, logprob in zip(G, self.rewards_history):
            loss = loss + -g * logprob
            
        loss.backward()
        self.policy.optimizer.step()
        
        self.rewards_history = list()
        self.actions_history = list()