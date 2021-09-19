import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):

    def __init__(self, lr, input_shape, n_actions, checkpoint_filename, 
                fc1_units=400, fc2_units=300, chpt_dir='tmp/ddpg'):
        super(Actor, self).__init__()

        seed = 10
        np.random.seed(seed)
        t.manual_seed(seed)

        self.input_shape = input_shape
        self.lr = lr
        self.checkpoint_file = os.path.join(chpt_dir, checkpoint_filename + '_ddpg')

        # Network
        self.fc1 = nn.Linear(*self.input_shape, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn1 = nn.LayerNorm(fc1_units) # instead of Batch Norm
        self.bn2 = nn.LayerNorm(fc2_units)

        self.mu = nn.Linear(fc2_units, n_actions)

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, s):
        x = self.fc1(s)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return t.tanh(self.mu(x)) # [-1, 1]

    def save_chk(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_chk(self):
        self.load_state_dict(t.load(self.checkpoint_file))