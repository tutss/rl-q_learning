import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Critic(nn.Module):

    def __init__(self, lr, input_shape, n_actions, checkpoint_filename, 
                fc1_units=400, fc2_units=300, chpt_dir='tmp/ddpg'):
        super(Critic, self).__init__()

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
        self.action_value = nn.Linear(n_actions, fc2_units)
        self.q = nn.Linear(fc2_units, 1)

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1/np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)
        
        self.opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)

    def forward(self, s, a):
        x = self.fc1(s)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        y = self.action_value(a)

        xy = F.relu(t.add(x, y))

        return self.q(xy)

    def save_chk(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_chk(self):
        self.load_state_dict(t.load(self.checkpoint_file))