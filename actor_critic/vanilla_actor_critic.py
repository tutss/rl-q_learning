import sys

sys.path.append('~/repos/tutss/rl-q_learning/')
from utils.utils import plot_curve

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 128)
        self.fc4 = nn.Linear(128, 64)
        
        self.pi = nn.Linear(64, n_actions)
        self.v = nn.Linear(64, 1)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)


class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
                 gamma=0.99):
        self.gamma = gamma
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = torch.tensor([observation])
        probabilities, _ = self.actor_critic(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.opt.zero_grad()

        state = torch.tensor([state])
        state_ = torch.tensor([state_])
        reward = torch.tensor(reward)

        _, critic_value = self.actor_critic(state)
        _, critic_value_ = self.actor_critic(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.opt.step()


def main():
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    episodes = 2000

    agent = Agent(gamma=0.99, lr=5e-6, input_dims=[8], n_actions=4,
                  fc1_dims=512, fc2_dims=256)

    filename = f'actor_critic_{env_name}_{agent.fc1_dims}_{agent.fc2_dims}_{episodes}.png'
    filename = f'pics/{filename}'
    
    scores = []
    for i in range(episodes):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, reward, obs_, done)
            obs = obs_
        scores.append(score)
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i} | Avg score for last 100 games: {avg_score}')
    plot_curve(scores, filename)
    

main()