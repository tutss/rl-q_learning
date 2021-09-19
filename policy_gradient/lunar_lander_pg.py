import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import gym
import numpy as np
import matplotlib.pyplot as plt

class PGAgent():
    def __init__(self, lr, in_dims=8, gamma=0.99, n_actions=4) -> None:
        self.gamma = gamma
        self.lr = lr
        self.in_dims = in_dims
        self.memory = list()
        self.action_memory = list()

        self.policy = nn.Sequential(
            nn.Linear(self.in_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.opt = optim.Adam(self.policy.parameters())

    def choose_action(self, obs):
        state = torch.from_numpy(obs)
        probs = F.softmax(self.policy(state))
        actions_probs = torch.distributions.Categorical(probs)
        action = actions_probs.sample()
        log_probs = actions_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.memory.append(reward)

    def learn(self):
        self.opt.zero_grad()
        G = np.zeros_like(self.memory)
        for t in range(len(self.memory)):
            g_sum = 0
            discount = 1
            for k in range(t, len(self.memory)):
                g_sum += self.memory[k] * discount
                discount *= self.gamma
            G[t] = g_sum
        G = torch.from_numpy(G)

        loss = torch.zeros(1, requires_grad=True)
        for g, logprob in zip(G, self.action_memory):
            loss = loss + -g * logprob
        loss.backward()
        self.opt.step()

        self.memory = list()
        self.action_memory = list()

def plot_curve(scores, x, file):
    avg = np.zeros(len(scores))
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i-100): i+1])

    plt.plot(x, avg)  # otherwise the right y-label is slightly clipped
    plt.title('Running avg of previous 100 scores')
    plt.show()
    plt.savefig(file)

def main():
    env = gym.make('LunarLander-v2')
    episodes = 3000
    lr = 0.0005
    agent = PGAgent(lr)
    filename = f'PG_lr{lr}_eps{episodes}.png'

    scores = []
    for i in range(episodes):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            obs = obs_
            # env.render()
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        if i % 300 == 0:
            print(f'episode {i}: score -> {score}, avg (100 games) -> {avg_score}')

    x = range(len(scores))
    plot_curve(scores, x, filename)

main()