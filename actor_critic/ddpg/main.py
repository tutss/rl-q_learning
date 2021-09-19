import gym
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from agent import Agent
import random
import time

def plot_curve(scores, file):
    x = [i+1 for i in range(len(scores))]

    avg = np.zeros(len(scores))
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i-100): i+1])

    plt.plot(x, avg)  # otherwise the right y-label is slightly clipped
    plt.title('Running avg of previous 100 scores')
    plt.show()
    plt.savefig(file)

def main():
    seed = 10
    np.random.seed(seed)
    t.manual_seed(seed)
    random.seed(seed)

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001,
                batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=env.action_space.shape[0])

    n_games = 1001
    filename = f'plots/LunarLanderContinuous_a{str(agent.alpha)}_b{str(agent.beta)}.png'

    best_score = env.reward_range[0]
    scores = []
    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            nobs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, nobs, done)
            agent.learn()
            score += reward
            obs = nobs
        scores.append(score)
        
        if i % 20 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i} | Avg score = {avg_score} |')
            if avg_score > best_score:
                best_score = avg_score
        
        if i % 500 == 0:
            agent.save_models()
    
    plot_curve(scores, filename)

main()
        