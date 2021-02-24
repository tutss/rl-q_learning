import gym
import numpy as np
import matplotlib.pyplot as plt

from policynetwork_with_agent import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    print(env.action_space)
    agent = Agent(lr=0.001, input_dims=[env.action_space.n*2], gamma=0.99, n_actions=env.action_space.n, l1_size=64, l2_size=32)
    score_history = list()
    n_episodes = 2500
    render_at = 500
    for i in range(1, n_episodes+1):
        if i % 100 == 0:
            render = True
            print(f'Episode {i}, score {score}')
        else:
            render = False
        done = False
        score = 0
        obs = env.reset()
        while not done:
            if render:
                env.render()
            action = agent.choose_action(obs)
            obs, reward, done, info = env.step(action)
            agent.store_reward(reward)
            score += reward
        score_history.append(score)
        agent.learn()
    
    filename = 'lunar-lander.png'
    plt.figure(figsize=(20,10))
    plt.plot(range(len(score_history)), score_history)
    plt.show()