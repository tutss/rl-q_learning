import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# parameters
lr = 0.1
gamma = 0.9
episodes = 100_000
show_every = 10_000
epsilon = 0.7
epsilon_decay_value = 0.9998

def plot_metrics():
    plt.figure(figsize=(15,10))
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
    plt.legend(loc=4)
    plt.show()
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['eps'], label='eps')
    plt.show()

def q_learning(q_table, state_action, new_state, reward):
    max_q = np.max(q_table[new_state]) if new_state != -1 else 0
    current_q = q_table[state_action]
    new_q = (1-lr) * current_q + lr * (reward + gamma * max_q)
    q_table[state_action] = new_q

env = gym.make('Taxi-v2')

print("Epsilon decay value = {}".format(epsilon_decay_value))
# env info
print("Observation space = {}".format(env.observation_space))
print("Action space = {}".format(env.action_space))
print("Number of available actions = {}".format(env.action_space.n))

# discrete values
size = [500]

# creating q table
q_table = np.zeros((size + [env.action_space.n]))

print("Q-Table shape = {}".format(q_table.shape))

ep_rewards = [] # ep rewards
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'eps': []}

# for plotting
epsilons = list()

for ep in range(1, episodes):
    epsilons.append(epsilon)
    ep_reward = 0
    
    render = False
    if ep % show_every == 0:
        print("===> Episode {}".format(ep))
        # render = True
    
    # initial state
    state = env.reset()
    while True:
        # acting greedy sometimes (exploration)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        ep_reward += reward
        state_action = (state, action)
        state = new_state
        if render: 
            env.render()
        if not done:
            q_learning(q_table, state_action, new_state, reward)
        else: #new_state[0] >= env.goal_position: # reached terminal state
            q_learning(q_table, state_action, -1, reward)
            break

    epsilon *= epsilon_decay_value
    
    ep_rewards.append(ep_reward)
    if not ep % show_every:
        avg_reward = sum(ep_rewards[-show_every:])/len(ep_rewards[-show_every:])
        aggr_ep_rewards['ep'].append(ep)
        aggr_ep_rewards['eps'].append(epsilon)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(np.min(ep_rewards[-show_every:]))
        aggr_ep_rewards['max'].append(np.max(ep_rewards[-show_every:]))
        print("===> Episode {}: avg - {} | min - {} | max - {} | eps - {}".format(ep, avg_reward, np.min(ep_rewards[-show_every:]), np.max(ep_rewards[-show_every:]), epsilon))

env.close()

# plotting
plt.figure(figsize=(15,10))
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['eps'], label='eps')
plt.show()