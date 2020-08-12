import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def plot_metrics():
    plt.figure(figsize=(15,10))
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
    plt.legend(loc=4)
    plt.show()
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['eps'], label='eps')
    plt.show()

env = gym.make('Taxi-v2')

# parameters
lr = 0.1
gamma = 0.9
episodes = 100_000
show_every = 10_000

epsilon = 0.7
epsilon_decay_value = 0.9998

print(epsilon_decay_value)

# env info
print(env.observation_space)
print(env.action_space)
print(env.action_space.n)

# discrete values
size = [500]

# creating q table
q_table = np.random.uniform(low=-10, high=20, size=(size + [env.action_space.n]))

print(q_table.shape)

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
    discrete_state = env.reset()

    # frames
    frames = []

    while True:
        # acting greedy sometimes (exploration)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = new_state
        ep_reward += reward

        state_action = (discrete_state, action)
        
        discrete_state = new_discrete_state

        if render: 
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) # max value for the new state at t+1
            current_q = q_table[state_action] # getting current q value at t
            new_q = (1-lr) * current_q + lr * (reward + gamma * max_future_q) # q learning formula
            q_table[state_action] = new_q # update state-action at t with new q
        else: #new_state[0] >= env.goal_position: # reached terminal state
            q_table[state_action] = 0 # given that we initialize q_table with random variables
            break

    if end_epsilon_decay >= ep >= start_epsilon_decay:
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