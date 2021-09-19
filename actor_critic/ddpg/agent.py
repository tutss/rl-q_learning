import os
import numpy as np

import torch as t
import torch.nn.functional as F

from critic import Critic
from actor import Actor
from ou_noise import OUActionNoise
from replay_buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, fc1_dims=400, fc2_dims=300, 
                gamma=0.99, mem_max_size=1000000, batch_size=64):
        
        seed = 10
        np.random.seed(seed)
        t.manual_seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(mem_max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.actor = Actor(alpha, input_dims, n_actions, checkpoint_filename='actor', 
                                fc1_units=fc1_dims, fc2_units=fc2_dims)
        self.critic = Critic(beta, input_dims, n_actions, checkpoint_filename='critic', 
                                fc1_units=fc1_dims, fc2_units=fc2_dims)
        self.target_actor = Actor(alpha, input_dims, n_actions, checkpoint_filename='target_actor', 
                                    fc1_units=fc1_dims, fc2_units=fc2_dims)
        self.target_critic = Critic(beta, input_dims, n_actions, checkpoint_filename='target_critic', 
                                    fc1_units=fc1_dims, fc2_units=fc2_dims)

        self.update_network(tau=1)

    def choose_action(self, obs) -> np.array:
        self.actor.eval()
        state = t.tensor([obs], dtype=t.float)
        mu = self.actor(state)
        mu_prime = mu + t.tensor(self.noise(), dtype=t.float)
        self.actor.train()
        return mu_prime.detach().numpy()[0]

    def store_transition(self, state, action, reward, nstate, done) -> None:
        self.memory.store_transition(state, action, reward, nstate, done)

    def save_models(self) -> None:
        self.actor.save_chk()
        self.critic.save_chk()
        self.target_actor.save_chk()
        self.target_critic.save_chk()

    def load_models(self) -> None:
        self.actor.load_chk()
        self.critic.load_chk()
        self.target_actor.load_chk()
        self.target_critic.load_chk()

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return

        s, a, ns, r, done = self.memory.sample(self.batch_size)
        s = t.tensor(s, dtype=t.float)
        a = t.tensor(a, dtype=t.float)
        r = t.tensor(r, dtype=t.float)
        ns = t.tensor(ns, dtype=t.float)
        done = t.tensor(done)

        target_actions = self.target_actor(ns)
        target_critic_val = self.target_critic(ns, target_actions)
        critic_val = self.critic(s, a)

        target_critic_val[done] = 0.0
        target_critic_val = target_critic_val.view(-1)

        target = r + self.gamma*target_critic_val
        target = target.view(self.batch_size, 1)

        # loss critic
        self.critic.opt.zero_grad()
        critic_loss = F.mse_loss(target, critic_val)
        critic_loss.backward()
        self.critic.opt.step()

        # loss actor
        self.actor.opt.zero_grad()
        actor_loss = -self.critic(s, self.actor(s))
        actor_loss = t.mean(actor_loss)
        actor_loss.backward()
        self.actor.opt.step()

        self.update_network()

    def update_network(self, tau=None):
        if tau == None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)

