import numpy as np

class ReplayBuffer():

    def __init__(self, mem_max_size, input_shape, n_actions) -> None:
        seed = 10
        np.random.seed(seed)

        self.size = mem_max_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.size, *input_shape))
        self.ns_memory = np.zeros((self.size, *input_shape))
        self.action_memory = np.zeros((self.size, n_actions))
        self.reward_memory = np.zeros(self.size)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool)

    def store_transition(self, s, a, r, ns, done):
        idx = self.mem_count % self.size
        self.state_memory[idx] = s
        self.ns_memory[idx] = ns
        self.action_memory[idx] = a
        self.reward_memory[idx] = r
        self.terminal_memory[idx] = done
        self.mem_count += 1

    def sample(self, batch_size=1):
        max_mem = min(self.mem_count, self.size)

        batch = np.random.choice(max_mem, batch_size)

        s = self.state_memory[batch]
        ns = self.ns_memory[batch]
        a = self.action_memory[batch]
        r = self.reward_memory[batch]
        done = self.terminal_memory[batch]

        return s, a, ns, r, done
