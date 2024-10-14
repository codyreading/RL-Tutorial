import random
import numpy as np
import torch
from collections import namedtuple, deque

class Buffer:
    def __init__(self, buffer_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'next_action', 'done'])
        self.device = device
        self.dtype = torch.float32

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def full(self):
        return len(self.memory) == self.memory.maxlen

    def clear(self):
        self.memory.clear()

    def add(self, state, action, reward, next_state, next_action, done):
        transition = self.transition(state, action, reward.astype(np.float32), next_state, next_action, done)
        self.memory.append(transition)

    def stack(self, data):
        data = data._asdict()
        data = {key: np.stack(value) for key, value in data.items()}
        return data

    def to_torch(self, data):
        data = {key: torch.as_tensor(value).to(self.device).to(self.dtype) for key, value in data.items()}
        data["reward"] = data["reward"].unsqueeze(-1)
        data["done"] = data["done"].unsqueeze(-1)
        return data

    def pop(self):
        data = self.transition(*zip(*self.memory))
        data = self.stack(data)
        data = self.to_torch(data)
        self.clear()
        return data

    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)
        data = self.stack(data)
        data = self.to_torch(data)
        return data
