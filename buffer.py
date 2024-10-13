from collections import namedtuple, deque

class Buffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
        self.transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        transition = self.transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def full(self):
        return len(self.memory) == self.memory.maxlen

    def clear(self):
        self.memory.clear()
