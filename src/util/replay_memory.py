from collections import deque, namedtuple
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(deque):
    def __init__(self, memory_capacity):
        super().__init__(self, maxlen=memory_capacity)

    def append(self, *args):
        super().append(Transition(*args))

    def sample(self, batch_size, random_sample=True):
        if random_sample:
            return Transition(*zip(*random.sample(self, batch_size)))
        i = random.randint(0, len(self) - batch_size)
        return Transition(*zip(*list(self)[i: i+batch_size]))
