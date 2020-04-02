import collections
import pickle
import random


class ReplayBuffer:
    def __init__(self, size=1e6) -> None:
        super().__init__()
        self.size = int(size)
        self.buffer = collections.deque(maxlen=self.size)

    @property
    def n_obs(self):
        return len(self.buffer)

    def add(self, s, a, r, next_s, terminal):
        self.buffer.append((s, a, r, next_s, terminal))

    def sample(self, size):
        return random.sample(self.buffer, size)

    def persist(self, path):
        pickle.dump(self.buffer, open(path, 'wb'))

    def load(self, path):
        other_buffer: collections.deque = pickle.load(open(path, 'rb'))
        if other_buffer.maxlen > self.size:
            as_list = list(other_buffer)
            self.buffer.append(as_list[len(as_list) - self.size:len(as_list)])

        else:
            self.buffer = other_buffer