import collections
import pickle
import random
import numpy as np


class ReplayBuffer:
	def __init__(self, size: int = int(10)) -> None:
		self.size: int = int(size)
		self.buffer: collections.deque = collections.deque(maxlen=self.size)

	@property
	def n_obs(self) -> int:
		return len(self.buffer)

	def add(self, s: np.ndarray, a: tuple, r: float, terminal: bool) -> None:
		self.buffer.append((s, a, r, terminal))

	def persist(self, path) -> None:
		pickle.dump(self.buffer, open(path, 'wb'))

	def load(self, path) -> None:
		other_buffer: collections.deque = pickle.load(open(path, 'rb'))
		if other_buffer.maxlen > self.size:
			as_list: list = list(other_buffer)
			self.buffer.append(as_list[len(as_list) - self.size:len(as_list)])
		else:
			self.buffer: collections.deque = other_buffer
