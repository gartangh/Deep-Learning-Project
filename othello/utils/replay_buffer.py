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

	def add_final_reward(self, final_reward: float) -> None:
		last_element = self.buffer.pop()
		if last_element is not None:
			self.buffer.append((last_element[0], last_element[1], last_element[2] + final_reward, True))

	def clear(self) -> None:
		self.buffer.clear()

	def persist(self, path) -> None:
		pickle.dump(self.buffer, open(path, 'wb'))

	def load(self, path) -> None:
		other_buffer: collections.deque = pickle.load(open(path, 'rb'))
		if other_buffer.maxlen > self.size:
			as_list: list = list(other_buffer)
			self.buffer.append(as_list[len(as_list) - self.size:len(as_list)])
		else:
			self.buffer: collections.deque = other_buffer
