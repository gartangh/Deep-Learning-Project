import numpy as np
import random
from tensorflow.keras import Sequential
from typing import Dict, List, Tuple

from utils.policies.policy import Policy


class EpsilonGreedyPolicy(Policy):
	def __init__(self, epsilon: float, board_size: int):
		self.epsilon: float = epsilon
		self.board_size: int = board_size

	def __str__(self):
		return f'EpsilonGreedy{super().__str__()}'

	def get_action(self, board: np.ndarray, action_value_network: Sequential,
	               legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> tuple:
		explore: bool = random.random() < self.epsilon
		if explore:
			return random.choice(list(legal_actions.keys()))
		else:
			q_values: np.array = action_value_network.predict(board)
			best_action: Tuple[int, int] = self.optimal(q_values, legal_actions)
			return best_action

	def optimal(self, q_values: np.array, legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]]):
		q_values: np.array = q_values.flatten()  # reshape (1,x) to (x,)
		q_values: np.array = [(q_values[row * self.board_size + col], (row, col)) for row in range(self.board_size) for
		                      col
		                      in range(self.board_size)]

		# get best legal action by sorting according to q value and taking the last legal entry
		q_values: np.array = sorted(q_values, key=lambda q: q[0])
		n_actions: int = len(q_values) - 1
		while n_actions >= 0:
			if q_values[n_actions][1] in legal_actions:
				return q_values[n_actions][1]
			n_actions -= 1

		return 'pass'
