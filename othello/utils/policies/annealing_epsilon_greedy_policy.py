import numpy as np
from tensorflow.keras import Sequential

from utils.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from utils.policies.policy import Policy


class AnnealingEpsilonGreedyPolicy(Policy):
	def __init__(self, start_eps, end_eps, n_steps, board_size):
		self.start_eps = start_eps
		self.end_eps = end_eps
		self.n_steps = n_steps
		self.inner_policy = EpsilonGreedyPolicy(self.start_eps, board_size)
		self.decisions_made = 0

	def __str__(self):
		return f'AnnealingEpsilonGreedy{super().__str__()}'

	@property
	def current_eps_value(self) -> float:
		# Linear annealed: f(x) = ax + b.
		a = -float(self.start_eps - self.end_eps) / float(self.n_steps)
		b = float(self.start_eps)
		value = max(self.end_eps, a * float(self.decisions_made) + b)
		return value

	def get_current_eps_policy(self) -> EpsilonGreedyPolicy:
		self.inner_policy.epsilon = self.current_eps_value
		return self.inner_policy

	def get_action(self, board: np.ndarray, action_value_network: Sequential, legal_actions: dict) -> tuple:
		self.decisions_made += 1
		return self.get_current_eps_policy().get_action(board, action_value_network, legal_actions)
