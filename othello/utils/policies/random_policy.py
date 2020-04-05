import numpy as np
import random
from typing import Dict, Tuple, List

from utils.policies.policy import Policy


class RandomPolicy(Policy):
	def __str__(self):
		return f'Random{super().__str__()}'

	def get_action(self, board: np.ndarray, legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Tuple[
		int, int]:
		return random.choice(list(legal_actions.keys()))

	def optimal(self, board, legal_actions) -> tuple:
		return self.get_action(board, legal_actions)
