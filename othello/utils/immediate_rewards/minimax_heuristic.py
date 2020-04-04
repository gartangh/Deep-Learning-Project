import numpy as np

from game_logic.board import Board
from utils.immediate_rewards.immediate_reward import ImmediateReward


class MinimaxHeuristic(ImmediateReward):
	# initialize static variables
	_edge: np.array = np.array([5, 2, 1])
	_corner: np.array = np.array(
		[[100, -25, 10],
		 [-25, -25, 2],
		 [10, 2, 5]])

	def __init__(self, board_size: int = 8):
		# check arguments
		assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
		assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'

		# create weights
		weights: np.array = MinimaxHeuristic._corner
		half_board_size: int = board_size // 2
		if half_board_size <= 3:
			weights = MinimaxHeuristic._corner[:half_board_size, :half_board_size]
		else:
			weights = np.vstack([weights, MinimaxHeuristic._edge])
			weights = np.column_stack([weights, np.append(MinimaxHeuristic._edge, 1)])
			weights = np.pad(weights, (0, half_board_size - 4), 'edge')
			weights[half_board_size - 1, half_board_size - 1] = 2

		self._weights = np.pad(weights, (0, half_board_size), 'symmetric')

	def _evaluate_board(self, board: np.array, color_value: int):
		player_disks: np.array = np.where(board == color_value, 1, 0)
		opponent_disks: np.array = np.where(board == 1 - color_value, -1, 0)
		disks: np.array = np.add(player_disks, opponent_disks)
		weighted_disks: np.array = np.multiply(disks, self._weights)
		result: int = np.sum(weighted_disks)  # will be an int, but numpy says it will be np.array

		return result

	def immediate_reward(self, board: Board, color_value: int) -> float:
		board_score: int = self._evaluate_board(board.board, color_value)
		prev_board_score: int = self._evaluate_board(board.prev_board, color_value)
		immediate_reward: float = 1.0 * (board_score - prev_board_score)

		return immediate_reward
