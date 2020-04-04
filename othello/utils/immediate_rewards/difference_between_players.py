import numpy as np

from game_logic.board import Board


def difference_between_players(board: Board, color_value: int) -> float:
	# number of players disks - number of opponent's disks
	num_player_disks: int = len(np.where(board.board == color_value)[0])
	num_opponent_disks: int = len(np.where(board.board == 1 - color_value)[0])
	immediate_reward: float = 1.0 * (num_player_disks - num_opponent_disks)

	return immediate_reward
