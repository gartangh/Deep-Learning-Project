import numpy as np

from game_logic.board import Board


def difference_with_prev_board(board: Board, color_value: int) -> float:
	# number of player's disks - number of player's disks on the previous board
	num_player_disks: int = len(np.where(board.board == color_value)[0])
	num_player_disks_prev: int = len(np.where(board.prev_board == color_value)[0])
	immediate_reward: float = 1.0 * (num_player_disks - num_player_disks_prev)

	return immediate_reward
