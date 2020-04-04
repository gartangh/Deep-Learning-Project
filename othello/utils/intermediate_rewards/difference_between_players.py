import numpy as np

from game_logic.board import Board


def difference_between_players(board: Board, color_value: int) -> float:
	# number of players disks - number of opponent's disks
	difference: int = len(np.where(board.board == color_value)[0]) - len(
		np.where(board.board == 1 - color_value)[0])
	return difference * 1.0
