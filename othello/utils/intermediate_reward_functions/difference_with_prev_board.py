import numpy as np

from game_logic.board import Board


def difference_with_prev_board(board: Board, color_value: int) -> float:
	# 1 + number of turned disks
	difference: int = len(np.where(board.board == color_value)[0]) - len(
		np.where(board.prev_board == 1 - color_value)[0])
	return difference * 1.0
