import numpy as np


def heur(board_size: int) -> np.array:
	assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
	assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'

	edge: np.array = np.array([5, 2, 1])
	corner: np.array = np.array(
		[[100, -25, 10],
		 [-25, -25, 2],
		 [10, 2, 5]])

	weights: np.array = corner
	half_board_size: int = board_size // 2
	if half_board_size <= 3:
		weights: np.array = corner[:half_board_size, :half_board_size]
	else:
		weights: np.array = np.vstack([weights, edge])
		weights: np.array = np.column_stack([weights, np.append(edge, 1)])
		weights: np.array = np.pad(weights, (0, half_board_size - 4), 'edge')
		weights[half_board_size - 1, half_board_size - 1] = 2

	weights: np.array = np.pad(weights, (0, half_board_size), 'symmetric')

	return weights


def bench(board_size: int) -> np.array:
	assert board_size == 8, f'Invalid board size: board_size should be 8, but got {board_size}'

	weights: np.array = np.array([
		[80, -26, 24, -1, -5, 28, -18, 79],
		[-23, -39, -18, -9, -6, -8, -39, -1],
		[46, -16, 4, 1, -3, 6, -20, 52],
		[-13, -5, 2, -1, 4, 3, -12, -2],
		[-5, -6, 1, -2, -3, 0, -9, -5],
		[48, -13, 12, 5, 0, 5, -24, 41],
		[-27, -53, -11, -1, -11, -16, -58, -15],
		[87, -25, 27, -1, 5, 36, -3, 100]
	])

	return weights
