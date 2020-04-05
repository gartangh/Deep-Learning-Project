import numpy as np
import copy

from utils.color import Color
from typing import Dict, List, Tuple


class Board:
	# initialize static variables
	_directions: List[Tuple[int, int]] = [
		(+1, +0),  # down
		(+1, +1),  # down right
		(+0, +1),  # right
		(-1, +1),  # up right
		(-1, +0),  # up
		(-1, -1),  # up left
		(+0, -1),  # left
		(+1, -1)  # down left
	]

	# public methods
	# constructor
	def __init__(self, board_size: int = 8):
		# check arguments
		assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
		assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'
		self.board_size: int = board_size

		# create board
		board: np.array = -np.ones([board_size, board_size], dtype=int)
		board[board_size // 2 - 1, board_size // 2 - 1] = 1  # white
		board[board_size // 2, board_size // 2 - 1] = 0  # black
		board[board_size // 2 - 1, board_size // 2] = 0  # black
		board[board_size // 2, board_size // 2] = 1  # white
		self.board: np.array = board
		self.prev_board: np.array = copy.deepcopy(board)
		self.num_black_disks: int = 2
		self.num_white_disks: int = 2
		self.num_free_spots: int = board_size ** 2 - 4

	def __str__(self):
		string: str = '\t|'
		for j in range(self.board_size):
			string += f'{j}\t'
		string += '\n_\t|'
		for j in range(self.board_size):
			string += '_\t'
		string += '\n'
		for i, row in enumerate(self.board):
			string += f'{i}\t|'
			for val in row:
				if val == -1:
					string += ' \t'
				elif val == 0:
					string += 'B\t'
				elif val == 1:
					string += 'W\t'
				else:
					raise Exception(f'Incorrect value on board: expected 1, -1 or 0, but got {val}')
			string += '\n'
		return string

	def get_legal_actions(self, color_value: int) -> dict:
		return self._get_legal_actions(self.board, self.board_size, color_value)

	def take_action(self, location: tuple, legal_directions: list, color_value: int) -> bool:
		# check if location does point to an empty spot
		assert self.board.item(
			location) == -1, f'Invalid location: location ({location}) does not point to an empty spot on the board)'

		# save state before action
		self.prev_board: np.array = copy.deepcopy(self.board)

		# put down own disk in the provided location
		i: int = location[0]
		j: int = location[1]
		self.board[i, j]: int = color_value

		# turn around opponent's disks
		for direction in legal_directions:
			i: int = location[0] + direction[0]
			j: int = location[1] + direction[1]
			while 0 <= i < self.board_size and 0 <= j < self.board_size:
				disk: int = self.board[i, j]
				if disk == color_value or disk == Color.EMPTY.value:
					break  # encountered empty spot or own disk
				if disk == 1 - color_value:
					self.board[i, j] = color_value  # encountered opponent's disk
				i += direction[0]
				j += direction[1]

		# update scores
		self._update_score()

		# check if othello is finished
		done: bool = self._is_game_finished()

		return done

	def _update_score(self) -> None:
		# get scores
		num_black_disks: int = len(np.where(self.board == Color.BLACK.value)[0])
		num_white_disks: int = len(np.where(self.board == Color.WHITE.value)[0])
		num_free_spots: int = len(np.where(self.board == Color.EMPTY.value)[0])
		num_disks: int = num_black_disks + num_white_disks

		# check scores
		assert 0 <= num_black_disks <= self.board_size ** 2, f'Invalid number of black disks: num_black_disks should be between 0 and {self.board_size ** 2}, but got {num_black_disks}'
		assert 0 <= num_white_disks <= self.board_size ** 2, f'Invalid number of white disks: num_white_disks should be between 0 and {self.board_size ** 2}, but got {num_white_disks}'
		assert 0 <= num_free_spots <= self.board_size ** 2 - 4, f'Invalid number of free spots: num_free_spots should be between 0 and {self.board_size ** 2 - 4}, but got {num_free_spots}'
		assert num_disks + num_free_spots == self.board_size ** 2, f'Invalid number of disks and free spots: sum of disks and num_free_spots should be {self.board_size ** 2}, but got {num_disks + num_free_spots}'

		self.num_black_disks: int = num_black_disks
		self.num_white_disks: int = num_white_disks
		self.num_free_spots: int = num_free_spots

	def _is_game_finished(self) -> bool:
		# return finished or not
		if self.num_black_disks == 0 or self.num_white_disks == 0 or self.num_free_spots == 0:
			return True
		else:
			return False

	@staticmethod
	def _get_legal_actions(board: np.array, board_size: int, color_value: int) -> dict:
		legal_actions: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
		for i in range(board_size):
			for j in range(board_size):
				legal_directions: List[Tuple[int, int]] = Board._get_legal_directions(board, board_size, (i, j),
				                                                                      color_value)
				if len(legal_directions) > 0:
					legal_actions[(i, j)]: list = legal_directions

		return legal_actions

	@staticmethod
	def _get_legal_directions(board: np.array, board_size: int, location: tuple, color_value: int) -> list:
		legal_directions: List[Tuple[int, int]] = []

		# check if location points to an empty spot
		if board.item(location) != -1:
			return legal_directions

		# search in all directions
		for direction in Board._directions:
			found_opponent: bool = False  # check wetter there is an opponent's disk
			i: int = location[0] + direction[0]
			j: int = location[1] + direction[1]
			while 0 <= i < board_size and 0 <= j < board_size:
				# while not out of the board, keep going
				disk: int = board[i, j]
				if disk == -1 or (disk == color_value and not found_opponent):
					break  # found empty spot or player's disk before finding opponent's disk

				if disk == 1 - color_value:
					found_opponent: bool = True  # found opponent's disk

				if disk == color_value and found_opponent:
					legal_directions.append(direction)  # found own disk after finding opponent's disk
					break

				i += direction[0]
				j += direction[1]

		return legal_directions
