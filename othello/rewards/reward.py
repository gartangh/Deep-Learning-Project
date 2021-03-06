from abc import abstractmethod

from game_logic.board import Board
from utils.color import Color


class Reward:
	def __str__(self) -> str:
		return 'Reward'

	@abstractmethod
	def reward(self, board: Board, color: Color) -> float:
		raise NotImplementedError
