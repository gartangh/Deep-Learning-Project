import numpy as np
import random

from agent import Agent
from board import Board
from color import Color
from colorama import init
from termcolor import colored


class RandomAgent(Agent):
	def __init__(self, color):
		super().__init__(color)
		self.name = 'Random'

	def __str__(self):
		return f'{self.name}{super().__str__()}'

	def get_next_action(self, board: Board, legal_directions: dict) -> tuple:
		location: tuple = random.choice(list(legal_directions.keys()))
		legal_directions: list = legal_directions[location]

		return location, legal_directions

	def get_immediate_reward(self, board: Board) -> float:
		# 1 + number of turned disks
		difference: int = len(np.where(board.board == self.color.value)[0]) - len(
			np.where(board.prev_board == 1 - self.color.value)[0])
		return difference * 1.0
