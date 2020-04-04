import random

from game_logic.agents.agent import Agent
from game_logic.board import Board
from utils.color import Color


class RandomAgent(Agent):
	def __init__(self, color: Color):
		super().__init__(color)
		self.name: str = 'Random'

	def __str__(self):
		return f'{self.name}{super().__str__()}'

	def get_next_action(self, board: Board, legal_directions: dict) -> tuple:
		location: tuple = random.choice(list(legal_directions.keys()))
		legal_directions: list = legal_directions[location]

		return location, legal_directions
