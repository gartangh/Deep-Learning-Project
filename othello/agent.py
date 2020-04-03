from board import Board
from color import Color


class Agent:
	def __init__(self, color: Color):
		self.color: Color = color
		self.score: int = 0
		self.num_games_won: int = 0

	def __str__(self):
		return f'Agent: {self.color.name}'

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		raise NotImplementedError

	def get_immediate_reward(self, board: Board) -> float:
		raise NotImplementedError

	def update_final_score(self, board: Board) -> None:
		# BLACK
		if self.color is Color.BLACK and board.num_black_disks > board.num_white_disks:
			self.num_games_won += 1
			self.score += 1
		elif self.color is Color.BLACK and board.num_black_disks < board.num_white_disks:
			self.score -= 1
		elif self.color is Color.BLACK and board.num_black_disks == board.num_white_disks:
			pass
		# WHITE
		elif self.color is Color.WHITE and board.num_white_disks > board.num_black_disks:
			self.num_games_won += 1
			self.score += 1
		elif self.color is Color.WHITE and board.num_white_disks < board.num_black_disks:
			self.score -= 1
		elif self.color is Color.WHITE and board.num_white_disks == board.num_black_disks:
			pass
		# WRONG
		else:
			raise Exception('Scores were miscalculated')
