from termcolor import colored

from game_logic.agents.agent import Agent
from game_logic.board import Board
from utils.color import Color


class Game:
	def __init__(self, episode: int, black: Agent, white: Agent, board_size: int, verbose: bool):
		# check arguments
		assert black.color is Color.BLACK, f'Invalid black agent'
		assert white.color is Color.WHITE, f'Invalid white agent'

		self.episode: int = episode
		self.black: Agent = black
		self.white: Agent = white
		self.board: Board = Board(board_size)  # create a new board
		self.verbose: bool = verbose

		# state of the game
		self.ply = 0  # no plies so far
		self.player: Agent = black  # black begins
		self.prev_pass: bool = False  # no player has passed before
		self.done: bool = False  # not done yet

	def play(self):
		if self.verbose:
			print(f'Episode: {self.episode}')

		if self.verbose:
			print(f'Ply: {self.ply} (INIT)')
			print(self.board)

		# play until done
		while not self.done:
			# update
			self.ply += 1
			if self.verbose:
				print(f'Ply: {self.ply} ({self.player.color.name})')

			# get legal actions
			legal_actions: dict = self.board.get_legal_actions(self.player.color.value)

			if not legal_actions:
				# pass if no legal actions
				if self.verbose:
					print(f'No legal actions')
					print(f'Next action: PASS')
				self.prev_pass = True  # this player has no legal actions, pass
				if self.prev_pass:
					self.done = True  # no player has legal actions, deadlock
			else:
				# get next action from legal actions and take it
				location, legal_directions = self.player.get_next_action(self.board, legal_actions)
				if self.verbose:
					print(f'Legal actions: {list(legal_actions.keys())}')
					print(f'Next action: {location}')
				self.prev_pass = False  # this player has legal actions, no pass
				self.done = self.board.take_action(location, legal_directions, self.player.color.value)

			# get immediate reward
			immediate_reward: float = self.player.immediate_reward_function(self.board, self.player.color.value)
			if self.verbose:
				print(self.board)
				print(f'Immediate reward: {immediate_reward}')

			if not self.done:
				# the game is not done yet
				# change turns
				self.player = self.black if self.player == self.white else self.white
			else:
				# the game is done
				# update scores of both players
				self.black.update_final_score(self.board)
				self.white.update_final_score(self.board)

				# print end result
				if self.board.num_black_disks > self.board.num_white_disks:
					print(colored(
						f'{self.episode:>5}: BLACK ({self.board.num_black_disks:>3}|{self.board.num_white_disks:>3}|{self.board.num_free_spots:>3})',
						'red'))
				elif self.board.num_black_disks < self.board.num_white_disks:
					print(colored(
						f'{self.episode:>5}: WHITE ({self.board.num_black_disks:>3}|{self.board.num_white_disks:>3}|{self.board.num_free_spots:>3})',
						'green'))
				else:
					print(colored(
						f'{self.episode:>5}: DRAW  ({self.board.num_black_disks:>3}|{self.board.num_white_disks:>3}|{self.board.num_free_spots:>3})',
						'cyan'))
