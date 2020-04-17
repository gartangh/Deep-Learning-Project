import tkinter
from typing import List, Tuple

from termcolor import colored

from game_logic.game import Game
from gui.view import View
from utils.color import Color


class Controller:
	def __init__(self, game: Game):
		self.game = game
		self.gui = None
		self.legal_actions: dict = self.game.board.get_legal_actions(Color.WHITE.value)
		self.prev_pass = False
		self.done = False
		self.gui = View()
		self.init_gui()

	def init_gui(self):
		self.gui.controller = self
		self.gui.configure_view()

	def start(self):
		self.gui.startup_gui()

	def on_board_clicked(self, event: tkinter.Event) -> None:
		if not self.done:
			move = self._convert_point_coord_to_move(event.x, event.y)
			if move in list(self.legal_actions.keys()):
				# Process white players turn
				legal_directions: List[Tuple[int, int]] = self.legal_actions[move]
				self.prev_pass = False
				self.done = self.game.board.take_action(move, legal_directions, Color.WHITE.value)
				if self.done:
					self._end_game()
					return
				else:
					# Process black players turn
					self._process_other_turn()
				self.gui.update_move()
		else:
			self.gui.close()  # closes window and starts new game if there are episodes left

	def _convert_point_coord_to_move(self, pointx: int, pointy: int) -> None:
		row = int(pointy // self.gui.get_cell_height())
		if row == self.gui.get_rows():
			row -= 1
		col = int(pointx // self.gui.get_cell_width())
		if col == self.gui.get_columns():
			col -= 1
		return (row, col)

	def _process_other_turn(self):
		legal_actions_black: dict = self.game.board.get_legal_actions(Color.BLACK.value)
		if not legal_actions_black:
			if self.prev_pass:
				self.done = True  # no agent has legal actions, deadlock
			self.prev_pass = True  # this agent has no legal actions, pass
		else:
			# get next action from legal actions and take it
			location, legal_directions = self.game.agent.get_next_action(self.game.board, legal_actions_black)
			self.prev_pass = False  # this agent has legal actions, no pass
			self.done = self.game.board.take_action(location, legal_directions, Color.BLACK.value)

		if self.done:
			self._end_game()
			return
		else:
			# Whites turn again
			self.legal_actions = self.game.board.get_legal_actions(Color.WHITE.value)
			if not self.legal_actions:
				if self.prev_pass:
					self.done = True  # no agent has legal actions, deadlock
					self._end_game()
					return
				else:
					self.prev_pass = True  # this agent has no legal actions, pass
					self._process_other_turn()

	def _end_game(self):
		self.game.black.update_final_score(self.game.board)
		self.game.white.update_final_score(self.game.board)
		# print end result
		if self.game.board.num_black_disks > self.game.board.num_white_disks:
			print(colored(
				f'{self.game.episode:>5}: BLACK ({self.game.board.num_black_disks:>3}|{self.game.board.num_white_disks:>3}|{self.game.board.num_free_spots:>3})',
				'red'))
		elif self.game.board.num_black_disks < self.game.board.num_white_disks:
			print(colored(
				f'{self.game.episode:>5}: WHITE ({self.game.board.num_black_disks:>3}|{self.game.board.num_white_disks:>3}|{self.game.board.num_free_spots:>3})',
				'green'))
		else:
			print(colored(
				f'{self.game.episode:>5}: DRAW  ({self.game.board.num_black_disks:>3}|{self.game.board.num_white_disks:>3}|{self.game.board.num_free_spots:>3})',
				'cyan'))
		print(f'Click on the screen to start a new episode!')
		self.gui.update_move()
