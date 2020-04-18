import tkinter

import numpy as np

from game_logic.board import Board
from utils.color import Color

PLAYERS = {Color.BLACK: 'Black', Color.WHITE: 'White'}


class Score:
	def __init__(self, color: Color, board: Board, root_window) -> None:
		self._player: Color = color
		self._board: np.array = board.board
		self._score: int = self.get_total_cells(self._player)
		self._score_label: tkinter.Label = tkinter.Label(master=root_window,
		                                                 text=self._score_text(),
		                                                 background='green',
		                                                 fg=PLAYERS[color],
		                                                 font=('System', 25))

	def update_score(self) -> None:
		self._score = self.get_total_cells(self._player)
		self._change_score_text()

	def get_score_label(self) -> tkinter.Label:
		return self._score_label

	def get_score(self) -> int:
		return self._score

	def _change_score_text(self) -> None:
		self._score_label['text'] = self._score_text()

	def _score_text(self) -> str:
		return f'{self._player.name}: {self._score}'

	def get_total_cells(self, color: Color) -> int:
		return len(np.where(self._board == color.value)[0])
