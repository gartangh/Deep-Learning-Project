import tkinter as tk
from typing import List, Tuple

import numpy as np

from game_logic.agents.agent import Agent
from game_logic.board import Board
from utils.color import Color


class HumanAgent(Agent):
	def __init__(self, color: Color):
		assert color == Color.WHITE, f'Invalid color for HumanAgent: color must be WHITE, but got {color.name}'

		super().__init__(color)

		self.first_move = True

	def __str__(self):
		return f'Human{super().__str__()}'

	def __update_board(self, board: Board, legal_directions: dict):
		disks: np.array = board.board
		board_size: int = board.board_size
		if self.first_move:
			for widget in self.frame.winfo_children():
				widget.destroy()
			self.first_move = False
		for i in range(board_size):
			for j in range(board_size):
				val = disks[i, j]
				if val == 0:
					label = tk.Label(self.frame, bg='black')
					label.place(relx=j / board_size, rely=i / board_size, relwidth=1 / board_size,
					            relheight=1 / board_size)
				elif val == 1:
					label = tk.Label(self.frame, bg='white')
					label.place(relx=j / board_size, rely=i / board_size, relwidth=1 / board_size,
					            relheight=1 / board_size)
				elif (i, j) in list(legal_directions.keys()):
					btn = tk.Button(self.frame, command=lambda: self.__click(j, i))
					btn.place(relx=j / board_size, rely=i / board_size, relwidth=1 / board_size,
					          relheight=1 / board_size)
				else:
					continue
		self.root.update_idletasks()
		self.root.update()

	def get_next_action(self, board: Board, legal_directions: dict) -> tuple:
		self.__update_board(board, legal_directions)
		while (not self.clicked):
			continue
		self.clicked = False
		location: Tuple[int, int] = (self.selected_col, self.selected_row)
		legal_directions: List[Tuple[int, int]] = legal_directions[location]
		return location, legal_directions
