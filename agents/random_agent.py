import numpy as np
import random

from agents.agent import Agent


class RandomAgent(Agent):
	def __init__(self, color):
		super().__init__(color)
		self.name = 'Random'

	def __str__(self):
		return f'{self.name}{super().__str__()}'

	def next_action(self, legal_actions):
		action = random.choice(list(legal_actions.keys()))
		legal_actions = legal_actions[action]

		return action, legal_actions

	def immediate_reward(self, board, prev_board, turn):
		# 1 + number of turned disks
		difference = len(np.where(board == turn)[0]) - len(np.where(prev_board == turn)[0])
		return difference
