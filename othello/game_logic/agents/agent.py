import numpy as np

from game_logic.board import Board
from utils.color import Color
from utils.replay_buffer import ReplayBuffer
from utils.immediate_rewards.immediate_reward import ImmediateReward


class Agent:
	def __init__(self, color: Color, immediate_reward: ImmediateReward = None):
		self.color: Color = color
		self.name = "Agent"
		self.immediate_reward: ImmediateReward = immediate_reward

		self.num_games_won: int = 0

	def __str__(self):
		return f'{self.name}'

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		raise NotImplementedError

	def update_final_score(self, board: Board) -> None:
		if self.color is Color.BLACK and board.num_black_disks > board.num_white_disks:
			self.num_games_won += 1  # BLACK won
		elif self.color is Color.WHITE and board.num_white_disks > board.num_black_disks:
			self.num_games_won += 1  # WHITE won

	def reset_score(self):
		self.num_games_won = 0


class TrainableAgent(Agent):
	def __init__(self, color: Color, immediate_reward: ImmediateReward = None, board_size: int = 8):
		super().__init__(color, immediate_reward)
		self.name = "TrainableAgent"
		self.board_size = board_size
		self.episode_rewards = []
		self.training_errors = []
		self.train_mode = False

		replay_buffer_size = 100_000
		self.replay_buffer = ReplayBuffer(size=replay_buffer_size)

	def __str__(self):
		return f'{self.name}'

	def create_model(self, verbose=False, lr: float=0.00025):
		raise NotImplementedError

	def set_train_mode(self, mode: bool):
		self.train_mode = mode

	def train(self, board: Board, action: tuple, reward: float, next_board: Board, terminal: bool, render: bool=False):
		raise NotImplementedError

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		raise NotImplementedError

	def q_learn_mini_batch(self):
		raise NotImplementedError

	def update_target_network(self):
		raise NotImplementedError

	def _can_start_learning(self):
		raise NotImplementedError

	def _persist_weights_if_necessary(self):
		raise NotImplementedError

	def board_to_nn_input(self, board: np.ndarray):
		# Transform board (numpy ndarray) to a numpy 1D array
		# that can serve as input to the neural network.
		# There are 2 nodes per board location:
		#   - 1 node that is 0 if location does not contain black, else 1
		#   - 1 node that is 0 if location does not contain white, else 1
		whites = np.zeros((1, self.board_size ** 2))
		blacks = np.zeros((1, self.board_size ** 2))
		for row in range(self.board_size):
			for col in range(self.board_size):
				if board[row][col] == 1: # white
					whites[0][row*self.board_size+col] = 1
				elif board[row][col] == 0: # black
					blacks[0][row*self.board_size+col] = 1

		# concatenate the 2 arrays
		return np.hstack([blacks, whites])
