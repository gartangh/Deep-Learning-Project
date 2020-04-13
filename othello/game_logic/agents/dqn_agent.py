import numpy as np
import os
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.board import Board
from utils.color import Color
from utils.immediate_rewards.immediate_reward import ImmediateReward
from utils.policies.annealing_epsilon_greedy_policy import AnnealingEpsilonGreedyPolicy
from utils.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from utils.replay_buffer import ReplayBuffer

import datetime


class DQNAgent(TrainableAgent):
	def __init__(self, color: Color, immediate_reward: ImmediateReward = None, board_size: int = 8):
		super().__init__(color, immediate_reward, board_size)
		self.epsilon: float = 0.1  # epsilon
		self.discount_factor: float = 1.0

		self.replay_buffer = ReplayBuffer(2)

		# start with epsilon 0.99 and slowly decrease it over 75 000 steps
		self.play_policy: EpsilonGreedyPolicy = EpsilonGreedyPolicy(self.epsilon, board_size)
		self.training_policy: AnnealingEpsilonGreedyPolicy = AnnealingEpsilonGreedyPolicy(self.epsilon, 0, 1000, 75_000,
		                                                                                  board_size)

		# old and new network to compare training loss
		self.action_value_network: Sequential = self.create_model()

		# save the weights of action_value_network periodically, i.e. when
		# self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
		self.persist_weights_every_n_times_trained: int = int(1e3)
		self.weight_persist_path: str = 'network_weights/' + ("BLACK" if self.color == Color.BLACK else "WHITE")
		if not os.path.exists(self.weight_persist_path):
			os.makedirs(self.weight_persist_path)

		# Bookkeeping values
		self.n_training_cycles: int = 0
		self.n_steps: int = 0

		self.load_weights()

	def __str__(self):
		return f'DQN{super().__str__()}'

	def create_model(self, verbose: bool = False, lr: float = 0.00025) -> Sequential:
		# input: 2 nodes per board location:
		#              - 1 node that is 0 if location does not contain black, else 1
		#              - 1 node that is 0 if location does not contain white, else 1
		# output: 1 node per board location, with probabilities to take action on that location
		model: Sequential = Sequential()
		model.add(Dense(2 * self.board_size ** 2, input_shape=(2 * self.board_size ** 2,), activation='relu'))
		model.add(Dense(self.board_size ** 2, activation='softmax'))
		model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

		return model

	def train(self, board: np.ndarray, action: tuple, reward: float, terminal: bool, render: bool = False):
		assert (self.train_mode is True)
		self.replay_buffer.add(board, action, reward, terminal)

		if self._can_start_learning():
			self.q_learn()
			self._persist_weights_if_necessary()

		self.n_steps += 1

		return

	def get_next_action(self, board: Board, legal_actions: dict) -> tuple:
		if self.train_mode is True:
			action = self.training_policy.get_action(self.board_to_nn_input(board.board), self.action_value_network, legal_actions)
		else:
			action = self.play_policy.get_action(self.board_to_nn_input(board.board), self.action_value_network, legal_actions)

		return action, legal_actions[action]

	def q_learn(self):
		self.n_training_cycles += 1

		# get most recent move: (s', a', r', t')
		curr_state, curr_action, curr_reward, curr_terminal = self.replay_buffer.buffer[-1]
		# get the move before that: (s, a, r, t)
		prev_state, prev_action, prev_reward, prev_terminal = self.replay_buffer.buffer[-2]
		if prev_terminal:
			target_q = self.action_value_network.predict(self.board_to_nn_input(prev_state))
			target_q[0][prev_action[0] * self.board_size + prev_action[1]] = prev_reward
			self.action_value_network.train_on_batch(self.board_to_nn_input(prev_state), target_q)
			return

		# calculate the new estimate of Q(s,a)
		# via formula Q(s,a) = r + gamma * max Q(s', a')
		# but only consider the legal actions a'
		q_values: np.array = self.action_value_network.predict(self.board_to_nn_input(curr_state)).flatten()
		q_values: list = [(q_values[row * self.board_size + col], (row, col)) for row in range(self.board_size) for col in range(self.board_size)]
		q_values: list = sorted(q_values, key=lambda q: q[0])

		legal_next_actions = Board._get_legal_actions(curr_state, self.board_size, self.color.value)
		best_q: int = 0
		n_actions: int = len(q_values) - 1
		while n_actions >= 0:
			if q_values[n_actions][1] in legal_next_actions:
				best_q: int = q_values[n_actions][0]
				break
			n_actions -= 1

		new_q_previous_state = prev_reward + curr_reward if curr_terminal and n_actions < 0 else prev_reward + self.discount_factor * best_q

		# train NN by backpropagating the error between old and new Q(s,a)
		target_q = self.action_value_network.predict(self.board_to_nn_input(prev_state))
		target_q[0][prev_action[0] * self.board_size + prev_action[1]] = new_q_previous_state
		self.action_value_network.train_on_batch(self.board_to_nn_input(prev_state), target_q)
		return

	def _can_start_learning(self) -> bool:
		return self.n_steps > 1

	def _persist_weights_if_necessary(self) -> None:
		if self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
			name = "BLACK" if self.color == Color.BLACK else "WHITE"
			name = name + datetime.datetime.now().strftime("%y%m%d%H%M%S")
			path: str = '{}/weights_agent_{}.h5f'.format(self.weight_persist_path, name)
			self.action_value_network.save_weights(path, overwrite=True)

			path: str = 'replay_buffers/' + ("BLACK" if self.color == Color.BLACK else "WHITE")
			if not os.path.exists(path):
				os.makedirs(path)
			file_path: str = os.path.join(path, 'replay_buffer_agent_{}.pkl'.format(name))
			self.replay_buffer.persist(file_path)

			path_values: str = 'hyper_values/' + ("BLACK" if self.color == Color.BLACK else "WHITE")
			if not os.path.exists(path_values):
				os.makedirs(path_values)
			path_values = os.path.join(path_values, "vals_{}.pkl".format(name))
			values = {"decisions_made": self.training_policy.decisions_made,
			          "n_training_cycles": self.n_training_cycles,
			          "n_steps": self.n_steps}
			pickle.dump(values, open(path_values, "wb"))

	def final_save(self) -> None:
		name = "FINAL_"
		name += "BLACK" if self.color == Color.BLACK else "WHITE"
		name += datetime.datetime.now().strftime("%y%m%d%H%M%S")
		path: str = '{}/weights_agent_{}.h5f'.format(self.weight_persist_path, name)
		self.action_value_network.save_weights(path, overwrite=True)

		path: str = 'replay_buffers/' + ("BLACK" if self.color == Color.BLACK else "WHITE")
		if not os.path.exists(path):
			os.makedirs(path)
		file_path: str = os.path.join(path, 'replay_buffer_agent_{}.pkl'.format(name))
		self.replay_buffer.persist(file_path)

		path_values: str = 'hyper_values/' + ("BLACK" if self.color == Color.BLACK else "WHITE")
		if not os.path.exists(path_values):
			os.makedirs(path_values)
		path_values = os.path.join(path_values, "vals_{}.pkl".format(name))
		values = {"decisions_made": self.training_policy.decisions_made,
		          "n_training_cycles": self.n_training_cycles,
		          "n_steps": self.n_steps}
		pickle.dump(values, open(path_values, "wb"))

	def load_weights(self, file_name=None) -> None:
		path = "replay_buffers/" + ("BLACK" if self.color == Color.BLACK else "WHITE")
		path_values = "hyper_values/" + ("BLACK" if self.color == Color.BLACK else "WHITE")
		if file_name is None:
			path_network = tf.train.latest_checkpoint(self.weight_persist_path)
			if path_network is None: return
			name = os.path.basename(path_network)
			name = name.replace(".h5f", ".pkl")
			path_replay = name.replace("weights_agent_", "replay_buffer_agent_")
			path_replay = os.path.join(path, path_replay)

			path_vals = name.replace("weights_agent_", "vals_")
			path_vals = os.path.join(path_values, path_vals)
		else:
			path_network = os.path.join(self.weight_persist_path, "weights_agent_" + file_name + ".h5f")
			path_replay = os.path.join(path, "replay_buffer_agent_" + file_name + ".pkl")
			path_vals = os.path.join(path_values, "vals_" + file_name + ".pkl")

		self.action_value_network.load_weights(path_network)  # loading in the action value network weights

		self.replay_buffer.load(path_replay)

		values = pickle.load(open(path_vals, 'rb'))

		self.training_policy.decisions_made = values["decisions_made"]
		self.n_training_cycles = values["n_training_cycles"]
		self.n_steps = values["n_steps"]

		print("WEIGHTS HAVE BEEN LOADED -> CONTINUING TRAINING")
