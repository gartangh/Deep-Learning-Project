from collections import defaultdict
from math import ceil
from typing import List

from colorama import init
from tqdm import tqdm

from agents.agent import Agent
from agents.human_agent import HumanAgent
from agents.trainable_agent import TrainableAgent
from game_logic.game import Game
from gui.controller import Controller
from policies.annealing_trainable_policy import AnnealingTrainablePolicy
from policies.epsilon_greedy_annealing_trainable_policy import EpsilonGreedyAnnealingTrainablePolicy
from policies.epsilon_greedy_trainable_policy import EpsilonGreedyTrainablePolicy
from utils.color import Color
from utils.config import Config
from utils.plot import Plot
from utils.types import Actions


class GlobalConfig:
	def __init__(self, board_size: int, black: Agent, train_configs: List[Config], eval_configs: List[Config],
	             test_configs: List[Config], human_configs: List[Config]) -> None:
		assert black.color is Color.BLACK, f'Invalid black agent: black agent\'s color is not black'

		self.board_size: int = board_size
		self.black = black
		self.train_configs: List[Config] = train_configs
		self.eval_configs: List[Config] = eval_configs
		self.test_configs: List[Config] = test_configs
		self.human_configs: List[Config] = human_configs

		self.total_episodes: int = 0

		# initialize plot
		if isinstance(self.black, TrainableAgent):
			self.plot: Plot = Plot()
			self.scores: defaultdict = defaultdict(list)

		# initialize colors
		init()

	def start(self) -> None:
		# set train mode
		self.black.train_mode = True

		# train and evaluate
		for config in self.train_configs:
			self.train_eval(config)
			self.total_episodes += config.num_episodes

		# set train mode
		self.black.train_mode = False

		# test
		for config in self.test_configs:
			self.test(config)

		# play against a human
		for config in self.human_configs:
			self.human(config)

	def train_eval(self, config: Config) -> None:
		assert isinstance(self.black, TrainableAgent)

		# agents
		black: TrainableAgent = self.black
		white: Agent = config.white
		if isinstance(white, TrainableAgent):
			white.train_mode = config.train_white
		# initialize train policy
		if isinstance(black.train_policy, AnnealingTrainablePolicy):
			black.train_policy.num_episodes = config.num_episodes
		if isinstance(config.white, TrainableAgent) and isinstance(white.train_policy, AnnealingTrainablePolicy):
			white.train_policy.num_episodes = config.num_episodes
		# print agents
		print(f'\nTRAINING\n\t{black}\n\t{white}\n')

		for episode in tqdm(range(1, config.num_episodes + 1)):
			# update policies
			if isinstance(black.train_policy, AnnealingTrainablePolicy):
				black.train_policy.update(episode)
			if isinstance(white, TrainableAgent) and isinstance(black.train_policy, AnnealingTrainablePolicy):
				white.train_policy.update(episode)

			# evaluate every 10 % of number of episodes
			if (episode - 1) % ceil(config.num_episodes / 10) == 0:
				# set train mode
				black.train_mode = False

				# evaluate
				print(f'\nEVALUATING episode {episode - 1:>5}')
				self.evals()

				# plot win ratio
				self.plot.update(self.total_episodes + episode - 1, self.scores)

				# set train mode back
				black.train_mode = True

			# play new game
			Game(self.board_size, self.black, config, episode, random_start=True).play()

		# set train mode one last time
		black.train_mode = False

		# evaluate one last time
		print(f'EVALUATING episode {config.num_episodes:>5}')
		self.evals()

		# plot win ratio one last time
		self.plot.update(self.total_episodes + config.num_episodes, self.scores)

		# set train mode back one last time
		black.train_mode = True

		# save models
		black.save_weights()
		if isinstance(white, TrainableAgent) and white.train_mode:
			white.save_weights()

	def evals(self) -> None:
		# win rate for each eval agent
		for i, config in enumerate(self.eval_configs):
			win_ratio: float = self.eval(config)
			self.scores[i].append(win_ratio)

		# epsilon for trainable agent
		if isinstance(self.black, TrainableAgent) and isinstance(self.black.train_policy, EpsilonGreedyTrainablePolicy):
			self.scores['epsilon'].append(self.black.train_policy.epsilon*100)
		elif isinstance(self.black, TrainableAgent) and isinstance(self.black.train_policy, EpsilonGreedyAnnealingTrainablePolicy):
			self.scores['epsilon'].append(self.black.train_policy.inner_policy.epsilon*100)

	def eval(self, config: Config) -> float:
		assert isinstance(self.black, TrainableAgent)

		# agents
		black: TrainableAgent = self.black
		white: Agent = config.white
		# set train mode
		if isinstance(white, TrainableAgent):
			white.train_mode = False
		# reset agents
		black.reset()
		white.reset()

		for episode in range(1, config.num_episodes + 1):
			# create new game
			game: Game = Game(self.board_size, self.black, config, episode)
			# play game
			game.play()

		# print score
		ties: int = config.num_episodes - black.num_games_won - white.num_games_won
		win_ratio: float = black.num_games_won / config.num_episodes * 100
		print(
			f'({black.num_games_won:>4}|{white.num_games_won:>4}|{ties:>4}) / {config.num_episodes:>4} -> win ratio: {win_ratio:>6.2f} %')

		# set train mode back
		if isinstance(white, TrainableAgent):
			white.train_mode = config.train_white

		return win_ratio

	def test(self, config: Config) -> None:
		# agents
		black: Agent = self.black
		white: Agent = config.white
		# set train mode
		if isinstance(white, TrainableAgent):
			white.train_mode = False
		# reset agents
		black.reset()
		white.reset()
		# print agents
		print(f'\nTESTING\n\t{black}\n\t{white}\n')

		for episode in tqdm(range(1, config.num_episodes + 1)):
			# create new game
			game: Game = Game(self.board_size, self.black, config, episode)
			# play game
			game.play()

		# print score
		ties: int = config.num_episodes - black.num_games_won - white.num_games_won
		win_ratio: float = black.num_games_won / config.num_episodes * 100
		print(
			f'({black.num_games_won:>4}|{white.num_games_won:>4}|{ties:>4}) / {config.num_episodes:>4} -> win ratio: {win_ratio:>6.2f} %')

		# set train mode back
		if isinstance(white, TrainableAgent):
			white.train_mode = config.train_white

	def human(self, config: Config) -> None:
		assert isinstance(config.white, HumanAgent)

		# agents
		black = self.black
		white = config.white
		# reset agents
		black.reset()
		white.reset()
		print(f'\nHUMAN\n\t{black}\n\t{white}\n')

		for episode in tqdm(range(1, config.num_episodes + 1)):
			# create new game
			game: Game = Game(self.board_size, self.black, config, episode)
			# let black begin
			# get legal actions
			legal_actions: Actions = game.board.get_legal_actions(game.agent.color)
			location, legal_directions = game.agent.next_action(game.board, legal_actions)
			game.board.take_action(location, legal_directions, game.agent.color)
			# create GUI controller
			controller: Controller = Controller(game, self.black)
			# play game
			controller.start()

		# print score
		ties: int = config.num_episodes - black.num_games_won - white.num_games_won
		win_ratio: float = black.num_games_won / config.num_episodes * 100
		print(
			f'({black.num_games_won:>4}|{white.num_games_won:>4}|{ties:>4}) / {config.num_episodes:>4} -> win ratio: {win_ratio:>6.2f} %')
