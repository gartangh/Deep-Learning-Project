from game_logic.agents.agent import Agent, TrainableAgent
from game_logic.agents.dqn_agent import DQNAgent
from utils.color import Color
from colorama import init
from termcolor import colored

from game_logic.game import Game
from game_logic.agents.random_agent import RandomAgent
from game_logic.agents.minimax_agent import MinimaxAgent

from utils.immediate_rewards.minimax_heuristic import MinimaxHeuristic


def main(train=False):
	# initialize colors
	init()

	# check global variables
	if train:
		assert isinstance(black, TrainableAgent)
		assert isinstance(white, TrainableAgent)
	assert 1 <= num_episodes <= 10000, f'Invalid number of episodes: num_episodes should be between 1 and 10000, but got {num_episodes}'
	assert black.num_games_won == 0, f'Invalid black agent'
	assert white.num_games_won == 0, f'Invalid white agent'

	print(f'\nAgents:\n\t{black}\n\t{white}\n')

	if train:
		black.set_train_mode(train)
		white.set_train_mode(train)

	for episode in range(1, num_episodes + 1):
		if train:
			# reset some agent attributes
			white.episode_rewards = []
			white.training_errors = []
			black.episode_rewards = []
			black.training_errors = []

		# create new game
		game: Game = Game(episode, black, white, board_size, verbose)
		# play game
		game.play(train)

	print()

	ties: int = num_episodes - black.num_games_won - white.num_games_won
	if black.num_games_won > white.num_games_won:
		print(colored(
			f'BLACK {black.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'red'))
	elif black.num_games_won < white.num_games_won:
		print(colored(
			f'WHITE {white.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'green'))
	elif black.num_games_won == white.num_games_won:
		print(colored(
			f'DRAW  {black.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'cyan'))
	else:
		raise Exception('The scores were miscalculated')

	print()


if __name__ == "__main__":
	# initialize global variables
	board_size: int = 8  # the size of the board e.g. 8x8
	num_episodes: int = 100  # the number of episodes e.g. 100
	verbose: bool = False  # whether or not to print intermediate steps

	# train 2 agents through deep Q learning
	black: Agent = DQNAgent(Color.BLACK, immediate_reward=MinimaxHeuristic(board_size), board_size=board_size)
	white: Agent = DQNAgent(Color.WHITE, immediate_reward=MinimaxHeuristic(board_size), board_size=board_size)
	main(train=True)

	# let the white agent play against a RandomAgent or a MinimaxAgent
	num_episodes = 50
	white = MinimaxAgent(color=Color.WHITE, immediate_reward=MinimaxHeuristic(board_size))  # the black agent
	# white = RandomAgent(color=Color.WHITE)
	black.reset_score()
	main(train=False)
