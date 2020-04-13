from game_logic.agents.dqn_agent import DQNAgent
from game_logic.agents.jaime_agent import JaimeAgent
from game_logic.agents.trainable_agent import TrainableAgent
from utils.color import Color
from colorama import init
from termcolor import colored

from game_logic.game import Game
from game_logic.agents.random_agent import RandomAgent
from utils.immediate_rewards.bench_heuristic import BenchHeuristic


def main():
	# initialize colors
	init()

	# check global variables
	assert 1 <= num_episodes <= 10000000, f'Invalid number of episodes: num_episodes should be between 1 and 10000, but got {num_episodes}'
	assert black.num_games_won == 0, f'Invalid black agent'
	assert white.num_games_won == 0, f'Invalid white agent'

	print(f'\nAgents:\n\t{black}\n\t{white}\n')

	for episode in range(1, num_episodes + 1):

		# create new game
		game: Game = Game(episode, black, white, board_size, verbose, tournament_mode)
		# play game
		game.play()

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
	verbose: bool = False  # whether or not to print intermediate steps
	tournament_mode = False #change every game of starting position or -> False every 4 games

	# train 2 agents through deep Q learning
	num_episodes: int = 1000  # the number of episodes e.g. 100
	black: DQNAgent = JaimeAgent(Color.BLACK, immediate_reward=BenchHeuristic(board_size), board_size=board_size)
	black.set_train_mode(True)
	white: DQNAgent = JaimeAgent(Color.WHITE, immediate_reward=BenchHeuristic(board_size), board_size=board_size)
	white.action_value_network = black.action_value_network
	white.target_network = white.target_network
	white.set_train_mode(True)
	main()

	black.final_save()
	white.final_save()

	#save agents
	dq_black = black
	dq_white = white

	# let the white agent play against a RandomAgent or a MinimaxAgent
	num_episodes: int = 50  # the number of episodes e.g. 100
	black.num_games_won = 0  # reset black agent
	black.set_train_mode(False)
	white: RandomAgent = RandomAgent(color=Color.WHITE) #MinimaxAgent(color=Color.WHITE, immediate_reward=MinimaxHeuristic(board_size))
	main()

	white = dq_white
	# let the white agent play against a RandomAgent or a MinimaxAgent
	white.num_games_won = 0  # reset black agent
	white.set_train_mode(False)
	black: RandomAgent = RandomAgent(color=Color.BLACK) #MinimaxAgent(color=Color.WHITE, immediate_reward=MinimaxHeuristic(board_size))
	main()

