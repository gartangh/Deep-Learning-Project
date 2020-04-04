from game_logic.agents.agent import Agent
from utils.color import Color
from colorama import init
from termcolor import colored

from game_logic.game import Game
from game_logic.agents.random_agent import RandomAgent
from utils.intermediate_rewards.difference_between_players import difference_between_players
from utils.intermediate_rewards.difference_with_prev_board import difference_with_prev_board


def main():
	# initialize colors
	init()

	# check global variables
	assert 1 <= num_episodes <= 10000, f'Invalid number of episodes: num_episodes should be between 1 and 10000, but got {num_episodes}'
	assert black.num_games_won == 0, f'Invalid black agent'
	assert white.num_games_won == 0, f'Invalid white agent'

	print(f'\nPlayers:\n\t{black}\n\t{white}\n')

	for episode in range(1, num_episodes + 1):
		# create new game
		game = Game(episode, black, white, board_size, verbose)
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

	print('\n')


if __name__ == "__main__":
	# initialize global variables
	black: Agent = RandomAgent(Color.BLACK, None)  # the black agent
	white: Agent = RandomAgent(Color.WHITE, None)  # the white agent
	board_size: int = 8  # the size of the board e.g. 8x8
	num_episodes: int = 100  # the number of episodes e.g. 100
	verbose: bool = False  # wetter or not to print intermediate steps

	# call main
	main()
