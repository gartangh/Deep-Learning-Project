from game_logic.agents.agent import Agent
from game_logic.agents.dqn_agent import DQNAgent
from game_logic.agents.jaime_agent import JaimeAgent
from game_logic.agents.risk_regions_agent import RiskRegionsAgent
from game_logic.agents.trainable_agent import TrainableAgent
from utils.color import Color
from colorama import init
from termcolor import colored
import matplotlib.pyplot as plt

from game_logic.game import Game
from game_logic.agents.random_agent import RandomAgent
from utils.immediate_rewards.minimax_heuristic import MinimaxHeuristic


def main(num_episodes, black, white, board_size, verbose, tournament_mode, plot_winratio):
	# initialize colors
	init()

	# initialize plot
	if plot_winratio:
		win_rates = []
		epsilons = []
		last_ten_matches = []
		plt.ion()  # non-blocking plot
		plt.title("Winratio of black (red), epsilon (green)")
		plt.xlabel("number of games played")
		plt.ylabel("winratio and epsilon")
		plt.show()

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

		if plot_winratio:
			if game.board.num_black_disks > game.board.num_white_disks:
				last_ten_matches.append(1)
			elif game.board.num_black_disks < game.board.num_white_disks:
				last_ten_matches.append(0)

			if episode % 10 == 9 and len(last_ten_matches) > 0:
				win_rates.append(sum(last_ten_matches) / len(last_ten_matches))
				epsilons.append(black.training_policy.current_eps_value)
				plt.plot([i*10 for i in range(len(win_rates))], win_rates, color='red')
				plt.plot([i*10 for i in range(len(win_rates))], epsilons, color='green')
				plt.draw()
				plt.pause(0.001)
				last_ten_matches = []
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
	if plot_winratio:
		# keep showing plot
		plt.ioff()
		plt.show()


def log(logline: str, path: str = "log.txt"):
	file = open(path, "a")
	file.write(logline + "\n")
	file.close()


def hardcore_training(black, white, board_size, total_iterations: int = 100_000, interval_log: int = 5000):
	black_dqn = black
	white_dqn = white

	total_runs = total_iterations // interval_log
	for i in range(total_runs):
		black = black_dqn
		white = white_dqn
		black.num_games_won = 0
		white.num_games_won = 0

		black.set_train_mode(True)
		white.set_train_mode(True)
		num_episodes = interval_log
		black = black_dqn
		white = white_dqn
		main(num_episodes, black, white, board_size, False, False, False)

		black.final_save()
		white.final_save()

		black.set_train_mode(False)
		white.set_train_mode(False)

		black_dqn = black
		white_dqn = white

		tournament_mode = True

		if isinstance(black, TrainableAgent):
			# test against random white
			print("test " + str(i) + ", BLACK DQN VS WHITE RANDOM")
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			white = RandomAgent(color=Color.WHITE)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log("test " + str(i) + "\tBLACK DQN VS WHITE RANDOM\t" + str(black.num_games_won) + "\t" + str(white.num_games_won))

		if isinstance(white, TrainableAgent):
			# test against random black
			black = black_dqn
			white = white_dqn
			print("test " + str(i) + ", BLACK RANDOM VS WHITE DQN")
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			black = RandomAgent(color=Color.BLACK)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log("test " + str(i) + "\tBLACK RANDOM VS WHITE DQN\t" + str(black.num_games_won) + "\t" + str(white.num_games_won))

		if isinstance(black, TrainableAgent):
			# test against risk region white
			black = black_dqn
			white = white_dqn
			print("test " + str(i) + ", BLACK DQN VS WHITE RISK_REGION")
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			white = RiskRegionsAgent(color=Color.WHITE)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log("test " + str(i) + "\tBLACK DQN VS WHITE RISK_REGION\t" + str(black.num_games_won) + "\t" + str(white.num_games_won))

		if isinstance(white, TrainableAgent):
			# test against risk region white
			black = black_dqn
			white = white_dqn
			print("test " + str(i) + ", BLACK RISK_REGION VS WHITE DQN")
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			black = RiskRegionsAgent(color=Color.BLACK)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log("test " + str(i) + "\tBLACK RISK_REGION VS WHITE DQN\t" + str(black.num_games_won) + "\t" + str(white.num_games_won))


if __name__ == "__main__":
	board_size: int = 8  # the size of the board e.g. 8x8
	verbose: bool = False  # whether or not to print intermediate steps
	tournament_mode = False # change every game of starting position or -> False every 4 games
	plot_winratio = True # every 10 games, plot how many Black won

	# train 1 agent through deep Q learning against a RandomAgent
	num_episodes: int = 2450 # enough to reach the end epsilon
	black: DQNAgent = JaimeAgent(Color.BLACK, immediate_reward=MinimaxHeuristic(board_size), board_size=board_size)
	black.set_train_mode(True)
	white: Agent = RandomAgent(Color.WHITE)
	main(num_episodes, black, white, board_size, verbose, tournament_mode, plot_winratio)

	# or instead, do hardcore training by uncommenting the following lines:
	# hardcore_training(black, white, board_size, verbose)

