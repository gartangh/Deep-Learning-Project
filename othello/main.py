import matplotlib.pyplot as plt
from colorama import init
from termcolor import colored

from game_logic.agents.dqn_agent import DQNAgent
from game_logic.agents.human_agent import HumanAgent
from game_logic.agents.random_agent import RandomAgent
from game_logic.agents.risk_regions_agent import RiskRegionsAgent
from game_logic.agents.trainable_agent import TrainableAgent
from game_logic.game import Game
from gui.controller import Controller
from utils.color import Color
from utils.immediate_rewards.minimax_heuristic import MinimaxHeuristic


def main(num_episodes, black, white, board_size, verbose, tournament_mode, plot_win_ratio):
	# initialize colors
	init()

	# check global variables
	assert 0 <= num_episodes <= 10000000, f'Invalid number of episodes: num_episodes should be between 0 and 10000, but got {num_episodes}'
	assert black.num_games_won == 0, f'Invalid black agent'
	assert white.num_games_won == 0, f'Invalid white agent'

	print(f'\nAgents:\n\t{black}\n\t{white}\n')

	# initialize plot
	win_rates = []
	epsilons = []
	last_twentyfive_matches = []
	if plot_win_ratio:
		plt.ion()  # non-blocking plot
		plt.title('Win ratio of black (red), epsilon (green)')
		plt.xlabel('number of games played')
		plt.ylabel('win ratio and epsilon')
		plt.show()

	for episode in range(1, num_episodes + 1):
		if isinstance(black, TrainableAgent):
			black.episode_rewards = []
			black.training_errors = []

		if isinstance(white, TrainableAgent):
			white.episode_rewards = []
			white.training_errors = []

		# create new game
		game: Game = Game(episode, black, white, board_size, verbose, tournament_mode)
		if isinstance(white, HumanAgent):
			# create GUI controller
			controller: Controller = Controller(game)
			controller.start()
		else:
			# play game
			game.play()

		if plot_win_ratio:
			if game.board.num_black_disks > game.board.num_white_disks:
				last_twentyfive_matches.append(1)
			elif game.board.num_black_disks < game.board.num_white_disks:
				last_twentyfive_matches.append(-1)
			else:
				last_twentyfive_matches.append(0)

			if episode % 25 == 24 and len(last_twentyfive_matches) > 0:
				win_rates.append(sum(last_twentyfive_matches) / len(last_twentyfive_matches))
				epsilons.append(black.training_policy.current_eps_value)
				plt.plot([i * 25 for i in range(len(win_rates))], win_rates, color='red')
				plt.plot([i * 25 for i in range(len(win_rates))], epsilons, color='green')
				plt.draw()
				plt.pause(0.001)
				last_twentyfive_matches = []
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
	if plot_win_ratio:
		# keep showing plot
		plt.ioff()
		plt.show()


def log(logline: str, path: str = 'log.txt'):
	with open(path, 'a') as f:
		f.write(f'{logline}\n')


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
			print('test ' + str(i) + ', BLACK DQN VS WHITE RANDOM')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			white = RandomAgent(color=Color.WHITE)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK DQN VS WHITE RANDOM\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))

		if isinstance(white, TrainableAgent):
			# test against random black
			black = black_dqn
			white = white_dqn
			print('test ' + str(i) + ', BLACK RANDOM VS WHITE DQN')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			black = RandomAgent(color=Color.BLACK)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK RANDOM VS WHITE DQN\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))

		if isinstance(black, TrainableAgent):
			# test against risk region white
			black = black_dqn
			white = white_dqn
			print('test ' + str(i) + ', BLACK DQN VS WHITE RISK_REGION')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			white = RiskRegionsAgent(color=Color.WHITE)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK DQN VS WHITE RISK_REGION\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))

		if isinstance(white, TrainableAgent):
			# test against risk region white
			black = black_dqn
			white = white_dqn
			print('test ' + str(i) + ', BLACK RISK_REGION VS WHITE DQN')
			black.num_games_won = 0
			white.num_games_won = 0
			num_episodes = 244
			black = RiskRegionsAgent(color=Color.BLACK)
			main(num_episodes, black, white, board_size, False, tournament_mode, False)
			log('test ' + str(i) + '\tBLACK RISK_REGION VS WHITE DQN\t' + str(black.num_games_won) + '\t' + str(
				white.num_games_won))


if __name__ == '__main__':
	# initialize global variables
	board_size: int = 8  # the size of the board e.g. 8x8
	verbose: bool = False  # whether or not to print intermediate steps
	tournament_mode: bool = False  # change every game of starting position or -> False every 4 games
	plot_win_ratio: bool = False
	height: int = 400  # GUI height
	width: int = 400  # GUI width

	# train 2 agents through deep Q learning
	num_episodes: int = 0  # the number of episodes e.g. 10000
	black: DQNAgent = DQNAgent(Color.BLACK, immediate_reward=MinimaxHeuristic(board_size), board_size=board_size)
	black.set_train_mode(True)
	white: DQNAgent = DQNAgent(Color.WHITE, immediate_reward=MinimaxHeuristic(board_size), board_size=board_size)
	white.set_train_mode(True)
	main(num_episodes, black, white, board_size, verbose, tournament_mode, plot_win_ratio)

	black.final_save()
	white.final_save()
	# let the black agent play against a RandomAgent or a MinimaxAgent or a HumanAgent
	num_episodes: int = 3  # the number of episodes e.g. 100
	tournament_mode = True
	black.num_games_won = 0  # reset black agent
	black.set_train_mode(False)
	white: HumanAgent = HumanAgent(color=Color.WHITE)

	main(num_episodes, black, white, board_size, verbose, tournament_mode, plot_win_ratio)
