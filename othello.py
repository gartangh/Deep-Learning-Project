from colorama import init
from color import Color
from help_functions import *
from random_agent import RandomAgent
from termcolor import colored


def main():
	# check global variables
	assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
	assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'
	assert 1 <= num_episodes <= 10000, f'Invalid number of episodes: num_episodes should be between 1 and 10000, but got {num_episodes}'
	assert black.color is Color.BLACK, f'Invalid black agent'
	assert white.color is Color.WHITE, f'Invalid white agent'

	print('Players:')
	print(black)
	print(white)
	print()

	for episode in range(1, num_episodes + 1):
		print(f'Episode: {episode}')

		# initialize episode
		board = create_board(board_size)
		ply = 0
		turn = 0  # black begins
		player = black
		prev_pass = False

		# print(f'Ply: {ply} (INIT)')
		# print_board(board)

		# play a new game
		done = False
		while not done:
			# update
			ply += 1
			# print(f'Ply: {ply} ({player.color.name})')

			# get legal actions
			legal_actions = get_legal_actions(board, turn)
			# print(f'Legal actions: {list(legal_actions.keys())}')
			# get next action from player
			location, legal_directions = player.next_action(legal_actions)
			# print(f'Action: {location}')

			# take action and get reward
			board, immediate_reward, done, final_reward, player_score, opponent_score, free = take_action(board,
			                                                                                              location,
			                                                                                              legal_directions,
			                                                                                              turn, player)
			# print_board(board)

			if location == 'pass' and prev_pass:
				# no player has legal actions, deadlock
				done = True
				final_reward = get_final_reward(player_score, opponent_score)
			elif location == 'pass':
				prev_pass = True
			else:
				prev_pass = False

			# update
			if not done:
				turn = 1 - turn
				player = black if turn == 0 else white
			else:
				# update scores of both players
				difference = player_score - opponent_score
				if final_reward == 1:
					print(colored(
						f'{player.color.name} won with {difference} disks after {ply} plies: ({player_score} over {opponent_score} with {free} spots)', 'green'))
					player.score += final_reward
					player.num_games_won += final_reward
					turn = 1 - turn
					player = black if turn == 0 else white
					player.score -= final_reward
				elif final_reward == -1:
					player.score += final_reward
					turn = 1 - turn
					player = black if turn == 0 else white
					player.score -= final_reward
					player.num_games_won -= final_reward
					print(colored(
						f'{player.color.name} won with {-difference} disks after {ply} plies: ({opponent_score} over {player_score} with {free} spots)', 'red'))
				else:
					print(colored(
						f'It\'s a draw after {ply} plies: ({player_score} for BLACK, {opponent_score} for WHITE, and {free} free)', 'cyan'))

	print()
	print()

	assert 0 == black.score + white.score, 'The scores were miscalculated'
	if black.score > white.score:
		print(f'BLACK won ({black.num_games_won} games out of {num_episodes}, end score = {black.score})')
	elif black.score < white.score:
		print(f'WHITE won ({white.num_games_won} games out of {num_episodes}, end score = {white.score})')
	elif black.score == white.score:
		print(f'It\'s a draw: BLACK and WHITE both won {black.score} games')
	else:
		raise Exception('The scores were miscalculated')

	print()


if __name__ == "__main__":
	# initialize colors
	init()

	# initialize global variables
	board_size: int = 8
	num_episodes: int = 100
	black: Agent = RandomAgent(Color.BLACK)
	white: Agent = RandomAgent(Color.WHITE)

	# call main
	main()
