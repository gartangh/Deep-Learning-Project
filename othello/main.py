from agent import Agent
from board import Board
from color import Color
from colorama import init
from termcolor import colored

from random_agent import RandomAgent


def main():
	# initialize colors
	init()

	# check global variables
	assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
	assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'
	assert 1 <= num_episodes <= 10000, f'Invalid number of episodes: num_episodes should be between 1 and 10000, but got {num_episodes}'
	assert black.color is Color.BLACK and black.score == 0 and black.num_games_won == 0, f'Invalid black agent'
	assert white.color is Color.WHITE and white.score == 0 and white.num_games_won == 0, f'Invalid white agent'

	print(f'\nPlayers:\n\t{black}\n\t{white}\n')

	for episode in range(1, num_episodes + 1):
		# TODO: create new game # game = Game(black, white)
		# TODO: play game # game.play()

		if verbose:
			print(f'Episode: {episode}')

		# initialize new episode
		board: Board = Board(board_size)  # create a new board
		ply: int = 0  # no plies so far
		turn: int = Color.BLACK.value  # black begins
		player: Agent = black  # black begins
		prev_pass: bool = False  # no player has passed before
		if verbose:
			print(f'Ply: {ply} (INIT)')
			print(board)

		# play until done
		done: bool = False
		while not done:
			# while not done, make a new ply
			# update
			ply += 1
			if verbose:
				print(f'Ply: {ply} ({player.color.name})')

			# get legal actions
			legal_actions: dict = board.get_legal_actions(player.color.value)
			if verbose:
				print(f'Legal actions: {list(legal_actions.keys())}')

			# get next action from player
			location, legal_directions = player.get_next_action(board, legal_actions)
			if verbose:
				print(f'Next action: {location}')

			# take action and get reward
			done = board.take_action(location, legal_directions, player.color.value)
			if verbose:
				print(board)

			# check for passes
			if location == 'pass' and prev_pass:
				done = True  # no player has legal actions, deadlock
				black.update_final_score(board)
				white.update_final_score(board)
			elif location == 'pass':
				prev_pass = True  # this player has no legal actions, pass
			else:
				prev_pass = False  # regular next action

			# update
			if not done:
				turn = 1 - turn  # change turns
				player = black if turn == 0 else white  # change players
			else:
				# update scores of both players
				black.update_final_score(board)
				white.update_final_score(board)

				if board.num_black_disks > board.num_white_disks:
					print(colored(
						f'{episode:>5}: BLACK ({board.num_black_disks:>3}|{board.num_white_disks:>3}|{board.num_free_spots:>3})',
						'red'))
				elif board.num_black_disks < board.num_white_disks:
					print(colored(
						f'{episode:>5}: WHITE ({board.num_black_disks:>3}|{board.num_white_disks:>3}|{board.num_free_spots:>3})',
						'green'))
				else:
					print(colored(
						f'{episode:>5}: DRAW  ({board.num_black_disks:>3}|{board.num_white_disks:>3}|{board.num_free_spots:>3})',
						'cyan'))

	print()

	assert 0 == black.score + white.score, 'The scores were miscalculated'
	ties: int = num_episodes - black.num_games_won - white.num_games_won
	if black.score > white.score:
		print(colored(
			f'BLACK {black.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'red'))
	elif black.score < white.score:
		print(colored(
			f'WHITE {white.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'green'))
	elif black.score == white.score:
		print(colored(
			f'DRAW  {black.num_games_won:>5}/{num_episodes:>5} ({black.num_games_won:>5}|{white.num_games_won:>5}|{ties:>5})',
			'cyan'))
	else:
		raise Exception('The scores were miscalculated')

	print('\n')


if __name__ == "__main__":
	# initialize global variables
	board_size: int = 6  # the size of the board e.g. 8x8
	num_episodes: int = 100  # the number of episodes e.g. 100
	black: Agent = RandomAgent(Color.BLACK)  # the black agent
	white: Agent = RandomAgent(Color.WHITE)  # the white agent
	verbose: bool = False  # wetter or not to print intermediate steps

	# call main
	main()
