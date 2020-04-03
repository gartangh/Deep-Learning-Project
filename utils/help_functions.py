import numpy as np
import copy

from agents.agent import Agent

# initialize global variables
directions = [[+1, +0],  # down
              [+1, +1],  # down right
              [+0, +1],  # right
              [-1, +1],  # up right
              [-1, +0],  # up
              [-1, -1],  # up left
              [+0, -1],  # left
              [+1, -1]]  # down left


def create_board(board_size: int = 8):
    # check arguments
    assert 4 <= board_size <= 12, f'Invalid board size: board_size should be between 4 and 12, but got {board_size}'
    assert board_size % 2 == 0, f'Invalid board size: board_size should be even, but got {board_size}'

    board = -np.ones([board_size, board_size], dtype=int)
    board[board_size // 2 - 1, board_size // 2 - 1] = 1  # white
    board[board_size // 2, board_size // 2 - 1] = 0  # black
    board[board_size // 2 - 1, board_size // 2] = 0  # black
    board[board_size // 2, board_size // 2] = 1  # white

    return board


def print_board(board: np.array):
    board_size = len(board)
    print('\t|', end='')
    for j in range(len(board)):
        print(f'{j}\t', end='')
    print('\n_\t|', end='')
    for j in range(board_size):
        print('_\t', end='')
    print()
    for i, row in enumerate(board):
        print(i, end='\t|')
        for val in row:
            if val == -1:
                print(' \t', end='')
            elif val == 0:
                print('B\t', end='')
            elif val == 1:
                print('W\t', end='')
            else:
                raise Exception(f'Incorrect value on board: expected 1, -1 or 0, but got {val}')
        print()


def get_legal_actions(board: np.array, turn: int):
    board_size = len(board)
    legal_actions = {}
    for i in range(board_size):
        for j in range(board_size):
            legal_directions = get_legal_directions(board, (i, j), turn)
            if len(legal_directions) > 0:
                legal_actions[(i, j)] = legal_directions

    # pass if no legal action
    if len(list(legal_actions.keys())) == 0:
        legal_actions['pass'] = None

    return legal_actions


# board: current game board
# location: tuple (row, column) where user want to play
# player: 1 white player/ 2 black player
# returns: if move was successful, list of list of directions where it is legal
def get_legal_directions(board: np.array, location: tuple, turn: int):
    legal_directions = []

    # check if location points to an empty spot
    if board.item(location) != -1:
        return legal_directions

    board_size = len(board)
    for direction in directions:
        new_x = location[0] + direction[0]
        new_y = location[1] + direction[1]
        found_player2 = False  # check wetter there is a player2 disk
        while 0 <= new_x < board_size and 0 <= new_y < board_size:
            disk = board[new_x, new_y]
            if disk == -1:
                break

            if disk == 1 - turn:
                found_player2 = True

            if disk == turn and found_player2:
                legal_directions.append(direction)
                break

            new_x += direction[0]
            new_y += direction[1]

    return legal_directions


def take_action(board: np.array, location, legal_directions: list, turn: int, player: Agent):
    if location == 'pass':
        # get scores
        player_score = len(np.where(board == turn)[0])
        opponent_score = len(np.where(board == 1 - turn)[0])
        free = len(np.where(board == -1)[0])

        # check scores
        board_size = len(board)
        assert 0 <= player_score <= board_size ** 2, f'Invalid player score: player_score should be between 0 and {board_size ** 2}, but got {player_score}'
        assert 0 <= opponent_score <= board_size ** 2, f'Invalid opponent score: opponent_score should be between 0 and {board_size ** 2}, but got {opponent_score}'
        assert 0 <= free <= board_size ** 2 - 4, f'Invalid free: free should be between 0 and {board_size ** 2 - 4}, but got {free}'
        assert player_score + opponent_score + free == board_size ** 2, f'Invalid scores: sum of scores should be {board_size ** 2}, but got {player_score + opponent_score + free}'

        return board, 0, False, 0, player_score, opponent_score, free  # do nothing

    # save state before action
    prev_board = copy.deepcopy(board)

    # put down own disk in the provided location
    board[location[0], location[1]] = turn

    # turn around opponent's disks
    board_size = len(board)
    for direction in legal_directions:
        new_x = location[0] + direction[0]
        new_y = location[1] + direction[1]
        while 0 <= new_x < board_size and 0 <= new_y < board_size:
            disk = board[new_x, new_y]
            if disk == turn or disk == -1:
                break  # encountered empty spot or own disk
            if disk == 1 - turn:
                board[new_x, new_y] = turn  # encountered opponent's disk
            new_x += direction[0]
            new_y += direction[1]

    # calculate immediate reward
    immediate_reward = player.immediate_reward(board, prev_board, turn)

    # get scores
    player_score = len(np.where(board == turn)[0])
    opponent_score = len(np.where(board == 1 - turn)[0])
    free = len(np.where(board == -1)[0])

    # check scores
    board_size = len(board)
    assert 0 <= player_score <= board_size ** 2, f'Invalid player score: player_score should be between 0 and {board_size ** 2}, but got {player_score}'
    assert 0 <= opponent_score <= board_size ** 2, f'Invalid opponent score: opponent_score should be between 0 and {board_size ** 2}, but got {opponent_score}'
    assert 0 <= free <= board_size ** 2 - 4, f'Invalid free: free should be between 0 and {board_size ** 2 - 4}, but got {free}'
    assert player_score + opponent_score + free == board_size ** 2, f'Invalid scores: sum of scores should be {board_size ** 2}, but got {player_score + opponent_score + free}'

    # check if game is finished
    done = check_game_finished(player_score, opponent_score, free, board_size)

    # calculate final reward
    final_reward = 0
    if done:
        final_reward = get_final_reward(player_score, opponent_score)

    return board, immediate_reward, done, final_reward, player_score, opponent_score, free


def check_game_finished(player_score: int, opponent_score: int, free: int, board_size: int):
    # return finished or not
    if player_score == 0 or opponent_score == 0 or player_score + opponent_score == board_size ** 2 or free == 0:
        return True

    return False


def get_final_reward(player_score: int, opponent_score: int):
    # return 1 if player wins, -1 if player loses, and 0 if it's a draw
    if player_score > opponent_score:
        return 1
    elif player_score < opponent_score:
        return -1
    elif player_score == opponent_score:
        return 0
    else:
        raise Exception('Scores were miscalculated')
