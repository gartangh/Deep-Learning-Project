import numpy as np

from game_logic.board import Board


def difference_with_prev_board(board: Board, color_value: int) -> float:
    # 1 + number of turned disks
    difference: int = len(np.where(board.board == color_value)[0]) - len(
        np.where(board.prev_board == 1 - color_value)[0])
    return difference * 1.0


# Initialisation of the arrays for HEUR
_edge = np.asarray([5, 2, 1])
_corner = np.asarray(
    [[100, -25, 10],
     [-25, -25, 2],
     [10, 2, 5]])
_heurBoard = np.asarray(
    [[100, -25, 10, 5, 5, 10, -25, 100],
     [-25, -25, 2, 2, 2, 2, -25, -25],
     [10, 2, 5, 1, 1, 5, 2, 10],
     [5, 2, 1, 2, 2, 1, 2, 5],
     [5, 2, 1, 2, 2, 1, 2, 5],
     [10, 2, 5, 1, 1, 5, 2, 10],
     [-25, -25, 2, 2, 2, 2, -25, -25],
     [100, -25, 10, 5, 5, 10, -25, 100]])
_specialHeurBoard = None


def _evaluate_board(board, color_value):
    global _edge, _corner, _heurBoard, _specialHeurBoard
    evaluation_score = 0
    n = len(board)
    coinsBoardA = np.where(board == color_value, 1, 0)
    coinsBoardB = np.where(board == (1 - color_value), -1, 0)
    coinsBoard = np.add(coinsBoardA, coinsBoardB)

    if n == 8:  # in standard length
        pointBoard = np.multiply(coinsBoard, _heurBoard)
    else:
        # create heurboard
        if _specialHeurBoard is None or len(_specialHeurBoard) != n:
            corner_nw = _corner
            n2 = n // 2  # should be all right -> n has to be even!
            if n2 <= len(_corner):
                corner_nw = _corner[:n2, :n2]
            else:
                corner_nw = np.vstack([corner_nw, _edge])
                corner_nw = np.column_stack([corner_nw, np.append(_edge, 1)])
                corner_nw = np.pad(corner_nw, (0, n2 - len(_corner)), "edge")
                corner_nw[n2 - 1, n2 - 1] = 2
            _specialHeurBoard = np.pad(corner_nw, (0, n2), "symmetric")

        pointBoard = np.multiply(coinsBoard, _specialHeurBoard)

    evaluation_score = np.sum(pointBoard)
    return evaluation_score


def heur(board: Board, color_value: int) -> float:
    board_score = _evaluate_board(board.board, color_value)
    prev_board_score = _evaluate_board(board.prev_board, color_value)
    reward = board_score - prev_board_score
    return reward
