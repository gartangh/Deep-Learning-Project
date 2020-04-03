import numpy as np 
from agents.agent import Agent
import random

class MinimaxAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = "minimax"
        self._heurBoard = np.asarray(
                          [[100, -25, 10, 5, 5, 10, -25, 100],
                           [-25, -25,  2, 2, 2,  2, -25, -25],
                           [ 10,   2,  5, 1, 1,  5,   2,  10],
                           [  5,   2,  1, 2, 2,  1,   2,   5],
                           [  5,   2,  1, 2, 2,  1,   2,   5],
                           [ 10,   2,  5, 1, 1,  5,   2,  10],
                           [-25, -25,  2, 2, 2,  2, -25, -25],
                           [100, -25, 10, 5, 5, 10, -25, 100]])
        self._corner = np.asarray(
            [[100, -25, 10],
             [-25, -25,  2],
             [ 10,   2,  5]]
        )
        self._edge = np.asarray([5, 2, 1])
        self._specialHeurBoard = None
    
    def __str__(self):
	    return f'{self.name}{super().__str__()}'
    
    def next_action(self, legal_actions):
	    action = random.choice(list(legal_actions.keys()))
	    legal_actions = legal_actions[action]

	    return action, legal_actions

    def evaluateBoard(self, board, turn):
        evaluation_score = 0

        n = len(board)
        coinsBoardA = np.where(board == turn, 1, 0)
        coinsBoardB = np.where(board == (turn - 1), -1, 0)
        coinsBoard = np.add(coinsBoardA, coinsBoardB)

        if n == 8: #in standard length
            pointBoard = np.multiply(coinsBoard, self._heurBoard)
        else: 
            # create heurboard
            if self._specialHeurBoard == None or len(self._specialHeurBoard) != n:
                corner_nw = self._corner
                n2 = n//2 # should be all right -> n has to be even!
                if n2 <= len(self._corner):
                    corner_nw = self._corner[:n2, :n2]
                else:
                    corner_nw = np.vstack([corner_nw, self._edge])
                    corner_nw = np.column_stack([corner_nw, np.append(self._edge, 1)])
                    corner_nw = np.pad(corner_nw, (0, n2 - len(self._corner)), "edge")
                    corner_nw[n2 - 1, n2 - 1] = 2
                self._specialHeurBoard = np.pad(corner_nw, (0, n2), "symmetric")
            
            pointBoard = np.multiply(coinsBoard, self._specialHeurBoard)

        evaluation_score = np.sum(pointBoard)
        return evaluation_score
            

    def immediate_reward(self, board, prev_board, turn):
        reward = self.evaluateBoard(board, turn) - self.evaluateBoard(prev_board, turn)
        return reward