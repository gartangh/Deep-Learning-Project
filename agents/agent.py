from utils.replay_buffer import ReplayBuffer
import numpy as np


class Agent:
    def __init__(self, color):
        self.color = color
        self.wins = 0
        self.score = 0
        self.num_games_won = 0
        self.play_mode = True

        # evaluation function
        self._heurBoard = np.asarray(
            [[100, -25, 10, 5, 5, 10, -25, 100],
             [-25, -25, 2, 2, 2, 2, -25, -25],
             [10, 2, 5, 1, 1, 5, 2, 10],
             [5, 2, 1, 2, 2, 1, 2, 5],
             [5, 2, 1, 2, 2, 1, 2, 5],
             [10, 2, 5, 1, 1, 5, 2, 10],
             [-25, -25, 2, 2, 2, 2, -25, -25],
             [100, -25, 10, 5, 5, 10, -25, 100]])
        self._corner = np.asarray(
            [[100, -25, 10],
             [-25, -25, 2],
             [10, 2, 5]]
        )
        self._edge = np.asarray([5, 2, 1])
        self._specialHeurBoard = None

    def __str__(self):
        return f'Agent: color={self.color.name}, score={self.score}'

    def next_action(self, board, legal_actions):
        raise NotImplementedError

    #@staticmethod
    #def immediate_reward(board, prev_board, turn):
    #    # 1 + number of turned disks
    #    difference = len(np.where(board == turn)[0]) - len(np.where(prev_board == turn)[0])
    #    return difference

    def evaluateBoard(self, board, turn):
        evaluation_score = 0

        n = len(board)
        coinsBoardA = np.where(board == turn, 1, 0)
        coinsBoardB = np.where(board == (1 - turn), -1, 0)
        coinsBoard = np.add(coinsBoardA, coinsBoardB)

        if n == 8:  # in standard length
            pointBoard = np.multiply(coinsBoard, self._heurBoard)
        else:
            # create heurboard
            if self._specialHeurBoard == None or len(self._specialHeurBoard) != n:
                corner_nw = self._corner
                n2 = n // 2  # should be all right -> n has to be even!
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
        board_score = self.evaluateBoard(board, turn)
        prev_board_score = self.evaluateBoard(prev_board, turn)
        reward = board_score - prev_board_score
        return reward

    def set_play_mode(self, mode: bool):
        self.play_mode = False


class TrainableAgent(Agent):
    def __init__(self, color, board_size):
        super().__init__(color)
        self.board_size = board_size
        self.episode_rewards = []
        self.training_errors = []
        self.play_mode = True

        replay_buffer_size = 100_000
        self.replay_buffer = ReplayBuffer(size=replay_buffer_size)

    def __str__(self):
        return f'TrainableAgent: color={self.color.name}, score={self.score}'

    def create_model(self, verbose=False, lr=0.00025):
        raise NotImplementedError

    def set_play_mode(self, mode: bool):
        self.play_mode = mode

    def train(self, state, action, reward, next_state, terminal, render=False):
        raise NotImplementedError

    def next_action(self, board, legal_actions):
        raise NotImplementedError

    def q_learn_mini_batch(self):
        raise NotImplementedError

    def update_target_network(self):
        raise NotImplementedError

    def _can_start_learning(self):
        raise NotImplementedError

    def _persist_weights_if_necessary(self):
        raise NotImplementedError

    def board_to_nn_input(self, board):
        # Transform board (numpy ndarray) to a numpy 1D array
        # that can serve as input to the neural network.
        # There are 2 nodes per board location:
        #   - 1 node that is 0 if location does not contain black, else 1
        #   - 1 node that is 0 if location does not contain white, else 1
        whites = np.zeros((1, self.board_size ** 2))
        blacks = np.zeros((1, self.board_size ** 2))
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == 1: # white
                    whites[0][row*self.board_size+col] = 1
                elif board[row][col] == 0: # black
                    blacks[0][row*self.board_size+col] = 1

        # concatenate the 2 arrays
        return np.hstack([blacks, whites])
