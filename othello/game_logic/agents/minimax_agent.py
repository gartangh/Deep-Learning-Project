from game_logic.board import Board
from game_logic.agents.agent import Agent
from utils.color import Color
from utils.intermediate_reward_functions.difference_with_prev_board import heur
import random
import copy


class MinimaxAgent(Agent):
    def __init__(self, color: Color, immediate_reward_function):
        super().__init__(color, immediate_reward_function)
        self.name = "minimax"

    def __str__(self):
        return f'{self.name}{super().__str__()}'

    def minimax(self, board: Board, legal_directions: dict, maxLevel: int = 2, level: int = 0, prev_best_points: dict = None) -> tuple:
        player = self.color.value
        cur_best_points = None
        cur_best_move = None

        for move in legal_directions:
            new_board = copy.deepcopy(board)
            done = new_board.take_action(move, legal_directions[move], player)
            points = 0
            if level < maxLevel:
                new_legal_actions: dict = new_board.get_legal_actions(player)
                points, _ = self.minimax(new_board, new_legal_actions, maxLevel, level + 1, cur_best_points)
            else:
                points = self.immediate_reward_function(new_board, player)


    def get_next_action(self, board: Board, legal_directions: dict, maxLevel: int = 2, level: int = 0, prev_best_points: dict = None) -> tuple:
        if "pass" in legal_directions:
            return "pass", []

        action = random.choice(list(legal_directions.keys()))
        legal_actions = legal_directions[action]

        return action, legal_actions

