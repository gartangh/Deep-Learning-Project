from utils.help_functions import *
from game_logic.board import Board
from game_logic.agents.agent import Agent
from utils.color import Color
import random


class MinimaxAgent(Agent):
    def __init__(self, color: Color, immediate_reward_function):
        super().__init__(color, immediate_reward_function)
        self.name = "minimax"

    def __str__(self):
        return f'{self.name}{super().__str__()}'

    def get_next_action(self, board: Board, legal_directions: dict, maxLevel: int = 2, level: int = 0, prev_best_points: dict = None) -> tuple:
        player = self.color.value
        cur_best_points = None
        cur_best_move = None

        for move in legal_directions:
            pass

        action = random.choice(list(legal_directions.keys()))
        legal_actions = legal_directions[action]

        return action, legal_actions

