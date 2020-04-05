from game_logic.agents import Agent
import random


class MinimaxAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = "minimax"

    def next_action(self, board, legal_actions: dict, maxLevel: int = 2, level: int = 0, prev_best_points: dict = None):
        action = random.choice(list(legal_actions.keys()))
        legal_actions = legal_actions[action]

        return action, legal_actions
