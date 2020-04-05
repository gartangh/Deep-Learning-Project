import random

from game_logic.agents import Agent


class RandomAgent(Agent):

    def __init__(self, color):
        super().__init__(color)
        self.name = 'Random'

    def __str__(self):
        return f'{self.name}{super().__str__()}'

    def next_action(self, board, legal_actions):
        action = random.choice(list(legal_actions.keys()))
        legal_actions = legal_actions[action]

        return action, legal_actions
