import random
import numpy as np
from keras import Sequential


class RandomPolicy:
    def __init__(self):
        pass

    def get_action(self, board: np.ndarray, legal_actions: dict) -> tuple:
        return random.choice(list(legal_actions.keys()))

    def optimal(self, board, legal_actions) -> tuple:
        return self.get_action(board, legal_actions)


# standard epsilon policy
class EpsGreedyPolicy:
    def __init__(self, eps: float, board_size: int):
        self.eps = eps
        self.board_size = board_size

    def get_action(self, board: np.ndarray, action_value_network: Sequential, legal_actions: dict) -> tuple:

        explore = random.random() < self.eps
        if explore:
            return random.choice(list(legal_actions.keys()))
        else:
            q_values = action_value_network.predict(board)
            best_action = self.optimal(q_values, legal_actions)
            return best_action

    def optimal(self, q_values, legal_actions) -> tuple:
        q_values = q_values.flatten() # reshape (1,x) to (x,)
        q_values = [(q_values[row * self.board_size + col], (row, col)) for row in range(self.board_size) for col in
                    range(self.board_size)]

        # get best legal action by sorting according to q value and taking the last legal entry
        q_values = sorted(q_values, key=lambda q: q[0])
        n_actions = len(q_values)-1
        while n_actions >= 0:
            if q_values[n_actions][1] in legal_actions:
                return q_values[n_actions][1]
            n_actions -= 1

        return 'pass'


# start with large epsilon and gradually decrease it
class AnnealingEpsGreedyPolicy:
    def __init__(self, start_eps, end_eps, n_steps, board_size):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.n_steps = n_steps
        self.inner_policy = EpsGreedyPolicy(self.start_eps, board_size)

        self.decisions_made = 0

    @property
    def current_eps_value(self) -> float:
        # Linear annealed: f(x) = ax + b.
        a = -float(self.start_eps - self.end_eps) / float(self.n_steps)
        b = float(self.start_eps)
        value = max(self.end_eps, a * float(self.decisions_made) + b)
        return value

    def get_current_eps_policy(self) -> EpsGreedyPolicy:
        self.inner_policy.eps = self.current_eps_value
        return self.inner_policy

    def get_action(self, board: np.ndarray, action_value_network: Sequential, legal_actions: dict) -> tuple:
        self.decisions_made += 1
        return self.get_current_eps_policy().get_action(board, action_value_network, legal_actions)

    #TODO: remove this method and change with 2 policies in your DQN algo: one target_policy and one optimizing_policy
    def optimal(self, q_values: np.ndarray, legal_actions: dict) -> tuple:
        return self.get_current_eps_policy().optimal(q_values, legal_actions)

