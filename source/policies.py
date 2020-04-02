import random

import numpy as np


class RandomPolicy:
    def __init__(self, board_size):
        self.board_size = board_size

    def get_action(self, possible_actions):
        return self.board_size ** 2 + 1 if len(possible_actions) == 0 else random.choice(possible_actions)

    def optimal(self, possible_actions):
        return self.get_action(possible_actions)


class EpsGreedyPolicy:
    def __init__(self, eps, board_size):
        self.eps = eps
        self.board_size = board_size

    def get_action(self, state, action_value_network, possible_actions):

        explore = random.random() < self.eps
        if explore:
            return random.choice(possible_actions)
        else:
            # In case of single window length, change shape from (h, w, channels) to (# samples, h, w, channels)
            state = state.reshape((1, ) + state.shape)
            q = action_value_network.predict(state)
            best_action = self.optimal(q, possible_actions)
            return best_action

    def optimal(self, q, possible_actions):
        q = q.flatten()
        n_actions = q.shape[0]
        q = q.sort()

        while n_actions >= 0:
            if q[n_actions] in possible_actions:
                return q[n_actions]
            n_actions -= 1

        return self.board_size ** 2 + 1


# start with large epsilon and gradually decrease it
class AnnealingEpsGreedyPolicy():
    def __init__(self, start_eps, end_eps, n_steps, board_size):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.n_steps = n_steps
        self.board_size = board_size
        self.inner_policy = EpsGreedyPolicy(self.start_eps, board_size)

        self.decisions_made = 0

    @property
    def current_eps_value(self):
        # Linear annealed: f(x) = ax + b.
        a = -float(self.start_eps - self.end_eps) / float(self.n_steps)
        b = float(self.start_eps)
        value = max(self.end_eps, a * float(self.decisions_made) + b)
        return value

    def get_current_eps_policy(self):
        self.inner_policy.eps = self.current_eps_value
        return self.inner_policy

    def get_action(self, state, action_value_network, possible_actions):
        self.decisions_made += 1
        return self.get_current_eps_policy().get_action(state, action_value_network, possible_actions)

    #TODO: remove this method and change with 2 policies in your DQN algo: one target_policy and one optimizing_policy
    def optimal(self, q, possible_actions):
        return self.get_current_eps_policy().optimal(q, possible_actions)

