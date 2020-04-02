import gym
import random
import numpy as np

from agents import DQNAgent


def create_model():
    return None

env = gym.make('Reversi8x8-v0')
env.reset()

end_eps = 0.2
agent1 = DQNAgent(env, create_model(), end_eps, max_n_episodes=100)
agent2 = DQNAgent(env, create_model(), end_eps, max_n_episodes=100)

# TODO change reversi.py to accomodate 2 agents
# by deleting all references to "self.opponent" and "self.player_color"
# and modifying where necessary