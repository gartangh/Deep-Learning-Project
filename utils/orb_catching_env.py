import random

import PIL
import numpy as np
from IPython.display import clear_output
from PIL import Image
from gym.spaces import Discrete
from orb_catching_game.game import OrbCatchingGame
from orb_catching_game.robot import Robot


class OrbCatchingEnvironment():
    obstacle_positions = [((1, 7), (3, 2), (23, 30)),
                          ((1, 1), (4, 4), (23, 30)),
                          ((1, 7), (6, 3), (23, 30)),
                          ((4, 7), (3, 2), (23, 30)),

                          ((2, 3), (8, 5), (23, 30)),
                          ((5, 4), (4, 6), (23, 30)),
                          ((8, 1), (3, 3), (23, 30)),
                          ((6, 7), (4, 2), (23, 30))

                          ]

    DISPLAY_SIZE = (128, 128)

    def __init__(self, reward_function=lambda _: 0, level=1):
        super().__init__()
        self.level = level
        self.viewer = None
        self.steps_taken = 0

        self.game = OrbCatchingGame(level)
        while self.game.on_init() == False:
            # wait for it
            pass

        self.action_space = Discrete(4)

        self.observation_space = None

        self.state = None

        self.game.step()

        self.input_shape = tuple(self.game.size)

        self.n_bonus_orbs_rewards_assigned = 0

        self.reward = reward_function

        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.game.step(action)

        obs = self.get_current_state()
        reward = self.reward(self.game)
        done = self.game.normal_orb_is_caught()

        self.steps_taken += 1
        self.state = obs
        return obs, reward, done, {}

    """Resets the state of the game. This should be called manually after an environment.step() returns a terminal state.
    Args:
        with_robot (boolean): if True, the robot will be spawned at a different place, 
                                else the robot will remain at the same location.
    Returns:
        the state of the resetted environment.
    """

    def reset(self, with_robot=False):
        obstacle_positions = None
        if self.level == 2:
            obstacle_positions = OrbCatchingEnvironment.obstacle_positions[
                random.randint(0, len(OrbCatchingEnvironment.obstacle_positions) - 1)]

        self.game.reset(obstacle_positions=obstacle_positions)
        if with_robot:
            self.game.respawn_robot()

        self.game.step(Robot.ACTION_NOTHING)
        self.n_bonus_orbs_rewards_assigned = 0
        self.steps_taken = 0

        self.state = self.get_current_state()
        return self.state

    def render(self, mode='rgb'):
        frame = self.get_current_state()
        img = Image.fromarray(frame, 'RGB')
        img = np.array(img)

        display(PIL.Image.fromarray(img).resize(OrbCatchingEnvironment.DISPLAY_SIZE, Image.NEAREST))
        clear_output(wait=True)
        return img

    def close(self):
        self.game.on_cleanup()
        if self.viewer is not None: self.viewer.close()

    def get_current_state(self):
        frame = self.game.get_last_frame()
        return frame

    @property
    def orbs_caught(self):
        return self.n_bonus_orbs_rewards_assigned + self.game.normal_orb_is_caught()
