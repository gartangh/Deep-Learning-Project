import os
import time

import numpy as np

from policies import AnnealingEpsGreedyPolicy, RandomPolicy, EpsGreedyPolicy
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, env, nn_model, end_eps, **kwargs):

        #######################
        # DQN setup
        #######################
        self.env = env

        replay_buffer_size = kwargs.get('replay_buffer_size', 100_000)
        self.replay_buffer = ReplayBuffer(size=replay_buffer_size)
        if kwargs.get('path_to_prev_buffer'):
            self.replay_buffer.load(kwargs.get('path_to_prev_buffer'))

        self.policy = AnnealingEpsGreedyPolicy(0.99, end_eps, 75_000, env.board_size)
        self.test_policy = EpsGreedyPolicy(end_eps, env.board_size)
        self.buffer_filling_policy = RandomPolicy(env.board_size)

        #######################
        # Duration parameters
        #######################
        self.max_n_episodes = kwargs.get('max_n_episodes', 20)

        #######################
        # Network & training
        #######################
        self.n_steps_start_learning = 25_000 if not kwargs.get('path_to_prev_buffer') else 0
        self.discount_factor = 0.99

        if nn_model is None:
            print('Error: this model needs a Q-network!')
        else:
            self.action_value_network = nn_model()
            self.target_network = nn_model()
            if kwargs.get('path_to_prev_weights'):
                [network.load_weights(kwargs.get('path_to_prev_weights')) for network in
                 (self.action_value_network, self.target_network)]

        self.mini_batch_size = kwargs.get('mini_batch_size', 32)
        self.n_training_cycles = 0
        self.learning_frequency = 1  # 4 in atari
        self.n_replay_episodes = 1

        self.weight_persist_path = 'network_weights/'
        if not os.path.exists(self.weight_persist_path):
            os.makedirs(self.weight_persist_path)

        self.target_network_update_freq = kwargs.get('target_network_update_freq', 0.01 * self.replay_buffer.size)

        #######################
        # Bookkeeping values
        #######################
        self.n_steps = 0
        self.n_episodes = 0

        self.persist_weights_every_n_times_trained = 10e3
    
    def train(self, render=False):
        while self.n_episodes < self.max_n_episodes:
            episode_rewards = []
            training_errors = []

            self.env.reset()

            terminal = False
            steps_this_episode = 0
            while not terminal:
                state = self.env.state
                action = self.act(state)
                next_state, reward, terminal, _ = self.env.step(action)
                
                steps_this_episode += 1

                episode_rewards.append(reward)

                self.replay_buffer.add(state, action, reward, next_state, terminal)

                if self._can_start_learning():
                    for i in range(self.n_replay_episodes):
                        training_error = self.q_learn_mini_batch(self.replay_buffer, self.action_value_network, self.target_network)
                        training_errors.append(training_error)
                        self._persist_weights_if_necessary()

                self.n_steps += 1

                if self.n_steps % self.target_network_update_freq == 1:
                    self.update_target_network()

                if render:
                    self.env.render()

            self.n_episodes += 1

        return
    
    def act(self, state):
        possible_actions = self.env.possible_actions
        if self.n_steps < self.n_steps_start_learning:
            action = self.buffer_filling_policy.get_action(possible_actions)
        else:
            action = self.policy.get_action(state, self.action_value_network, possible_actions)
            
        return action

    def q_learn_mini_batch(self, replay_buffer, action_value_network, target_network):
        self.n_training_cycles += 1

        # Sample a mini batch from our buffer
        mini_batch = replay_buffer.sample(self.mini_batch_size)

        # Extract states and subsequent states from mini batch
        states = np.array([sample[0] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])

        # We predict the Q values for all current states in the batch using the online network to use as targets.
        # The Q values for the state-action pairs that are being trained on in the mini batch will be overridden with
        #       the real target r + gamma * argmax[Q(s', a); target network]
        targets = action_value_network.predict(states)

        # Predict all next Q values for the subsequent states in the batch using the target network
        target_q_values_next_states = target_network.predict(next_states)

        for sample_nr, transition in enumerate(mini_batch):
            state, action, reward, next_state, terminal = transition
            q_values = target_q_values_next_states[sample_nr].flatten()

            best_next_action = np.argmax(q_values) 

            # Insert real target value using r + gamma * argmax[Q(s', a); target network] if not terminal
            q_value_next_state = q_values[best_next_action]
            if terminal:
                # This is important as it gives you a stable, consistent reward which will not fluctuate (i.e. depend on the target network)
                targets[sample_nr, action] = reward
            else:
                targets[sample_nr, action] = reward + self.discount_factor * q_value_next_state

        training_loss = action_value_network.train_on_batch(states, targets)
        return training_loss

    def update_target_network(self):
        target_weights = self.action_value_network.get_weights()
        self.target_network.set_weights(target_weights)

    # test run where states are rendered
    def test(self, render=False, episodes=100):
        n_test_episodes = 0
        while n_test_episodes < episodes:
            episode_rewards = []

            self.env.reset()

            terminal = False
            steps_this_episode = 0
            while not terminal:
                state = self.env.state
                possible_actions = self.env.possible_actions
                action = self.test_policy.get_action(state, self.action_value_network, possible_actions)
                next_state, reward, terminal, _ = self.env.step(action)

                steps_this_episode += 1

                episode_rewards.append(reward)

                self.n_steps += 1

                if render:
                    self.env.render()
                    time.sleep(0.5)
            
            n_test_episodes += 1

            self.n_episodes += 1
            
        return

    def _can_start_learning(self):
        return self.n_steps > self.n_steps_start_learning and \
               self.n_steps % self.learning_frequency == 0 and \
               self.replay_buffer.n_obs > self.mini_batch_size

    def _persist_weights_if_necessary(self):
        if self.n_training_cycles % self.persist_weights_every_n_times_trained == 0:
            path = '{}/weights_level_{}.h5f'.format(self.weight_persist_path, self.env.level)
            self.action_value_network.save_weights(path, overwrite=True)
            
            path = 'replay_buffers'
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, 'replay_buffer_level_{}.pkl'.format(self.env.level))
            self.replay_buffer.persist(file_path)

