from collections import deque

import gymnasium as gym
import torch

from functions import train_dqn, init_replay_buffer
from net import Net

ENV_VERSION = "CartPole-v1"
RENDER_MODE = "rgb_array"
RANDOM_SEED = 42

# HyperParameters
REQUIRED_MEAN_REWARD = 500  # stop training if mean and median reward is >= REQUIRED_MEAN_REWARD
REWARD_BUFFER_SIZE = 50  # check mean and median reward score over last REWARD_BUFFER_SIZE episodes

HIDDEN_LAYER_SIZE = 512  # number of neurons in hidden layer
GAMMA = 0.99  # discount factor for future rewards
BUFFER_SIZE = 50000  # replay buffer size
MIN_REPLAY_SIZE = 1000  # minimum replay buffer size before training
BATCH_SIZE = 32  # minibatch size
LEARNING_RATE = 5e-4  # learning rate (alpha)

EPSILON_DECAY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQ = 500  # how often to update target network (in steps)

if __name__ == "__main__":
    # Initialize environment
    env = gym.make(ENV_VERSION, render_mode=RENDER_MODE)
    online_net = Net(env, HIDDEN_LAYER_SIZE)  # main net for train (получение нового опыта)
    target_net = Net(env, HIDDEN_LAYER_SIZE)  # helper net for target Q values (не дает сойти с ума первой)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters())

    # Initialize replay buffer
    replay_buffer = init_replay_buffer(env, MIN_REPLAY_SIZE, BUFFER_SIZE)
    # Initialize reward buffer
    rew_buffer = deque([0.0], maxlen=REWARD_BUFFER_SIZE)

    # Training loop
    train_dqn(
        env=env,
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        rew_buffer=rew_buffer,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ,
        required_mean_reward=REQUIRED_MEAN_REWARD,
        random_seed=RANDOM_SEED,
    )
