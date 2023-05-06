import time
from os.path import join

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from main import HIDDEN_LAYER_SIZE, ENV_VERSION, RANDOM_SEED
from net import Net

MODEL_PATH = join("model", "model.pth")

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    env = gym.make(ENV_VERSION, render_mode="human")
    # env.seed(RANDOM_SEED)
    model = Net(env, HIDDEN_LAYER_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()[0]
    env = RecordVideo(env=env, video_folder="videos", name_prefix=f"video_{time.time()}")
    while not done:
        action = model.action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        rewards += reward
        print("Num steps: {} reward {}: ".format(steps, rewards))
    env.close()
