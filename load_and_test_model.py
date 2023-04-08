from os.path import join

import gymnasium as gym
import torch

from main import HIDDEN_LAYER_SIZE, ENV_VERSION
from net import Net

MODEL_PATH = join("model", "model.pth")

if __name__ == "__main__":
    env = gym.make(ENV_VERSION, render_mode="human")
    model = Net(env, HIDDEN_LAYER_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()[0]
    while not done:
        action = model.action(observation)
        observation, reward, done, _, _ = env.step(action)
        steps += 1
        rewards += reward
        print("Num steps: {} reward {}: ".format(steps, rewards))
