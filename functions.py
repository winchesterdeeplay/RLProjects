import itertools
import os
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from numpy.random import Generator

from net import Net


def make_video(env: gym.Env, net: Net) -> None:
    """Save video replay of the agent playing the game"""
    env = RecordVideo(env=env, video_folder="videos", name_prefix=f"video_{time.time()}")

    rewards = 0
    steps = 0
    done = False
    observation = env.reset()[0]
    env.start_video_recorder()

    while not done:
        action = net.action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))
    env.video_recorder.close()
    env.close()


def init_replay_buffer(env: gym.Env, min_replay_buffer_size: int, buffer_size: int) -> deque:
    """Initialize replay buffer with random transitions."""
    # Replay buffer помогает улучшить перформанс обучения за счет:
    # 1) сэмплим из env не последовательно - избавляемся от корреляции в последовательности
    # 2) получаем место откуда доставать batch
    # 3) во время обучения агенту нужно получать новый опыт(за счет эксперементов)
    # и максимизировать награду(чему научился)
    # 4) буфер позволяет обучаться на разных временных промежутках env

    replay_buffer = deque(maxlen=buffer_size)
    obs = env.reset()[0]
    for _ in range(min_replay_buffer_size):
        action = env.action_space.sample()
        next_obs, reward, done, _, _ = env.step(action)
        transition = (obs, action, reward, done, next_obs)
        replay_buffer.append(transition)
        obs = next_obs

        if done:
            obs = env.reset()[0]
    return replay_buffer


def epsilon_greedy(
    env: gym.Env,
    step: int,
    online_net: Net,
    obs: gym.core.ObsType,
    epsilon_decay: float,
    epsilon_start: float,
    epsilon_end: float,
    rng: Generator,
) -> int:
    """Epsilon-greedy action selection."""
    # выбираем лучшее изученное действие или случайное
    # с уменьшением шага уменьшаем вероятность случайного действия, те уменьшаем кол-во экспериментов
    esp = np.interp(step, [0, epsilon_decay], [epsilon_start, epsilon_end])
    if rng.random() < esp:
        return env.action_space.sample()
    else:
        return online_net.action(obs)


def train_dqn(
    env: gym.Env,
    online_net: Net,
    target_net: Net,
    optimizer: torch.optim.Adam,
    replay_buffer: deque,
    rew_buffer: deque,
    epsilon_decay: float,
    epsilon_start: float,
    epsilon_end: float,
    batch_size: int,
    gamma: float,
    target_update_freq: int,
    required_mean_reward: int,
    random_seed: int,
) -> None:
    """Train a DQN agent on the given environment."""

    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)

    obs = env.reset()[0]
    episode_reward = 0.0

    for step in itertools.count():
        action = epsilon_greedy(env, step, online_net, obs, epsilon_decay, epsilon_start, epsilon_end, rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = (obs, action, reward, done, next_obs)
        replay_buffer.append(transition)
        obs = next_obs

        episode_reward += reward

        if done:
            obs = env.reset()[0]
            rew_buffer.append(episode_reward)
            episode_reward = 0.0
        transitions = rng.choice(len(replay_buffer), size=batch_size, replace=False)
        transitions = tuple([replay_buffer[i] for i in transitions])
        obses, actions, rewards, dones, next_obses = zip(*transitions)

        obses_t = torch.tensor(obses, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(dim=-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(dim=-1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(dim=-1)
        next_obses_t = torch.tensor(next_obses, dtype=torch.float32)

        target_q_values = target_net(next_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        # immediate reward + discounted max future reward
        # баланс между наградой за текущее действии и за действие которое даст импакт в будущем
        targets = rewards_t + gamma * max_target_q_values * (1 - dones_t)

        q_values = online_net(obses_t)
        action_q_values = q_values.gather(dim=1, index=actions_t)

        loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)  # значения от -1 до 1 - l1, иначе ^2 штраф

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        if step % 1000 == 0:
            mean_reward = np.mean(rew_buffer)
            median_reward = np.median(rew_buffer)
            print("Step: %d, Mean reward: %.3f, Median reward: %.3f" % (step, mean_reward.item(), median_reward.item()))
            if mean_reward >= required_mean_reward and median_reward >= required_mean_reward:
                save_model(online_net, "model", rew_buffer, env, required_mean_reward)
                break


def save_model(net: Net, path: str, rew_buffer: deque, env: gym.Env, required_mean_reward: int) -> None:
    """Save the model and plot with reward history"""
    net.eval()
    os.makedirs(path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(path, "model.pth"))
    plt.plot(list(range(len(rew_buffer))), list(rew_buffer))
    plt.axhline(y=required_mean_reward, color="r", linestyle="--")
    plt.title("Reward history")
    plt.xlabel("Game Number")
    plt.ylabel("Reward")

    plt.savefig(os.path.join(path, "reward_history.png"))
    make_video(env, net)
