from stable_baselines3 import PPO


def train_agent(env, total_timesteps):
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=0.0005, n_steps=2048, batch_size=128, gamma=0.99, gae_lambda=0.95
    )
    model.learn(total_timesteps=total_timesteps)
    return model
