from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import PPO

from maze_envirement import MazeEnvironment
from model import train_agent

env = MazeEnvironment()
trained_agent = train_agent(env, total_timesteps=100)

trained_agent.save("trained_maze_agent")
loaded_agent = PPO.load("trained_maze_agent")

obs = env.reset()
done = False

video_recorder = VideoRecorder(env, "video_output.mp4")

if __name__ == "__main__":
    while not done:
        action, _ = loaded_agent.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        video_recorder.capture_frame()

video_recorder.close()
env.close()
