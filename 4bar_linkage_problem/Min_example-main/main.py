import gymnasium as gym
import gym_env

env_ = gym.make("Minimal_env", render_mode='human', max_episode_steps=100)
observation, info = env_.reset(seed=1)


for _ in range(1000):
   action = env_.action_space.sample()
   observation, reward, terminated, truncated, info = env_.step(action)

   if terminated or truncated:
      print("terminated: ", terminated, " or truncated: ", truncated)
      observation, info = env_.reset()
