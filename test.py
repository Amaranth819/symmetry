import gymnasium as gym
import time

if __name__ == '__main__':
    env = gym.make('Walker2d-v4', render_mode = 'human')

    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        time.sleep(0.01)
        env.render()
    env.close()