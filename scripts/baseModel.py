import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import numpy as np

class Simulation():
    def __init__(self, env_id: str = 'BipedalWalkerEnvCustom-v0', render: bool = False, bumpiness: float = 0.0, friction: float = 0.0):
        if(render):
            self.env = gym.make(env_id, render_mode='human', bumpiness=bumpiness, friction=friction)
        else:
            self.env = gym.make(env_id, bumpiness=bumpiness, friction=friction)

    def getEnv(self) -> gym.Env:
        return self.env

    def __del__(self):
        self.env.close()
        self.env = None

class Agent():
    def __init__(self):
        pass

    def train(self, env: gym.Env):
        pass


def main():
    env = make_vec_env('BipedalWalker-v3', n_envs=16)
    model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 2048,
    batch_size = 128,
    n_epochs = 6,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

    traing = False
    model_name = "ppo5millionIt-v3.zip"

    if traing:
        # Train the model
        itaraions = 5_000_000
        model.learn(total_timesteps=itaraions)

        # Save the model
        model.save(model_name)
    

    else:
        # Load the model
        model.load(model_name)
        # Evaluate the model
        eval_env = gym.make('BipedalWalker-v3', render_mode='human')
        while True:
            obs, _ = eval_env.reset()

            done = False
            time = 0
            while not done and time < 1000:
                time += 1
                action, _states = model.predict(obs)
                obs, _reward, done, _T, _info = eval_env.step(action)

                eval_env.render()


if __name__ == "__main__":
    main()
