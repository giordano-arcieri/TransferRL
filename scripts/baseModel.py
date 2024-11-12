import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import numpy as np

import bipedal_walker

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


def runSimulation(model: PPO, eval_times: int = 1):
    # Evaluate the model
        eval_env = gym.make('BipedalWalkerEnvCustom-v0', render_mode='human')
        for i in range(eval_times):
            obs, _ = eval_env.reset()

            done = False
            time = 0
            print("Running simulation: ", i + 1)
            while not done and time < 1000:
                if(time % 100 == 0):
                    print("Time: ", time)
                time += 1
                action, _states = model.predict(obs)
                obs, _reward, done, _T, _info = eval_env.step(action)

                eval_env.render()
        eval_env.close()


def main():
   

    traing = False
    model_name = "PPOModels/ppo4million_legRewoppSignHipandKnee.zip"

    if traing:
        
        env = make_vec_env('BipedalWalkerEnvCustom-v0', n_envs=16)
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

        # Train the model
        Total_timesteps = 4_000_000
        show_times = 4
        iteraions = Total_timesteps // show_times

        for i in range(4):
            print("Training model for ", iteraions, " iterations")
            model.learn(total_timesteps=iteraions)

            if(i == 3): # Don't evaluate the model on the first iteration
                print("evaluating model")
                input("Press Enter to continue...")
                runSimulation(model, 3)
        
            
        # Save the model
        model.save(model_name)
        env.close()

    
    else:
        # Evaluate the model
        eval_env = gym.make('BipedalWalkerEnvCustom-v0', render_mode='human')
        # Load the model
        model = PPO.load(model_name, env=eval_env)
        for i in range(30):
            obs, _ = eval_env.reset()

            done = False
            time = 0
            while not done and time < 1000:
                time += 1
                action, _states = model.predict(obs)
                obs, _reward, done, _T, _info = eval_env.step(action)

                eval_env.render()

        eval_env.close()

if __name__ == "__main__":
    main()
