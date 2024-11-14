import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import numpy as np
import time as t
import os

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


def runSimulation(envToRun: str, model: PPO, eval_times: int = 1):
    # Evaluate the model
        eval_env = gym.make(envToRun, render_mode='human')
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
                # input()
                obs, _reward, done, _T, _info = eval_env.step(action)

                eval_env.render()
        eval_env.close()

        
def train(envToTrainOn: str, totalTimeSteps: int, showInterval: int, loadModelTo_FilePath: str):
    # Create the 16 environments
    env = make_vec_env(envToTrainOn, n_envs=16)
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
    iteraions = totalTimeSteps // showInterval

    for i in range(iteraions):
        print("Training model for ", showInterval, " iterations. Total iterations: ", (i) * showInterval)
        model.learn(total_timesteps=showInterval)

        # if(i == iteraions - 1): 
        #     input("Press Enter to continue...")

        if(i != 0): # Don't evaluate the model on the first iteration
            print("evaluating model")
            runSimulation(envToTrainOn, model, 1)

    # Save the model
    model.save(loadModelTo_FilePath)
    env.close()

def eval_model(envToEvaluate: str, getModelFrom_FilePath:str , times: int = 3):
    # Load and simulate the model
    runSimulation(envToEvaluate, PPO.load(getModelFrom_FilePath), times)


def main():
   

    traing: bool = False
    model_FilePath: str = "PPOModels/4millionIt/LegReward/oppSignHardHipAndDoubleContactPen.zip"
    envName: str = 'BipedalWalkerEnvCustom-v0'

    # Check if the model file path is valid
    if not traing:  
        assert os.path.isfile(model_FilePath), f"File not found: {model_FilePath}"

    if traing:
        if os.path.isfile(model_FilePath):
            print("Warning: The model file already exists. It will be overwritten.")
            if input("Press Enter to continue...") != "":
                raise Exception("User cancelled the operation.")

    # Check if the environment name is valid
    assert envName in gym.envs.registry, f"Invalid environment name: {envName}"


    if traing:
        # Train a model using specified environment, total time steps, and show interval 
        # and save the model to the specified file path
        train(envToTrainOn=envName, 
              totalTimeSteps=4_000_000, 
              showInterval=1000000, 
              loadModelTo_FilePath=model_FilePath)
        
    
    else:
        # Evaluate the model using the specified environment and model file path
        eval_model(envToEvaluate=envName, getModelFrom_FilePath=model_FilePath, times=3)


if __name__ == "__main__":
    main()
