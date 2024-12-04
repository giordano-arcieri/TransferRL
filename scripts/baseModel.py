import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import numpy as np
import os

import bipedal_walker
from Simulation import Simulation
from PPOAgent import PPOAgent
import torch

def runSimulation(envToRun: str, model: PPO, eval_times: int = 1):
    sim = Simulation(env_id=envToRun, render=True)
    sim.runSimulation(model, eval_times)

        
def train(envToTrainOn: str, totalTimeSteps: int, showInterval: int, loadModelTo_FilePath: str):
    # Create the 16 environments
    policy_kwargs = dict(
        net_arch=[128, 128, 64, 64], # Number of neurons in each hidden layer
        activation_fn=torch.nn.ReLU # Activation function
    )
    
    model = PPOAgent(
        env_id = envToTrainOn,
        policy = 'MlpPolicy',
        n_steps = 2048,
        batch_size = 128,
        n_epochs = 6,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    
    for i in range(0, totalTimeSteps, showInterval):
        print(f"Training model: {i}/{totalTimeSteps}")
        model.learn(total_timesteps=showInterval)
        if(input("Press Y to show simulation").lower() == 'y'):
            runSimulation(envToTrainOn, model, 1)
    
    print(f"Training complete. Saving model to {loadModelTo_FilePath}")
    model.save(loadModelTo_FilePath)


def eval_model(envToEvaluate: str, getModelFrom_FilePath:str , times: int = 3):
    # Load and simulate the model
    runSimulation(envToEvaluate, PPO.load(getModelFrom_FilePath), times)


def main():
   

    traing: bool = True
    model_FilePath: str = "PPOModels/baseModel.zip"
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
