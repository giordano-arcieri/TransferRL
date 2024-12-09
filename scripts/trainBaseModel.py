import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import os

import Package.bipedal_walker as bipedal_walker
from Package.PPOAgent import PPOAgent
import torch
import json

        
def train(envToTrainOn: str, totalTimeSteps: int, interval: int, loadModelTo_FilePath: str, printMessages: bool = False):

    def show(inp: str): 
        if printMessages: 
            print(inp)
    
    # Create the 16 environments
    policy_kwargs = dict(
        net_arch=[64, 64, 64, 64, 64], # Number of neurons in each hidden layer [24(input layer), 64, 64, 64, 64, 64, 4(output layer)]
        activation_fn=torch.nn.ReLU # Activation function
    )
    
    model = PPOAgent(
        env_id = envToTrainOn,
        policy = 'MlpPolicy',
        n_steps = 2048,
        batch_size = 256,
        n_epochs = 6,
        gamma = 0.995,
        gae_lambda = 0.98,
        ent_coef = 0.03,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    
    # Array to keep track of all the statistics (rewards, episode lengths, etc.)  
    all_stats = []
    
    show("Succesfully initialized model. Training model...")
    for i in range(0, totalTimeSteps, interval):
        # Train
        show(f"Training model: {i}/{totalTimeSteps}")
        model.learn(total_timesteps=interval)
        
        # Retrieve statistics
        stats = model.get_episode_stats()
        all_stats.append(stats)
        
        # Save model and stats
        show(f"Saving model to testCheckpoint{i}.zip")
        model.save("testCheckpoint" + str(i) + ".zip")
        
        # Save statistics
        show(f"Saving stats to of iteration {i}. Stats: {stats}")
        with open("testCheckpoint" + str(i) + ".json", "w") as f:
            json.dump(all_stats, f, indent=4)
    
    show(f"Training complete. Saving model to {loadModelTo_FilePath}")
    model.save(loadModelTo_FilePath)


def main():
   
    # Define the model file path and the environment name
    model_FilePath: str = "test.zip"
    envName: str = 'BipedalWalkerEnvCustom-v0'

    # Check if the model file already exists
    if os.path.isfile(model_FilePath):
        print("Warning: The model file already exists. It will be overwritten.")
        if input("Press Enter to continue...") != "":
            raise Exception("User cancelled the operation.")

    # Check if the environment name is valid
    assert envName in gym.envs.registry, f"Invalid environment name: {envName}"

    # Train the model
    train(envToTrainOn=envName, 
            totalTimeSteps=100_000_000, 
            interval=1000000, 
            loadModelTo_FilePath=model_FilePath)
        



if __name__ == "__main__":
    main()
