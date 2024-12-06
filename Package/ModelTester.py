
import os
import gymnasium as gym
from stable_baselines3 import PPO
from Package.Simulation import Simulation 
from typing import Dict

'''
Class that returns the mean_reward of the model for a given environment

returns:    
    mean_reward: float
        The mean reward of the model for the given environment
'''

BASE_SIM_PROPERTIES = { 'bumpiness': 1.0, 'friction': 0.7 }
BUMPY_SIM_PROPERTIES = { 'bumpiness': 10.0, 'friction': 0.7 }
SLIPPERY_SIM_PROPERTIES = { 'bumpiness': 1.0, 'friction': 0.1 }
HARD_SIM_PROPERTIES = { 'bumpiness': 10.0, 'friction': 0.1 }


class ModelTester:
    """
        Initializes the TestModel class by loading the model and preparing the environment.

        Parameters:
        - model_path (str): Path to the model file.
        - testing_env_id (str): Environment ID to test the model on.
        - testing_sim (str): Simulation type, e.g., 'base', 'bumpy', 'slippery', 'hard'. Default is 'base'.
    """        
    def __init__(self, model_path: str, testing_env_id: str, testing_sim: str):
        
        # Load model from file_path
        self.model = PPO.load(model_path)
        
        # Assert that the simulation type is valid
        if testing_sim not in ['base', 'bumpy', 'slippery', 'hard']:
            raise ValueError(f"Invalid simulation type: {testing_sim}. Must be one of ['base', 'bumpy', 'slippery', 'hard'].")
        
        # Set the simulation properties based on the simulation type
        if testing_sim == 'base':
            bumpiness = BASE_SIM_PROPERTIES['bumpiness']
            friction = BASE_SIM_PROPERTIES['friction']
        elif testing_sim == 'bumpy':
            bumpiness = BUMPY_SIM_PROPERTIES['bumpiness']
            friction = BUMPY_SIM_PROPERTIES['friction']
        elif testing_sim == 'slippery':
            bumpiness = SLIPPERY_SIM_PROPERTIES['bumpiness']
            friction = SLIPPERY_SIM_PROPERTIES['friction']
        elif testing_sim == 'hard':
            bumpiness = HARD_SIM_PROPERTIES['bumpiness']
            friction = HARD_SIM_PROPERTIES['friction']
        else:
            raise RuntimeError("Invalid simulation type.")


        # Initialize the simulation environment
        self.sim = Simulation(env_id=testing_env_id, render=True, bumpiness=bumpiness, friction=friction)
    
    """
        Run the simulation multiple times and compute the mean reward.

        Parameters:
        - eval_times (int): Number of evaluation runs. Default is 300.

        Returns:
        {
            mean_reward (float): Mean reward obtained over all runs.
            mean_steps (float): Mean number of steps taken over all runs.
        }
    """ 
    def run(self, eval_times: int = 300) -> Dict[str, float]:
        return self.sim.runSimulation(self.model, eval_times=eval_times, maxTime=10000)
        