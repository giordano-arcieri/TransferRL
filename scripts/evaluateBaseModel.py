import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import numpy as np
import os

import TransferRL.bipedal_walker as bipedal_walker
import json
from TransferRL.Simulation import Simulation
from TransferRL.PPOAgent import PPOAgent
import torch

def runSimulation(envToRun: str, model: PPO, eval_times: int = 1):
    sim = Simulation(env_id=envToRun, render=True)
    sim.runSimulation(model, eval_times)

        

def eval_model(envToEvaluate: str, getModelFrom_FilePath:str , times: int = 3):
    # Load and simulate the model
    runSimulation(envToEvaluate, PPO.load(getModelFrom_FilePath), times)


