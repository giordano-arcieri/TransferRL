
import os
import gymnasium as gym
from stable_baselines3 import PPO
from Simulation import Simulation 

class TestModel:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.sim = Simulation(env_id="BipedalWalker-v3", render=True)

    def load_model(self, model_path):
        self.model = model_path

    def predict(self, data):
        return self.model.predict(data)
    