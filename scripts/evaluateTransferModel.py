import gymnasium as gym
from Package import bipedal_walker
import os
from Package.ModelTester import ModelTester
from typing import Dict

def main():
    
    # Define
    testing_env_id = "BipedalWalkerEnvCustom-v0"
    testing_sim = 'slippery'
    model_path = '../PPOModels/BaseModel/baseModel.zip'

    
    path_to_models = ['../PPOModels/TransferModel/bumpy/TransferModel_bumpy.zip', 
                      '../PPOModels/TransferModel/slippery/TransferModel_slippery.zip', 
                      '../PPOModels/TransferModel/hard/TransferModel_hard.zip']
    
    for model_path in path_to_models:
        # Check if the model exists
        assert os.path.exists(model_path), f"Model not found at {model_path}"
        
        # Check if the environment name is valid
        assert testing_env_id in gym.envs.registry, f"Invalid environment name: {testing_env_id}"
    
        # Evaluate the model
        model_tester = ModelTester(model_path, testing_env_id, testing_sim)
        stats: Dict[str, float] = model_tester.run()
        
        # Save the statistics
        print(f"Stats of \"{model_path}\": {stats}")
   

if __name__ == '__main__':
    main()