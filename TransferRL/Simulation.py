'''

'''

import gymnasium as gym
from typing import Dict

class Simulation:
    def __init__(self, env_id: str = 'BipedalWalkerEnvCustom-v0', render: bool = False, bumpiness: float = 1.0, friction: float = 0.7):
        render_mode = 'human' if render else None
        self.env = gym.make(env_id, render_mode=render_mode, bumpiness=bumpiness, friction=friction)

    def runSimulation(self, model, eval_times: int = 1, maxTime: int = 1000) -> Dict[str, float]:
        stats: Dict[str, float] = {"mean_reward": 0.0, "mean_steps": 0.0}
        for i in range(eval_times):
            obs, _ = self.env.reset()

            done: bool = False
            time: int = 0
            
            print("Running simulation: ", i + 1)
            while not done and time < maxTime:
                if(time % 100 == 0):
                    print("Time: ", time, "/", maxTime)
                time += 1
                action, _states = model.predict(obs)
                obs, reward, done, _T, _info = self.env.step(action)
                
                stats["mean_reward"] += reward
                stats["mean_steps"] += 1
                
                self.env.render()

        stats["mean_reward"] /= eval_times
        stats["mean_steps"] /= eval_times
        return stats
    
    
    def __del__(self):
        self.env.close()
        self.env = None

