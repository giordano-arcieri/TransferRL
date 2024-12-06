from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from typing import Union, Type, Dict, Any, Optional
from stable_baselines3.common.type_aliases import *
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


class PPOAgent(PPO):
    '''
    Just so that you dont risk not closing env
    This is a PPO agent that has its enviroment as a class variable
    '''
    def __init__(
        self, 
        env_id: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.env = make_vec_env(env_id, n_envs=16, wrapper_class=Monitor)
        super().__init__(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )


    def get_episode_stats(self):
        """
        Retrieve aggregated episode statistics from all environments.
        """
        # Get episode rewards and lengths from each environment
        rewards = self.env.env_method("get_episode_rewards")
        lengths = self.env.env_method("get_episode_lengths")

        # Flatten the lists
        all_rewards = [reward for env_rewards in rewards for reward in env_rewards]
        all_lengths = [length for env_lengths in lengths for length in env_lengths]

        # Compute statistics
        ep_rew_mean = np.mean(all_rewards) if all_rewards else 0
        ep_len_mean = np.mean(all_lengths) if all_lengths else 0

        return {"ep_rew_mean": ep_rew_mean, "ep_len_mean": ep_len_mean}
    
    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
            self.env = None
        