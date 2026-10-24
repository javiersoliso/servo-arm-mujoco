from gymnasium.envs.registration import register
from .reach_env import ServoArmReachEnv, RewardCfg, DomainRandCfg

register(id="ServoArmReach-v0", entry_point="servoarm.reach_env:ServoArmReachEnv",
         max_episode_steps=80)

__all__ = ["ServoArmReachEnv", "RewardCfg", "DomainRandCfg"]
