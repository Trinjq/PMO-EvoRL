from evorl.envs.mujoco_playground import MjxEnvAdapter
from evorl.envs.wrappers.training_wrapper import EpisodeWrapper, VmapAutoResetWrapper

from .mo_walker2d import MOWalker2d
from .preference import PreferenceConditionedEnv


def create_mo_walker2d_env(
    num_envs: int = 10, num_preference_workers: int = 10
) -> PreferenceConditionedEnv:
    """Build the 500-step, vectorized EvoRL environment."""
    env = EpisodeWrapper(MjxEnvAdapter(MOWalker2d()), episode_length=500)
    env = VmapAutoResetWrapper(env, num_envs=num_envs)
    return PreferenceConditionedEnv(env, num_envs, num_preference_workers)


__all__ = [
    "MOWalker2d",
    "PreferenceConditionedEnv",
    "create_mo_walker2d_env",
]
