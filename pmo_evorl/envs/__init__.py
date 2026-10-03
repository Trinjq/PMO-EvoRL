from evorl.envs.mujoco_playground import MjxEnvAdapter
from evorl.envs.wrappers.training_wrapper import EpisodeWrapper, VmapAutoResetWrapper

from .mo_walker2d import MOWalker2d


def create_mo_walker2d_env(num_envs: int = 1) -> VmapAutoResetWrapper:
    """Build the 500-step, vectorized EvoRL environment."""
    env = EpisodeWrapper(MjxEnvAdapter(MOWalker2d()), episode_length=500)
    return VmapAutoResetWrapper(env, num_envs=num_envs)


__all__ = ["MOWalker2d", "create_mo_walker2d_env"]
