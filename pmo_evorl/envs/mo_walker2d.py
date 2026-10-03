"""PD-MORL's two-objective reward on Brax Walker2d dynamics."""

import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.walker2d import Walker2d


class MOWalker2d(Walker2d):
    """Walker2d with the vector reward used by the original PD-MORL code."""

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        return state.replace(reward=jnp.zeros(2, dtype=state.reward.dtype))

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        next_state = super().step(state, action)
        speed = next_state.metrics["reward_forward"] + next_state.metrics["reward_healthy"]
        energy = 4.0 - jnp.square(action).sum() + next_state.metrics["reward_healthy"]
        return next_state.replace(reward=jnp.stack((speed, energy)))
