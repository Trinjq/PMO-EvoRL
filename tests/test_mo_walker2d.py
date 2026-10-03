"""Minimal executable contract check for MOWalker2d."""

import jax
import jax.numpy as jnp

from pmo_evorl.envs import MOWalker2d


def main() -> None:
    env = MOWalker2d()
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))

    assert state.obs.shape == (17,)
    assert state.reward.shape == (2,)

    action = jnp.full((env.action_size,), 2.0)
    next_state = step(state, action)
    expected = jnp.array(
        [
            next_state.metrics["reward_forward"] + next_state.metrics["reward_healthy"],
            4.0 - env.action_size + next_state.metrics["reward_healthy"],
        ]
    )
    assert next_state.reward.shape == (2,)
    assert bool(jnp.allclose(next_state.reward, expected))
    print("MOWalker2d contract check passed:", next_state.reward)


if __name__ == "__main__":
    main()
