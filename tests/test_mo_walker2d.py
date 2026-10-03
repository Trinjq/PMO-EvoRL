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

    keys = jax.random.split(jax.random.PRNGKey(1), 32)
    states = jax.jit(jax.vmap(env.reset))(keys)
    next_states = jax.jit(jax.vmap(env.step))(
        states, jnp.zeros((32, env.action_size))
    )
    assert next_states.obs.shape == (32, 17)
    assert next_states.reward.shape == (32, 2)
    print("MOWalker2d single and 32-env checks passed:", next_state.reward)


if __name__ == "__main__":
    main()
