"""Minimal executable contract check for MOWalker2d."""

import jax
import jax.numpy as jnp
import mujoco

from pmo_evorl.envs import MOWalker2d, create_mo_walker2d_env


def main() -> None:
    env = MOWalker2d()
    assert env.mjx_model.impl.value == "jax"
    assert env.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert env.sim_dt == 0.002
    assert env.dt == 0.008
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))

    assert state.obs.shape == (17,)
    assert state.reward.shape == (2,)

    action = jnp.full((env.action_size,), 2.0)
    next_state = step(state, action)
    expected = jnp.array(
        [next_state.metrics["reward_speed"], 5.0 - env.action_size]
    )
    assert next_state.reward.shape == (2,)
    assert bool(jnp.allclose(next_state.reward, expected))

    wrapped_env = create_mo_walker2d_env(num_envs=20)
    states = jax.jit(wrapped_env.reset)(jax.random.PRNGKey(1))
    next_states = jax.jit(wrapped_env.step)(
        states, jnp.zeros((20, env.action_size))
    )
    assert next_states.obs.state.shape == (20, 17)
    assert next_states.obs.preference.shape == (20, 2)
    assert next_states.reward.shape == (20, 2)
    assert bool(jnp.allclose(next_states.obs.preference.sum(-1), 1.0))
    assert bool((next_states.obs.preference[:2, 0] <= 0.1).all())
    assert bool((next_states.obs.preference[-2:, 0] >= 0.901).all())
    assert bool(
        jnp.allclose(next_states.obs.preference, states.obs.preference)
    )
    print("MJX MOWalker2d single and preference-batch checks passed")


if __name__ == "__main__":
    main()
