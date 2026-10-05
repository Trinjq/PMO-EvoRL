"""Minimal executable contract check for MOWalker2d."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from evorl.envs.mujoco_playground import MjxEnvAdapter
from evorl.envs.wrappers.training_wrapper import EpisodeWrapper, VmapWrapper
from mujoco_playground._src import mjx_env

# Lock the caller-selected backend before evaluate_cpu pins its CLI to CPU.
jax.devices()
from evaluate_cpu import observation, step_environment, validate_model
from pmo_evorl.envs import MOWalker2d, create_mo_walker2d_env
from pmo_evorl.envs.preference import PreferenceConditionedEnv


def main() -> None:
    env = MOWalker2d()
    assert env.mjx_model.impl.value == "jax"
    assert env.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert env.sim_dt == 0.002
    assert env.dt == 0.008

    cpu_model = mujoco.MjModel.from_xml_path(env.xml_path)
    validate_model(cpu_model)
    qpos = np.asarray(cpu_model.qpos0) + np.linspace(-0.001, 0.001, cpu_model.nq)
    qvel = np.linspace(-0.002, 0.002, cpu_model.nv)
    comparison_action = np.linspace(-0.5, 0.5, cpu_model.nu)
    cpu_data = mujoco.MjData(cpu_model)
    cpu_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    mujoco.mj_forward(cpu_model, cpu_data)
    cpu_reward, cpu_done = step_environment(
        cpu_model, cpu_data, comparison_action
    )

    mjx_data = mjx_env.make_data(
        env.mj_model,
        qpos=jnp.asarray(qpos),
        qvel=jnp.asarray(qvel),
        impl=env.mjx_model.impl.value,
    )
    mjx_data = mjx.forward(env.mjx_model, mjx_data)
    zero = jnp.zeros(())
    comparison_state = mjx_env.State(
        mjx_data,
        env._get_obs(mjx_data),
        jnp.zeros(2),
        zero,
        {"reward_speed": zero, "reward_energy": zero},
        {"rng": jax.random.PRNGKey(0)},
    )
    comparison_next = jax.jit(env.step)(
        comparison_state, jnp.asarray(comparison_action)
    )
    assert np.allclose(observation(cpu_data), comparison_next.obs, atol=1e-2)
    assert np.allclose(cpu_reward, comparison_next.reward, atol=1e-3)
    assert cpu_done == bool(comparison_next.done)
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

    eval_env = create_mo_walker2d_env(
        num_envs=3, num_preference_workers=3, autoreset=False
    )
    eval_states = jax.jit(eval_env.reset)(jax.random.PRNGKey(2))
    eval_next = jax.jit(eval_env.step)(
        eval_states, jnp.zeros((3, env.action_size))
    )
    assert eval_next.obs.state.shape == (3, 17)
    assert eval_next.reward.shape == (3, 2)
    serial_env = PreferenceConditionedEnv(
        VmapWrapper(
            EpisodeWrapper(MjxEnvAdapter(MOWalker2d()), episode_length=500),
            num_envs=3,
        ),
        num_envs=3,
        num_workers=3,
    )
    serial_states = jax.jit(serial_env.reset)(jax.random.PRNGKey(2))
    serial_next = jax.jit(serial_env.step)(
        serial_states, jnp.zeros((3, env.action_size))
    )
    print(
        "map/vmap max errors:",
        float(jnp.max(jnp.abs(eval_next.obs.state - serial_next.obs.state))),
        float(jnp.max(jnp.abs(eval_next.reward - serial_next.reward))),
        flush=True,
    )
    assert bool(
        jnp.allclose(eval_next.obs.state, serial_next.obs.state, atol=1e-6)
    )
    assert bool(jnp.allclose(eval_next.reward, serial_next.reward))
    assert bool(jnp.array_equal(eval_next.done, serial_next.done))
    print("MJX MOWalker2d single and preference-batch checks passed")


if __name__ == "__main__":
    main()
