"""One-iteration GPU smoke check for the complete MO-TD3 workflow."""

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from pmo_evorl.workflow import MOTD3Workflow


def main() -> None:
    config = OmegaConf.create(
        {
            "num_envs": 10,
            "num_eval_envs": 10,
            "num_preference_workers": 10,
            "discount": 0.995,
            "exploration_epsilon": 0.1,
            "policy_noise": 0.2,
            "clip_policy_noise": 0.5,
            "angle_coefficient": 10.0,
            "optimizer": {"lr": 3e-4, "grad_clip_norm": 100.0},
            "replay_buffer_capacity": 256,
            "batch_size": 8,
            "learning_start_timesteps": 20,
            "random_timesteps": 0,
            "her_start_timesteps": 100,
            "her_weight_num": 3,
            "interpolator_eval_episodes": 1,
            "key_update_interval": 5,
            "seed": 1,
            "rollout_length": 1,
            "actor_update_interval": 10,
            "num_updates_per_iter": 1,
            "tau": 0.005,
            "output_dir": "",
            "checkpoint": {"enable": False},
            "save_replay_buffer": False,
            "fold_iters": 1,
            "total_timesteps": 40,
            "eval_interval": 100,
            "eval_episodes": 1,
            "pareto_step_size": 0.5,
            "pareto_eval_batch_size": 3,
        }
    )
    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    state = workflow.init(jax.random.PRNGKey(0))
    assert int(state.replay_buffer_state.buffer_size) == 20
    state = state.replace(
        env_state=state.env_state.replace(
            info=state.env_state.info.replace(
                episode_count=jnp.full(10, 2, dtype=jnp.uint32)
            )
        )
    )

    metrics, state = workflow.step(state)
    assert int(state.metrics.sampled_timesteps) == 30
    assert int(state.replay_buffer_state.buffer_size) == 30
    assert bool(jnp.isfinite(metrics.critic_loss))
    assert bool(jnp.isfinite(metrics.actor_loss))
    assert int(state.agent_state.extra_state.interpolator_updates) == 1
    assert state.agent_state.extra_state.rbf_coefficients.shape == (4, 2)
    state = workflow._maybe_update_interpolator(state)
    assert int(state.agent_state.extra_state.interpolator_updates) == 1
    state = state.replace(
        env_state=state.env_state.replace(
            info=state.env_state.info.replace(
                episode_count=jnp.full(10, 6, dtype=jnp.uint32)
            )
        )
    )
    state = workflow._maybe_update_interpolator(state)
    assert int(state.agent_state.extra_state.interpolator_updates) == 6
    eval_metrics, state = workflow.evaluate(state)
    assert bool(jnp.isfinite(eval_metrics.hypervolume))
    assert bool(jnp.isfinite(eval_metrics.sparsity))
    state = workflow.learn(state)
    assert int(state.metrics.sampled_timesteps) == 40
    print(
        "MO-TD3 workflow check passed:",
        float(metrics.critic_loss),
        float(metrics.actor_loss),
    )


if __name__ == "__main__":
    main()
